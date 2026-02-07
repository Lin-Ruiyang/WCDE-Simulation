# =========================
# ① Reproducibility Settings: Must be placed before all imports
# =========================
import os
os.environ["PYTHONHASHSEED"] = "0"       # Fix Python hash order
os.environ["OMP_NUM_THREADS"] = "1"      # Disable nondeterministic multithreading
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Normal imports
import random
import networkx as nx
from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling
from pgmpy.models import BayesianNetwork
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from itertools import combinations
import psutil
import ray
from scipy.special import expit as sigmoid
from scipy.special import logit
from scipy.stats import norm
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from pgmpy.base import DAG
# =========================
# ② Global determinism control
# =========================
GLOBAL_SEED = 20250902  
def set_global_determinism(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def load_dag_from_bif(bif_path):
    reader = BIFReader(bif_path)
    model = reader.get_model()
    return model


def sample_data_from_bif_model(model: BayesianNetwork, n_samples: int, seed: int):
    sampler = BayesianModelSampling(model)
    data = sampler.forward_sample(size=n_samples, seed=seed)
    data = data.replace({'yes': 1, 'no': 0}).infer_objects(copy=False)
    return data



def estimate_marginal_effect(data, treatment, outcome, vas, all_mediators):
    """
    Simplified version without z_block / pair_block:
    - Precompute E_{Z2}[mu(a, z1, Z2)] and E_{Z1}[mu(a, Z1, z2)].
    - Probabilities are computed using sorted groupby for stable results.
    - AIPW structure with two marginalization steps.
    - Q-model: PolynomialFeatures (degree=2) + Multinomial Logistic Regression returning E[Y|·].
    """

    # ---------- 0) Split Z1 / Z2 (stable order) ----------
    mediators_in_vas = vas & set(all_mediators)
    adjusters_in_vas = vas - mediators_in_vas
    Z1_list = sorted(adjusters_in_vas)   # Stable order
    Z2_list = sorted(mediators_in_vas)   # Stable order
    feature_cols = [treatment] + Z1_list + Z2_list

    n = len(data)
    A = data[treatment].to_numpy()
    Y = data[outcome].to_numpy()

    # ---------- 1) Q model (discrete numeric Y: PolynomialFeatures + multinomial logistic; predict→E[Y|·]) ----------
    y_vals = np.asarray(Y)
    uniq_y = np.unique(y_vals)
    if not np.all(uniq_y == np.arange(uniq_y.min(), uniq_y.max() + 1)):
        raise ValueError(f"Y must be a continuous integer label (e.g., 0/1/2), found {uniq_y}")

    poly = PolynomialFeatures(degree=2, include_bias=False)
    base_clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        C=1.0,
        max_iter=500,
        n_jobs=1
    )

    class _QExpectWrapper:
        """Wraps the model to provide a consistent interface: predict returns E[Y|·] (scalar array)"""
        def __init__(self, feature_cols, poly, clf):
            self.feature_cols = feature_cols
            self.pipe = make_pipeline(poly, clf)
            self.scores_ = None

        def fit(self, X, y):
            self.pipe.fit(X[self.feature_cols], y)
            classes_ = self.pipe.named_steps['logisticregression'].classes_
            self.scores_ = np.arange(len(classes_), dtype=float)
            return self

        def predict(self, X):
            proba = self.pipe.predict_proba(X[self.feature_cols])  # (n, K)
            return proba @ self.scores_  # (n,)

    Q_model = _QExpectWrapper(feature_cols, poly, base_clf)
    Q_model.fit(data[feature_cols], data[outcome])

    # ---------- 2) Construct keys & unique sets (sorted) ----------
    def _to_key_tuple(vals):
        return tuple(vals) if len(vals) else ()

    if len(Z1_list):
        z1_keys = pd.Series([_to_key_tuple(data.iloc[i][Z1_list].tolist()) for i in range(n)])
        uniq_z1 = sorted(set(z1_keys.tolist()))
    else:
        z1_keys = pd.Series([()] * n)
        uniq_z1 = [()]

    if len(Z2_list):
        z2_keys = pd.Series([_to_key_tuple(data.iloc[i][Z2_list].tolist()) for i in range(n)])
        uniq_z2 = sorted(set(z2_keys.tolist()))
    else:
        z2_keys = pd.Series([()] * n)
        uniq_z2 = [()]

    # ---------- 3) Empirical joint/marginal probabilities (stable groupby) ----------
    tmp = pd.DataFrame({'z1': z1_keys, 'z2': z2_keys, 'A': A})
    # Joint probabilities
    joint_counts = tmp.groupby(['z1', 'z2'], sort=True).size().div(n).to_dict()
    # Marginal probabilities
    pZ1 = tmp.groupby('z1', sort=True).size().div(n).to_dict()
    pZ2 = tmp.groupby('z2', sort=True).size().div(n).to_dict()
    # Conditional probabilities p(A=1|Z)
    p_a1_given_z = tmp.groupby(['z1', 'z2'], sort=True)['A'].mean().to_dict()

    # ---------- 4) Two “integration” lookup tables: EZ2_mu*, EZ1_mu* ----------
    # 4.1 For each z1: E_{Z2}[mu(a,z1,Z2)]
    EZ2_mu1, EZ2_mu0 = {}, {}
    if len(uniq_z2):
        z2_rows_1 = {z1: pd.DataFrame([[1] + (list(z1) if len(Z1_list) else []) + list(z2) for z2 in uniq_z2],
                                      columns=feature_cols) for z1 in uniq_z1}
        z2_rows_0 = {z1: pd.DataFrame([[0] + (list(z1) if len(Z1_list) else []) + list(z2) for z2 in uniq_z2],
                                      columns=feature_cols) for z1 in uniq_z1}
        for z1 in uniq_z1:
            mu1 = Q_model.predict(z2_rows_1[z1])
            mu0 = Q_model.predict(z2_rows_0[z1])
            s1 = sum(mu1[k] * pZ2[uniq_z2[k]] for k in range(len(uniq_z2)))
            s0 = sum(mu0[k] * pZ2[uniq_z2[k]] for k in range(len(uniq_z2)))
            EZ2_mu1[z1] = float(s1)
            EZ2_mu0[z1] = float(s0)
    else:
        for z1 in uniq_z1:
            X1 = pd.DataFrame([[1] + (list(z1) if len(Z1_list) else [])], columns=feature_cols)
            X0 = pd.DataFrame([[0] + (list(z1) if len(Z1_list) else [])], columns=feature_cols)
            EZ2_mu1[z1] = float(Q_model.predict(X1)[0])
            EZ2_mu0[z1] = float(Q_model.predict(X0)[0])

    # 4.2 For each z2: E_{Z1}[mu(a,Z1,z2)]
    EZ1_mu1, EZ1_mu0 = {}, {}
    if len(uniq_z1):
        z1_rows_1 = {z2: pd.DataFrame([[1] + list(z1) + (list(z2) if len(Z2_list) else []) for z1 in uniq_z1],
                                      columns=feature_cols) for z2 in uniq_z2}
        z1_rows_0 = {z2: pd.DataFrame([[0] + list(z1) + (list(z2) if len(Z2_list) else []) for z1 in uniq_z1],
                                      columns=feature_cols) for z2 in uniq_z2}
        for z2 in uniq_z2:
            mu1 = Q_model.predict(z1_rows_1[z2])
            mu0 = Q_model.predict(z1_rows_0[z2])
            s1 = sum(mu1[k] * pZ1[uniq_z1[k]] for k in range(len(uniq_z1)))
            s0 = sum(mu0[k] * pZ1[uniq_z1[k]] for k in range(len(uniq_z1)))
            EZ1_mu1[z2] = float(s1)
            EZ1_mu0[z2] = float(s0)
    else:
        for z2 in uniq_z2:
            X1 = pd.DataFrame([[1] + (list(z2) if len(Z2_list) else [])], columns=feature_cols)
            X0 = pd.DataFrame([[0] + (list(z2) if len(Z2_list) else [])], columns=feature_cols)
            EZ1_mu1[z2] = float(Q_model.predict(X1)[0])
            EZ1_mu0[z2] = float(Q_model.predict(X0)[0])

    # ---------- 5) Predict μ1_obs / μ0_obs for all samples at once ----------
    X1_full = pd.DataFrame([[1] + (list(z1_keys.iat[i]) if len(Z1_list) else []) + (list(z2_keys.iat[i]) if len(Z2_list) else [])
                            for i in range(n)], columns=feature_cols)
    X0_full = pd.DataFrame([[0] + (list(z1_keys.iat[i]) if len(Z1_list) else []) + (list(z2_keys.iat[i]) if len(Z2_list) else [])
                            for i in range(n)], columns=feature_cols)
    mu1_obs = Q_model.predict(X1_full)
    mu0_obs = Q_model.predict(X0_full)

    # ---------- 6) Assemble phi1 / phi0 per sample ----------
    aipw1 = np.empty(n, dtype=float)
    aipw0 = np.empty(n, dtype=float)

    for i in range(n):
        z1 = z1_keys.iat[i]
        z2 = z2_keys.iat[i]
        a_i, y_i = A[i], Y[i]

        p_z1 = pZ1[z1] if len(Z1_list) else 1.0
        p_z2 = pZ2[z2] if len(Z2_list) else 1.0
        p_z  = p_z1 * p_z2
        p_a1 = p_a1_given_z.get((z1, z2), 0.0)

        joint_p = joint_counts.get((z1, z2), 0.0)
        p_joint_1 = joint_p * p_a1
        p_joint_0 = joint_p * (1.0 - p_a1)

        # phi1
        ipw1 = (p_z / p_joint_1) * (y_i - mu1_obs[i]) if (a_i == 1 and p_joint_1 > 0) else 0.0
        aipw1[i] = ipw1 + EZ2_mu1[z1] + EZ1_mu1[z2]

        # phi0
        ipw0 = (p_z / p_joint_0) * (y_i - mu0_obs[i]) if (a_i == 0 and p_joint_0 > 0) else 0.0
        aipw0[i] = ipw0 + EZ2_mu0[z1] + EZ1_mu0[z2]

    wcde = 0.5 * (aipw1.mean() - aipw0.mean())
    return float(wcde)


# =========================
# Adjustment set search related
# =========================
def get_all_mediator_path_nodes(graph, treatment, outcome):
    mediators = set()
    for path in nx.all_simple_paths(graph, source=treatment, target=outcome):
        mediators.update(path[1:-1])
    return mediators

# ========= Helper functions for the (generalized) adjustment criterion on DAGs =========

def _descendants(G: nx.DiGraph, start):
    """
    Return the set of descendants of `start` (including `start` itself) in a DAG.
    """
    dec = set()
    stack = [start]
    while stack:
        u = stack.pop()
        if u in dec:
            continue
        dec.add(u)
        stack.extend(G.successors(u))
    return dec

def _has_directed_path_to_any(G: nx.DiGraph, src, targets):
    """
    Return True if there exists a directed path from `src` to any node in `targets`.
    """
    targets = set(targets)
    if src in targets:
        return True
    visited = {src}
    stack = [src]
    while stack:
        u = stack.pop()
        for v in G.successors(u):
            if v in targets:
                return True
            if v not in visited:
                visited.add(v)
                stack.append(v)
    return False

def _forbidden_set_DAG(G: nx.DiGraph, Xset, Yset):
    """
    DAG version of the forbidden set Forb(X, Y, G).

    Definition (DAG): Forb(X, Y, G) contains every node that lies on any proper
    causal path from X to Y (excluding the start node in X) and all descendants of such nodes.
    Intuition: mediators and their descendants (except X itself) must not be adjusted for.
    """
    Xset = set(Xset); Yset = set(Yset)
    forb = set()
    for x in Xset:
        for y in Yset:
            try:
                # Enumerate simple directed paths from x to y
                for path in nx.all_simple_paths(G, x, y):
                    # Exclude the start node x; collect each node and all its descendants
                    for w in path[1:]:
                        forb |= _descendants(G, w)
            except nx.NetworkXNoPath:
                continue
    # Exclude X itself per definition (the "some W not in X" clause)
    forb -= Xset
    return forb

def _proper_backdoor_graph_DAG(G: nx.DiGraph, Xset, Yset):
    """
    Proper back-door graph for a DAG:

    Remove every "first edge" X->v that lies on some directed path X -> ... -> Y.
    After this removal, d-separation with respect to Z checks whether Z blocks all
    non-causal paths between X and Y (equivalent to the blocking condition).
    """
    Xset = set(Xset); Yset = set(Yset)
    H = G.copy()
    for x in Xset:
        for v in list(G.successors(x)):
            # If v can still reach Y via a directed path, X->v is a first edge on an X->...->Y path
            if _has_directed_path_to_any(G, v, Yset):
                if H.has_edge(x, v):
                    H.remove_edge(x, v)
    return H

# -------------------------
# Precise backdoor checker for c2 
# -------------------------
def _is_collider(G: nx.DiGraph, a, b, c):
    """
    On path a - b - c, b is a collider iff a->b and c->b are both edges in G.
    """
    return G.has_edge(a, b) and G.has_edge(c, b)

def _enumerate_backdoor_paths(G: nx.DiGraph, x, y):
    """
    Enumerate all undirected simple paths from x to y whose first step is INTO x (i.e., path[1] -> path[0]).
    These are candidate backdoor paths for x.
    """
    UG = G.to_undirected()
    try:
        for path in nx.all_simple_paths(UG, source=x, target=y):
            if len(path) >= 2 and G.has_edge(path[1], path[0]):
                yield path
    except nx.NetworkXNoPath:
        return

def _is_path_active_under_Zc(G: nx.DiGraph, path, Zc_set, DescZc_set):
    """
    D-separation rule on the given path with respect to Zc:
      - For every internal node b:
          * if b is a collider (→b←), the path is OPEN at b iff b in Zc or b in Desc(Zc)
          * else (non-collider), the path is OPEN at b iff b not in Zc
      - The path is active iff it's open at all internal nodes.
    """
    if len(path) <= 2:
        return True
    for i in range(1, len(path)-1):
        a, b, c = path[i-1], path[i], path[i+1]
        if _is_collider(G, a, b, c):
            if (b not in Zc_set) and (b not in DescZc_set):
                return False  # collider closed
        else:
            if b in Zc_set:
                return False  # non-collider blocked
    return True

def _is_d_separated_backdoor_precise(G: nx.DiGraph, Xset, Yset, Zc_set, verbose=False):
    """
    Precise backdoor blocking check on proper back-door graph G:
      - Enumerate backdoor paths (first step into t ∈ Xset)
      - Apply collider/non-collider opening rules w.r.t. Zc (colliders can open via descendants in Zc)
    Returns:
      (is_blocked: bool, details: dict)
    """
    blocked_all = True
    details = {}

    # Precompute Desc(Zc)
    DescZc = set()
    for z in Zc_set:
        DescZc |= _descendants(G, z)

    y = list(Yset)[0]
    for x in Xset:
        paths_info = []
        active_found = False
        for path in _enumerate_backdoor_paths(G, x, y):
            is_active = _is_path_active_under_Zc(G, path, Zc_set, DescZc)
            paths_info.append({"path": path, "active": is_active})
            if is_active:
                active_found = True
        if active_found:
            blocked_all = False
        details[x] = paths_info

    if verbose:
        print(f"  c2/backdoor check on proper back-door graph:")
        for x, plist in details.items():
            print(f"    - From t={x}:")
            if not plist:
                print("        (no backdoor paths)")
            for item in plist:
                pstr = " -> ".join(item["path"])
                print(f"        path [{pstr}] | active={item['active']}")
        print(f"  c2 result: {'All blocked (True)' if blocked_all else 'Unblocked backdoor exists (False)'}")

    return blocked_all, details

def _is_d_separated_pgmpy(graph, X, Y, Z):
    """
    Check whether X and Y are d-separated given Z in a DAG.

    Uses pgmpy's active_trail_nodes() method:
      - A path is active (open) if it is not blocked by Z.
      - If no active trail connects X and Y, then X ⟂ Y | Z (d-separated).

    Parameters
    ----------
    graph : nx.DiGraph
        Original causal graph.
    X, Y, Z : str, list, or set
        Node(s) representing the conditioning structure.

    Returns
    -------
    bool
        True  if X and Y are d-separated given Z,
        False if they are d-connected.
    """
    dag = DAG()
    dag.add_nodes_from(graph.nodes())
    dag.add_edges_from(graph.edges())

    # Normalize inputs to lists
    def _as_list(v):
        if v is None:
            return []
        if isinstance(v, (str, int)):
            return [v]
        return list(v)

    X = _as_list(X)
    Y = _as_list(Y)
    Z = set(_as_list(Z))

    # Collect all nodes reachable from X via active trails under observed Z
    reachable = set()
    for x in X:
        reachable |= set(dag.active_trail_nodes(x, observed=Z))

    # If Y has no intersection with reachable, X ⟂ Y | Z
    return set(Y).isdisjoint(reachable)

def is_valid_structured_vas(graph, treatment, outcome, adj_set, mediators):
    """
    Sufficient Condition [VAS]:
    (1) All backdoor paths between A and Y are blocked by Z;
    (2) All backdoor paths between M and Y are blocked by Z;
    (3) All directed A→Y paths that pass through M are blocked by Z;
    (4) (M' \ Z1) ⟂ (Pa(Y) \ M') | Z1, where M' = Pa(Y)∩M, Z1 = Z∩M.
    Output: True / False
    """
    Z = set(adj_set)
    M = set(mediators)

    # ----------  c1/c2: Adjustment criterion for X={A}∪Z1 (NOT {A}∪M),
    #                       enforced only on the "confounder part" Z_c = Z \ M ----------
    Z1 = Z & M               # mediators that are actually included in Z
    Zc = Z - M               # the confounder part to be checked by the adjustment criterion
    T  = {treatment} | Z1    # X = {A} ∪ Z1
    Yset = {outcome}         # Y as a singleton set

    # (AC-1) Forbidden condition: Zc ∩ Forb(X, Y, G) must be empty
    forb = _forbidden_set_DAG(graph, T, Yset)
    c1 = len(Zc & forb) == 0

    # (AC-2) Blocking condition (precise backdoor check on proper back-door graph)
    Gpbd = _proper_backdoor_graph_DAG(graph, T, Yset)
    c2, _c2_details = _is_d_separated_backdoor_precise(Gpbd, T, Yset, Zc_set=Zc, verbose=False)

    # ---------- c3 ----------
    try:
        dir_paths = list(nx.all_simple_paths(graph, source=treatment, target=outcome))
    except Exception:
        dir_paths = []
    med_paths = [p for p in dir_paths if any(n in M for n in p[1:-1])]
    c3 = all(any(v in Z1 for v in p[1:-1]) for p in med_paths) if med_paths else True

    # ---------- c4 ----------
    try:
        pa_y = set(graph.predecessors(outcome))
    except Exception:
        pa_y = set()
    M_prime = pa_y & M
    # Z1 is already defined above; the next line is optional (kept for clarity):
    Z1 = Z & M
    N_nonmed_parents = pa_y - M_prime

    X = M_prime - Z1
    Yset = set(N_nonmed_parents) | {treatment}
    if len(X) == 0 or len(Yset) == 0:
        c4 = True
    else:
        c4 = _is_d_separated_pgmpy(graph, X, Yset, Z1)

    return c1 and c2 and c3 and c4



def find_potential_adjustment_sets(candidate_vars):
    return [set(c) for k in range(1, len(candidate_vars)+1) for c in combinations(candidate_vars, k)]

def filter_valid_structured_vas(graph, treatment, outcome, all_sets):
    med = get_all_mediator_path_nodes(graph, treatment, outcome)
    return [s for s in all_sets if is_valid_structured_vas(graph, treatment, outcome, s, med)]

# =========================
# ③ Set seeds again inside Ray remote functions (and return n_samples)
# =========================
@ray.remote
def estimate_single_seed(vas, model, treatment, outcome, all_mediators, n_samples, seed):
    try:
        import os as _os, random as _random
        import numpy as _np
        _os.environ["PYTHONHASHSEED"] = "0"
        _random.seed(seed)
        _np.random.seed(seed)

        data = sample_data_from_bif_model(model, n_samples=n_samples, seed=seed)
        est = estimate_marginal_effect(data, treatment, outcome, vas, all_mediators)
        return (str(sorted(vas)), int(n_samples), est)
    except Exception as e:
        return (str(sorted(vas)), int(n_samples), None)




if __name__ == '__main__':
    # Set the random seed for the main process
    set_global_determinism(GLOBAL_SEED)

    # ============ 1) Ray & Monitoring ============
    # ② Pass the same environment variables to workers when initializing Ray
    ray.init(
        num_cpus=psutil.cpu_count(logical=False),
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
        runtime_env={
            "env_vars": {
                "PYTHONHASHSEED": "0",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        }
    )

    # ============ 2) Configuration ============
    bif_path = r'data/asia.bif'

    model = load_dag_from_bif(bif_path)
    
    graph = nx.DiGraph()
    graph.add_nodes_from(model.nodes())
    graph.add_edges_from(model.edges())

    treatment = 'either'
    outcome = 'dysp'
    all_mediators = list(get_all_mediator_path_nodes(graph, treatment, outcome))

    all_sets = find_potential_adjustment_sets(graph, treatment, outcome)
    valid_sets = filter_valid_structured_vas(graph, treatment, outcome, all_sets)
    valid_sets = [s for s in valid_sets if 'xray' not in s and 'tub' not in s and 'asia' not in s]

    print(f" {len(valid_sets)} VASs found:")

    n_samples_list = [250, 500, 1000, 4000, 10000]
    n_rep = 100

    # ============ 3) Submit tasks ============
    futures = []
    for n_samples in n_samples_list:
        for vas in valid_sets:
            for seed in range(n_rep):
                futures.append(
                    estimate_single_seed.remote(vas, model, treatment, outcome, all_mediators, n_samples, seed)
                )
    results = ray.get(futures)

    # ============ 4) Aggregate & Save (by (VAS, n_samples)) ============
    from collections import defaultdict as _dd
    # key: (adjustment_set, n_samples) -> list(estimates)
    result_dict = _dd(list)
    for vas_str, n_samples_ret, est in results:
        if est is not None:
            result_dict[(vas_str, int(n_samples_ret))].append(est)

    result_data_all = []
    per_n_rows = _dd(list)

    for (vas_str, n_samp), values in result_dict.items():
        if not values:
            continue
        variance = float(np.var(values, ddof=1)) if len(values) >= 2 else float('nan')
        mean_est = float(np.mean(values))
        # 0.375 is the theoretical true value of WCDE (computed analytically from the DAG)
        mse = (variance if not np.isnan(variance) else 0.0) + (mean_est - 0.375) ** 2
        row = {
            'n_samples': n_samp,
            'adjustment_set': vas_str,
            'mean': mean_est,
            'variance': variance,
            'mse': mse,
            'n_success': len(values),
            'n_rep': n_rep
        }
        result_data_all.append(row)
        per_n_rows[n_samp].append(row)

    # Merge table: sort by (n_samples, variance)
    results_all_df = pd.DataFrame(result_data_all)
    if not results_all_df.empty:
        results_all_df = results_all_df.sort_values(by=['n_samples', 'variance'])

    # Output directory
    out_dir = 'asia_results'
    os.makedirs(out_dir, exist_ok=True)

    # 1) Merge CSV (including all n_samples)
    combined_csv_path = os.path.join(out_dir, 'asia_wcde_estimation_results_ALL.csv')
    results_all_df.to_csv(combined_csv_path, index=False)
    print(f"Combined results saved to: {combined_csv_path}")

    # 2) Individual CSV for each n_samples
    for n_samp, rows in per_n_rows.items():
        df_n = pd.DataFrame(rows).sort_values(by='variance')
        csv_path = os.path.join(out_dir, f'asia_wcde_estimation_results_n{n_samp}.csv')
        df_n.to_csv(csv_path, index=False)
        print(f"Results for n={n_samp} saved to: {csv_path}")

    # Print summary
    print("\nResults summary (top 10 rows, sorted by n_samples & variance):")
    print(results_all_df.head(10))

    ray.shutdown()
