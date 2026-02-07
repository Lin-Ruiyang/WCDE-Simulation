# =========================
# ① Reproducibility settings: must be placed before all imports
# =========================
import os
os.environ["PYTHONHASHSEED"] = "0"       # Fix Python hash randomization
os.environ["OMP_NUM_THREADS"] = "1"      # Disable nondeterministic multithreading
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Normal imports
import random
import numpy as np
import pandas as pd
import networkx as nx
from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling
import ray
import psutil
from itertools import combinations
from sklearn.pipeline import make_pipeline
from pgmpy.base import DAG
from pgmpy.models import DiscreteBayesianNetwork  
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
# Global determinism control
GLOBAL_SEED = 20250902
def set_global_determinism(seed: int):
    random.seed(seed)
    np.random.seed(seed)

# =========================
# Utility: sample and collapse treatment to two levels {0,1}; if insufficient, oversample until filled
# =========================
def sample_with_binary_treatment(model,
                                 n_samples: int,
                                 seed: int,
                                 treatment: str,
                                 level0: int | None = None,
                                 level1: int | None = None,
                                 oversample_factor: int = 20,
                                 max_trials: int = 300,
                                 ensure_balance: bool = True,
                                 min_each: int | None = None):
    """
    Sample from the BN -> stably map all discrete states to integers ->
    keep only two treatment levels and map them to {0,1} ->
    if not enough rows, continue oversampling until reaching n_samples.

    - When level0/level1 are not specified: choose the two most frequent treatment levels
      from the first batch.
    - When ensure_balance=True: both groups must have at least min_each rows
      (default: n_samples//10).
    """
    rng = np.random.RandomState(seed)
    sampler = BayesianModelSampling(model)

    def _map_states(df):
        mapping_dict = {}
        for var in model.nodes():
            states = model.get_cpds(var).state_names[var]
            if set(states) == {'True','False'} or set(states) == {'False','True'}:
                mapping = {'False':0, 'True':1}
            else:
                mapping = {s:i for i,s in enumerate(states)}
            mapping_dict[var] = mapping
        return df.replace(mapping_dict).infer_objects(copy=False)

    collected = []
    picked = None  # (level0, level1)

    if ensure_balance and (min_each is None):
        min_each = max(1, n_samples // 10)

    for trial in range(1, max_trials+1):
        need = n_samples - sum(len(x) for x in collected)
        if need <= 0:
            break

        batch_size = int(max(need * oversample_factor, need + 200))
        batch = sampler.forward_sample(size=batch_size, seed=int(seed + trial))
        batch = _map_states(batch)

        # -----------------------------
        # Determine the two selected categories
        # -----------------------------
        if picked is None:
            if (level0 is None) or (level1 is None):
                vc = batch[treatment].value_counts(dropna=True)
                if len(vc) < 2:
                    continue
                top2 = sorted(vc.index[:2].tolist())
                picked = (top2[0], top2[1])
            else:
                picked = (level0, level1)
            print(f"✅ Treatment {treatment} selected two values: {picked}")

        l0, l1 = picked
        sub = batch[batch[treatment].isin([l0, l1])].copy()
        if sub.empty:
            continue
        sub[treatment] = sub[treatment].map({l0:0, l1:1}).astype(int)

        collected.append(sub)

        if ensure_balance:
            tmp = pd.concat(collected, ignore_index=True)
            c0 = (tmp[treatment] == 0).sum()
            c1 = (tmp[treatment] == 1).sum()
            # If the total has been reached but the groups are still underrepresented, continue sampling until satisfied
            if len(tmp) >= n_samples and (c0 < min_each or c1 < min_each):
                continue

    all_rows = sum(len(x) for x in collected)
    if all_rows < n_samples:
        raise ValueError(
            f"Insufficient samples after binarization: target {n_samples}, "
            f"but only {all_rows} obtained; selected classes: {picked}. "
            f"Consider increasing oversample_factor/max_trials, "
            f"or manually specifying level0/level1."
        )


    data_bin = pd.concat(collected, ignore_index=True)

    if ensure_balance:
        c0 = (data_bin[treatment] == 0).sum()
        c1 = (data_bin[treatment] == 1).sum()
        if (c0 < min_each) or (c1 < min_each):
            raise ValueError(
                f"Total sample size reached but per-class minimum not satisfied: "
                f"min_each={min_each}, count0={c0}, count1={c1}; "
                f"please increase oversample_factor or choose a different pair of classes. "
                f"Selected: {picked}"
            )

    data_bin = data_bin.iloc[:n_samples].reset_index(drop=True)
    return data_bin, picked

def estimate_marginal_effect(data, treatment, outcome, vas, all_mediators):
    """
    Simplified version without z_block / pair_block (stabilized for reproducibility):
    - Precompute lookup tables for E_{Z2}[mu(a,z1,Z2)] and E_{Z1}[mu(a,Z1,z2)]
    - Probabilities computed with groupby(..., sort=True); Z1/Z2 lists and unique keys are sorted
    - Keep the exact AIPW + double-marginalization structure
    - Q model: PolynomialFeatures (degree 2) + Multinomial Logistic; predict -> E[Y|·]
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
    # Require Y to be a continuous integer label (e.g., {0,1} or {0,1,2})
    if not np.all(uniq_y == np.arange(uniq_y.min(), uniq_y.max() + 1)):
        raise ValueError(f"Y must be a continuous integer label (e.g., 0/1/2), found {uniq_y}")

    poly = PolynomialFeatures(degree=2, include_bias=False)
    base_clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        C=1.0,
        max_iter=500,
        n_jobs=1
        # If classes are highly imbalanced, consider adding class_weight="balanced"
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

    # ---------- 3) Empirical joint/marginal probabilities (stable groupby computation) ----------
    tmp = pd.DataFrame({'z1': z1_keys, 'z2': z2_keys, 'A': A})
    # Joint probability
    joint_counts = tmp.groupby(['z1', 'z2'], sort=True).size().div(n).to_dict()
    # Marginal probability
    pZ1 = tmp.groupby('z1', sort=True).size().div(n).to_dict()
    pZ2 = tmp.groupby('z2', sort=True).size().div(n).to_dict()
    # Conditional probability p(A=1|Z)
    p_a1_given_z = tmp.groupby(['z1', 'z2'], sort=True)['A'].mean().to_dict()

    # ---------- 4) Two integration lookup tables: EZ2_mu*, EZ1_mu* ----------
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

    # ---------- 6) Construct phi1 / phi0 for each sample ----------
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
# Ray remote functions: Call binary sampler here
# =========================
@ray.remote
def estimate_wcde_for_seed(vas, model, treatment, outcome, all_mediators, n_samples, seed):
    try:
        import os as _os, random as _random, numpy as _np
        _os.environ["PYTHONHASHSEED"] = "0"; _random.seed(seed); _np.random.seed(seed)

        data, chosen_lvls = sample_with_binary_treatment(
            model, n_samples=n_samples, seed=seed, treatment=treatment,
            oversample_factor=3, max_trials=30,
            ensure_balance=True, min_each=max(1, n_samples//10)
        )
        # print("chosen levels:", chosen_lvls)

        wcde_est = estimate_marginal_effect(data, treatment, outcome, vas, all_mediators)
        return (tuple(sorted(vas)), n_samples, wcde_est)
    except Exception as e:
        return (tuple(sorted(vas)), n_samples, None)


if __name__ == '__main__':
    set_global_determinism(GLOBAL_SEED)

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

    bif_path = r'data/mildew.bif'

    reader = BIFReader(bif_path)
    model_full = reader.get_model()

    treatment = 'mikro_1'
    outcome   = 'meldug_4'

    # ① Build a graph from the full model edges to compute ancestors correctly
    graph_full = nx.DiGraph(model_full.edges())

    # ② Nodes: explicitly sort to avoid nondeterminism from set iteration order
    required_nodes_set = nx.ancestors(graph_full, outcome) | {outcome, treatment}
    required_nodes = sorted(required_nodes_set)
    required_nodes_lookup = set(required_nodes)

    # ③ Edges: explicitly sort to avoid nondeterminism from edge iteration order
    edges_sub = [
        (u, v) for (u, v) in model_full.edges()
        if u in required_nodes_lookup and v in required_nodes_lookup
    ]
    edges_sub = sorted(edges_sub)

    # Build the submodel with deterministic node/edge insertion order
    model_sub = DiscreteBayesianNetwork()
    model_sub.add_nodes_from(required_nodes)
    model_sub.add_edges_from(edges_sub)

    # ④ Add CPDs in the same deterministic node order to avoid internal state-order drift
    for node in required_nodes:
        cpd = model_full.get_cpds(node)
        if cpd is None:
            raise RuntimeError(f"Missing CPD for node={node}")
        model_sub.add_cpds(cpd)

    # ⑤ The submodel must pass the consistency check deterministically
    model_sub.check_model()

    # ⑥ Build the downstream graph using the same sorted edges
    #     (do NOT rely on model_sub.edges(), which may have internal ordering differences)
    graph = nx.DiGraph()
    graph.add_nodes_from(required_nodes)
    graph.add_edges_from(edges_sub)



    all_mediators = list(get_all_mediator_path_nodes(graph, treatment, outcome))
    print("mediators:", all_mediators)

    vas_candidates = ['middel_3','mikro_3','meldug_3','mikro_2','lai_3','lai_2','meldug_2','lai_1']
    vas_candidates = [v for v in vas_candidates if v in model_sub.nodes]
    
    all_sets = find_potential_adjustment_sets(vas_candidates)
    valid_sets = filter_valid_structured_vas(graph, treatment, outcome, all_sets)

    print(f"Number of candidate adjustment sets: {len(all_sets)}")
    print(f"Number of valid adjustment sets: {len(valid_sets)}")

    n_samples_list = [250, 500, 1000, 4000, 10000]
    n_rep = 100

    model_id = ray.put(model_sub)
    mediators_id = ray.put(all_mediators)

    all_tasks = []
    for vas in valid_sets:
        for n_samples in n_samples_list:
            for seed in range(n_rep):
                all_tasks.append(
                    estimate_wcde_for_seed.remote(
                        vas, model_id, treatment, outcome, mediators_id, n_samples, seed
                    )
                )

    raw_results = ray.get(all_tasks)  # (vas_tuple_sorted, n_samples, est or None)

    from collections import defaultdict as _dd
    agg = _dd(list)
    for vas_key, n, est in raw_results:
        if est is not None:
            agg[(vas_key, n)].append(est)

    rows_all = []
    per_n_rows = _dd(list)

    for (vas_key, n), ests in agg.items():
        if len(ests) >= 2:
            variance = float(np.var(ests, ddof=1))
        else:
            variance = float('nan')
        mean_est = float(np.mean(ests)) if len(ests) else float('nan')
        mse = (0.0 if np.isnan(variance) else variance) + (mean_est ** 2) # NOTE: In this experiment the true WCDE is 0
        row = {
            'n_samples': n,
            'adjustment_set': str(list(vas_key)),
            'mean': mean_est,
            'variance': variance,
            'mse': mse,
            'n_success': len(ests),
            'n_rep': n_rep
        }
        rows_all.append(row)
        per_n_rows[n].append(row)

    results_all_df = pd.DataFrame(rows_all)
    if not results_all_df.empty:
        results_all_df = results_all_df.sort_values(by=['n_samples', 'variance'])

    out_dir = 'mildew_results'
    os.makedirs(out_dir, exist_ok=True)

    combined_csv_path = os.path.join(out_dir, f'wcde_estimation_results_{treatment}_{outcome}_ALL_mildew.csv')
    results_all_df.to_csv(combined_csv_path, index=False)
    print(f"Combined results saved to: {combined_csv_path}")

    for n in sorted(per_n_rows.keys()):
        df_n = pd.DataFrame(per_n_rows[n]).sort_values(by='variance')
        csv_path = os.path.join(out_dir, f'wcde_estimation_results_{treatment}_{outcome}_n{n}_mildew.csv')
        df_n.to_csv(csv_path, index=False)
        print(f"Results for n={n} saved to: {csv_path}")

    print("\nResult summary (first 10 rows, sorted by n_samples & variance):")
    print(results_all_df.head(10))
