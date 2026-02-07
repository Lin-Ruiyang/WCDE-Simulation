# ===== Deterministic settings: MUST be at very top (before numpy/sklearn import) =====
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
# For a more reproducible MKL code path (avoid different CPU features selecting different kernels)
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
os.environ.setdefault("MKL_DEBUG_CPU_TYPE", "5")

try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)  # Limit to 1 thread globally to avoid non-exchangeability due to multi-threading
except Exception:
    pass

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import LinearRegression, LogisticRegression
from itertools import combinations
import psutil
import ray
import time
from scipy.special import expit as sigmoid
import random
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from pgmpy.base import DAG
# ========================= Utility: enforce determinism inside Worker/main process =========================
def _enforce_worker_determinism(seed: int = 0):
    """Enforce single-threading and random seed in the current process; must be called again at the top of Ray worker."""
    try:
        from threadpoolctl import threadpool_limits as _tpl
        _tpl(1)
    except Exception:
        pass
    import os as _os
    _os.environ["OMP_NUM_THREADS"] = "1"
    _os.environ["MKL_NUM_THREADS"] = "1"
    _os.environ["OPENBLAS_NUM_THREADS"] = "1"
    _os.environ["NUMEXPR_NUM_THREADS"] = "1"
    _os.environ["PYTHONHASHSEED"] = "0"
    _os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
    _os.environ["MKL_CBWR"] = "COMPATIBLE"
    _os.environ["MKL_DEBUG_CPU_TYPE"] = "5"

    import numpy as _np, random as _random
    _np.random.seed(seed)
    _random.seed(seed)

# ========================= Base DAG and data generation =========================
def generate_dag_structure():
    G = nx.DiGraph()
    G.add_edges_from([
        ('A', 'G1'),
        ('G1', 'Y'),
        ('B2', 'A'),
        ('B2', 'G1'),
        ('B2', 'G2'),
        ('G2', 'Y'),
        ('A', 'Y')
    ])
    return G

def generate_random_coefficients(G=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    coefficients = {}
    edges = ['A_G1', 'G1_Y', 'B2_A', 'B2_G1', 'B2_G2', 'G2_Y', 'A_Y']
    for edge in edges:
        if np.random.rand() < 0.5:
            coefficients[edge] = np.random.uniform(-1.5, -0.5)
        else:
            coefficients[edge] = np.random.uniform(0.5, 1.5)
    return coefficients

def generate_data_from_dag(G, coefficients, n_samples=1000, seed=0, nonlinear_func=None):
    """
    All nodes with parents use the same selected_func:
      - Continuous node: value = f(coeff_i * parent_i, ...) + N(0, 0.25^2)
      - Binary node A:   A ~ Bernoulli(sigmoid(f(coeff_i * parent_i, ...)))
    B2 is exogenous ~ N(0,1).
    """
    np.random.seed(seed)
    data = pd.DataFrame(index=range(n_samples))
    topo_order = list(nx.topological_sort(G))

    # Exogenous variable
    data['B2'] = np.random.normal(0, 1, n_samples)

    selected_func = nonlinear_func
    NOISE_SD = 0.25

    for node in topo_order:
        if node == 'B2':
            continue

        parents = list(G.predecessors(node))
        if not parents:
            data[node] = np.random.normal(0, 1, n_samples)
            continue

        # terms: (num_parents, n_samples)
        terms = np.vstack([
            coefficients.get(f'{p}_{node}', 1.0) * data[p].to_numpy()
            for p in parents
        ])
        latent = selected_func(terms)  # (n_samples,)

        if node == 'A':
            p = np.clip(sigmoid(latent), 1e-6, 1-1e-6)
            data[node] = np.random.binomial(1, p, n_samples)
        else:
            noise = np.random.normal(0, NOISE_SD, n_samples)
            data[node] = latent + noise

    return data, selected_func

# ============ 3. Theoretical truth: Monte Carlo approximation for (marginal x marginal) target ============

def compute_theoretical_wcde_montecarlo(coeffs, nonlinear_func,
                                        n_g1=4000, n_b2g2=4000, seed=0):
    """
    Target (marginal x marginal):
      T(a) = E_{G1 ~ P_nat(G1)}  E_{(B2,G2) ~ P_nat(B2,G2)} [ Y(a, G1, G2) ].
    Return: T(1) - T(0)  (not multiplied by 0.5)

    DAG (Figure 2): B2->A,G1,G2; A->G1,Y; G1->Y; G2->Y; A->Y
    """
    np.random.seed(seed)

    sd_B2 = 1.0
    sd_G2 = 0.25
    sd_G1 = 0.25

    # ------------------------------------------------------------
    # 1) Sample G1 from its NATURAL MARGINAL P_nat(G1)
    #    by simulating (B2 -> A -> G1) under the natural mechanism,
    #    then discarding (A,B2).
    # ------------------------------------------------------------
    b2_for_g1 = np.random.normal(0, sd_B2, n_g1)

    # A | B2 : sigmoid( f([B2_A * B2]) )
    eta_terms  = np.vstack([coeffs['B2_A'] * b2_for_g1])   
    eta_latent = nonlinear_func(eta_terms)                
    p_a1 = np.clip(sigmoid(eta_latent), 1e-6, 1 - 1e-6)
    A_nat = (np.random.rand(n_g1) < p_a1).astype(float)

    # G1 | (A, B2) : f([B2_G1 * B2, A_G1 * A]) + noise
    g1_terms  = np.vstack([
        coeffs['B2_G1'] * b2_for_g1,
        coeffs['A_G1']  * A_nat
    ])  # (2, n_g1)
    g1_latent = nonlinear_func(g1_terms)                 
    g1_samples = np.random.normal(g1_latent, sd_G1)       

    # ------------------------------------------------------------
    # 2) Sample (B2,G2) from its NATURAL JOINT P_nat(B2,G2)
    #    by simulating B2 then G2|B2, and keep both.
    # ------------------------------------------------------------
    b2_pool = np.random.normal(0, sd_B2, n_b2g2)

    # G2 | B2 : f([B2_G2 * B2]) + noise
    g2_terms  = np.vstack([coeffs['B2_G2'] * b2_pool])     
    g2_mean   = nonlinear_func(g2_terms)                   
    g2_pool   = np.random.normal(g2_mean, sd_G2)            

    # ------------------------------------------------------------
    # For fixed g1, integrate over (B2,G2) marginal (here only G2 enters Y)
    # ------------------------------------------------------------
    def inner_EY_given_a_g1(a, g1):
        g2 = g2_pool
        n = len(g2)

        g1_array = np.full(n, float(g1))
        a_array  = np.full(n, float(a))

        y_terms = np.vstack([
            coeffs['G1_Y'] * g1_array,
            coeffs['G2_Y'] * g2,
            coeffs['A_Y']  * a_array
        ])  # (3, n)
        y = nonlinear_func(y_terms)  
        return float(np.mean(y))

    def T(a):
        vals = [inner_EY_given_a_g1(a, g1) for g1 in g1_samples]
        return float(np.mean(vals))

    return (T(1) - T(0))


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

    # ---------- c1/c2: Adjustment criterion for X={A}∪Z1 (NOT {A}∪M),
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

def find_potential_adjustment_sets(G, treatment, outcome):
    nodes = set(G.nodes()) - {treatment, outcome}
    return [set(c) for k in range(1, len(nodes)+1) for c in combinations(nodes, k)]

def filter_valid_structured_vas(graph, treatment, outcome, all_sets):
    med = get_all_mediator_path_nodes(graph, treatment, outcome)
    return [s for s in all_sets if is_valid_structured_vas(graph, treatment, outcome, s, med)]


# ========================= Estimator (AIPW + two marginal integrations) =========================
def _unique_and_weights(keys_series: pd.Series):
    """Stable key order + one-to-one with weights; avoid default sorting that can differ across platforms when ties occur."""
    if not isinstance(keys_series, pd.Series) or len(keys_series) == 0:
        return [()], {(): 1.0}
    vc = keys_series.value_counts(dropna=False, sort=False)
    def _safe_tuple(x):
        return x if isinstance(x, tuple) else (x,)
    # Stable ordering: by length, then lexicographically element-wise (no forced float casts)
    keys_sorted = sorted(vc.index, key=lambda t: (_safe_tuple(t).__len__(), _safe_tuple(t)))
    total = float(vc.sum())
    weights = {k: float(vc[k] / total) for k in keys_sorted}
    return keys_sorted, weights

def estimate_marginal_effect(data, treatment, outcome, vas, all_mediators,
                             z_block=8192, pair_block=100000, rng_seed=0):
    # Determinism hook (safe even without Ray)
    _enforce_worker_determinism(seed=777 + rng_seed)

    mediators_in_vas = set(vas) & set(all_mediators)
    adjusters_in_vas = set(vas) - mediators_in_vas
    Z1_list = list(mediators_in_vas)   # mediator
    Z2_list = list(adjusters_in_vas)   # adjuster/confounder
    feature_cols = [treatment] + Z1_list + Z2_list

    n = len(data)
    A = data[treatment].to_numpy().astype(int)
    Y = data[outcome].to_numpy()

    # 1) Q model: spline + linear regression (force float64 to avoid dtype path divergence across platforms)
    degree, n_knots = 5, 10
    spline_transformer = ColumnTransformer([
        (col, SplineTransformer(degree=degree, n_knots=n_knots, include_bias=False), [col])
        for col in feature_cols
    ])
    Q_model = make_pipeline(spline_transformer, LinearRegression())
    Q_model.fit(data[feature_cols].astype(np.float64), data[outcome].astype(np.float64))

    # 2) (Z1,Z2) -> tuple keys; unique sets (stable order)
    def _to_tuple_frame(df, cols):
        if not cols:
            return pd.Series([()], index=df.index)
        return pd.Series(list(df[cols].itertuples(index=False, name=None)), index=df.index)

    z1_keys = _to_tuple_frame(data, Z1_list)
    z2_keys = _to_tuple_frame(data, Z2_list)

    uniq_z1, pZ1_w = _unique_and_weights(z1_keys) if Z1_list else ([()], {(): 1.0})
    uniq_z2, pZ2_w = _unique_and_weights(z2_keys) if Z2_list else ([()], {(): 1.0})

    # 3) Blocked computation of E_{Z2}[mu] and E_{Z1}[mu]
    EZ2_mu1, EZ2_mu0 = {}, {}
    for i in range(0, len(uniq_z1)):
        z1 = uniq_z1[i]
        sum1 = 0.0; sum0 = 0.0
        for j in range(0, len(uniq_z2), z_block):
            z2_chunk = uniq_z2[j:j+z_block]
            rows_1 = [[1] + (list(z1) if Z1_list else []) + (list(z2) if Z2_list else []) for z2 in z2_chunk]
            rows_0 = [[0] + (list(z1) if Z1_list else []) + (list(z2) if Z2_list else []) for z2 in z2_chunk]
            X1_blk = np.asarray(rows_1, dtype=np.float64)
            X0_blk = np.asarray(rows_0, dtype=np.float64)
            mu1_blk = Q_model.predict(pd.DataFrame(X1_blk, columns=feature_cols, dtype=np.float64))
            mu0_blk = Q_model.predict(pd.DataFrame(X0_blk, columns=feature_cols, dtype=np.float64))
            for k, z2 in enumerate(z2_chunk):
                w2 = pZ2_w[z2]
                sum1 += float(mu1_blk[k]) * w2
                sum0 += float(mu0_blk[k]) * w2
            del rows_1, rows_0, X1_blk, X0_blk, mu1_blk, mu0_blk
        EZ2_mu1[z1] = float(sum1)
        EZ2_mu0[z1] = float(sum0)

    EZ1_mu1, EZ1_mu0 = {}, {}
    for j in range(0, len(uniq_z2)):
        z2 = uniq_z2[j]
        sum1 = 0.0; sum0 = 0.0
        for i in range(0, len(uniq_z1), z_block):
            z1_chunk = uniq_z1[i:i+z_block]
            rows_1 = [[1] + (list(z1) if Z1_list else []) + (list(z2) if Z2_list else []) for z1 in z1_chunk]
            rows_0 = [[0] + (list(z1) if Z1_list else []) + (list(z2) if Z2_list else []) for z1 in z1_chunk]
            X1_blk = np.asarray(rows_1, dtype=np.float64)
            X0_blk = np.asarray(rows_0, dtype=np.float64)
            mu1_blk = Q_model.predict(pd.DataFrame(X1_blk, columns=feature_cols, dtype=np.float64))
            mu0_blk = Q_model.predict(pd.DataFrame(X0_blk, columns=feature_cols, dtype=np.float64))
            for k, z1 in enumerate(z1_chunk):
                w1 = pZ1_w[z1]
                sum1 += float(mu1_blk[k]) * w1
                sum0 += float(mu0_blk[k]) * w1
            del rows_1, rows_0, X1_blk, X0_blk, mu1_blk, mu0_blk
        EZ1_mu1[z2] = float(sum1)
        EZ1_mu0[z2] = float(sum0)

    # 4) Sample-level μ1_obs / μ0_obs (for IPW residuals)
    mu1_obs = np.empty(n, dtype=np.float64)
    mu0_obs = np.empty(n, dtype=np.float64)
    for s in range(0, n, pair_block):
        t = min(s + pair_block, n)
        z1_batch = z1_keys.iloc[s:t].tolist()
        z2_batch = z2_keys.iloc[s:t].tolist()
        rows_1 = [[1] + (list(z1) if Z1_list else []) + (list(z2) if Z2_list else []) for z1, z2 in zip(z1_batch, z2_batch)]
        rows_0 = [[0] + (list(z1) if Z1_list else []) + (list(z2) if Z2_list else []) for z1, z2 in zip(z1_batch, z2_batch)]
        X1_blk = np.asarray(rows_1, dtype=np.float64)
        X0_blk = np.asarray(rows_0, dtype=np.float64)
        mu1_obs[s:t] = Q_model.predict(pd.DataFrame(X1_blk, columns=feature_cols, dtype=np.float64))
        mu0_obs[s:t] = Q_model.predict(pd.DataFrame(X0_blk, columns=feature_cols, dtype=np.float64))
        del rows_1, rows_0, X1_blk, X0_blk

    # 5) Discriminative density ratio + propensity score (unified random_state, float64)
    Z1_arr = data[Z1_list].to_numpy(dtype=np.float64) if Z1_list else np.zeros((n, 0), dtype=np.float64)
    Z2_arr = data[Z2_list].to_numpy(dtype=np.float64) if Z2_list else np.zeros((n, 0), dtype=np.float64)

    def _log_density_ratio(Z1_arr, Z2_arr):
        if Z1_arr.shape[1] + Z2_arr.shape[1] == 0:
            return np.zeros(Z1_arr.shape[0], dtype=float)
        n_local = Z1_arr.shape[0]
        rng = np.random.default_rng(rng_seed + 7919)
        perm = rng.permutation(n_local)
        Z_joint = np.hstack([Z1_arr, Z2_arr]).astype(np.float64, copy=False)
        Z_prod  = np.hstack([Z1_arr, Z2_arr[perm]]).astype(np.float64, copy=False)
        X_cls = np.vstack([Z_joint, Z_prod]).astype(np.float64, copy=False)
        y_cls = np.hstack([np.zeros(n_local, dtype=int), np.ones(n_local, dtype=int)])

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_cls)
        clf = LogisticRegression(
            max_iter=1000, class_weight="balanced", solver="lbfgs",
            n_jobs=1, random_state=0
        )
        clf.fit(Xs, y_cls)
        s = clf.predict_proba(Xs[:n_local])[:, 1]
        s = np.clip(s, 1e-6, 1-1e-6)
        return np.log(s) - np.log(1.0 - s)

    log_r = _log_density_ratio(Z1_arr, Z2_arr)

    Z_arr = np.hstack([Z1_arr, Z2_arr]) if (Z1_arr.shape[1] + Z2_arr.shape[1] > 0) else np.zeros((n, 0), dtype=np.float64)
    if Z_arr.shape[1] > 0:
        clf_e = LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=1, random_state=0)
        clf_e.fit(Z_arr, A)
        e_hat = clf_e.predict_proba(Z_arr)[:, 1]
    else:
        e_hat = np.full(n, A.mean(), dtype=np.float64)
    e_hat = np.clip(e_hat, 1e-3, 1-1e-3)
    pa_obs = np.where(A == 1, e_hat, 1 - e_hat)

    LOGW_MIN, LOGW_MAX = -30.0, 30.0
    log_w = np.clip(log_r - np.log(pa_obs), LOGW_MIN, LOGW_MAX)
    w = np.exp(log_w)

    z1_vals = z1_keys.tolist()
    z2_vals = z2_keys.tolist()
    phi1 = np.where(A == 1, w * (Y - mu1_obs), 0.0) \
           + np.array([EZ2_mu1[z1] for z1 in z1_vals]) \
           + np.array([EZ1_mu1[z2] for z2 in z2_vals])
    phi0 = np.where(A == 0, w * (Y - mu0_obs), 0.0) \
           + np.array([EZ2_mu0[z1] for z1 in z1_vals]) \
           + np.array([EZ1_mu0[z2] for z2 in z2_vals])

    wcde = 0.5 * (phi1.mean() - phi0.mean())
    return float(wcde)

# ========================= Monte Carlo repeats (fixed coefficients) =========================
def montecarlo_var_fixed_coeff(n_rep, n_sample, treatment, outcome, vas, all_mediators, coeffs_and_func, true_wcde, seed_base=0):
    coeffs, nonlinear_func = coeffs_and_func
    ests = []
    G = generate_dag_structure()
    for i in range(n_rep):
        data, _ = generate_data_from_dag(
            G, coeffs, n_samples=n_sample,
            seed=seed_base + i, nonlinear_func=nonlinear_func
        )
        est = estimate_marginal_effect(data, treatment, outcome, vas, all_mediators, rng_seed=seed_base + i)
        ests.append(est)
    variance = np.var(ests, ddof=1)
    mean_est = np.mean(ests)
    bias_sq = (mean_est - true_wcde) ** 2
    mse = bias_sq + variance
    return variance, mse, ests

# ========================= Ray: parallel generation of coefficient sets + ground truth =========================
@ray.remote
def generate_single_coeff_tuple(G_edges, seed, nonlinear_funcs):
    _enforce_worker_determinism(seed=seed)
    np.random.seed(seed)
    rnd = random.Random(seed)
    G = nx.DiGraph(); G.add_edges_from(G_edges)
    coeffs = generate_random_coefficients(G, seed=seed)
    selected_func = rnd.choice(nonlinear_funcs)
    true_wcde = compute_theoretical_wcde_montecarlo(
        coeffs, selected_func, n_g1=6000, n_b2g2=6000, seed=seed
    )
    return (coeffs, selected_func, true_wcde)

def generate_coeffs_list_parallel(G, n_coeff_sets, seed_base=0):
    nonlinear_funcs = [
        lambda x: np.sum(np.sin(x), axis=0),
        lambda x: np.sum(x, axis=0),
        lambda x: np.sum(np.cos(x), axis=0)
    ]
    futures = []
    G_edges = list(G.edges())
    for j in range(n_coeff_sets):
        seed = seed_base + j
        futures.append(
            generate_single_coeff_tuple.remote(G_edges, seed, nonlinear_funcs)
        )
    print(f"Submitted {n_coeff_sets} Monte Carlo truth tasks; waiting for Ray to finish...")
    coeffs_list = ray.get(futures)
    print("All coefficient sets and true WCDEs have been generated in parallel.")
    return coeffs_list

# ========================= Ray: parallel estimation =========================

@ray.remote(num_cpus=0.25, max_retries=1)  
def compute_for_adj_single(vas, coeffs_tuple, coeff_idx, n_rep, n_sample, treatment, outcome, all_mediators):
    _enforce_worker_determinism(seed=coeff_idx + 12345)
    coeffs, nonlinear_func, true_wcde = coeffs_tuple
    variance, mse, ests = montecarlo_var_fixed_coeff(
        n_rep, n_sample, treatment, outcome, set(vas), all_mediators,
        (coeffs, nonlinear_func), true_wcde, seed_base=coeff_idx * 1000
    )
    return frozenset(vas), coeff_idx, variance, mse, ests

def compare_adjusters_fine_grain(coeffs_list, n_rep, n_sample, treatment, outcome, all_mediators, adjusters_list):
    futures = []
    batch_size = 200
    for i in range(0, len(coeffs_list), batch_size):
        batch = coeffs_list[i:i+batch_size]
        for coeff_idx, coeffs_tuple in enumerate(batch, start=i):
            for vas in adjusters_list:
                futures.append(
                    compute_for_adj_single.remote(
                        frozenset(vas), coeffs_tuple, coeff_idx, n_rep, n_sample,
                        treatment, outcome, all_mediators
                    )
                )
        print(f"Submitted batch {i//batch_size + 1}/{(len(coeffs_list)-1)//batch_size + 1}, {len(batch)*len(adjusters_list)} tasks total")

    results = []
    total_tasks = len(futures)
    while futures:
        ready, futures = ray.wait(futures, num_returns=min(100, len(futures)), timeout=30.0)
        if ready:
            completed = ray.get(ready)
            results.extend(completed)
            print(f"Completed {len(results)}/{total_tasks} tasks ({len(results)/total_tasks*100:.1f}%)")

    results_dict = {}
    for vas, coeff_idx, var, mse, ests in results:
        if vas not in results_dict:
            results_dict[vas] = {
                'variances': [],
                'mses': [],
                'estimates': [],
                'coeff_indices': []
            }
        results_dict[vas]['variances'].append(var)
        results_dict[vas]['mses'].append(mse)
        results_dict[vas]['estimates'].append(ests)
        results_dict[vas]['coeff_indices'].append(coeff_idx)
    return results_dict

# ========================= Save Results =========================


def save_performance_results(adj_results_dict, n_coeff_sets, n_rep, n_sample, base_dir):
    os.makedirs(base_dir, exist_ok=True)
    performance_data = []
    for adj_set, results in adj_results_dict.items():
        adj_label = ",".join(sorted(adj_set))
        avg_variance = np.mean(results['variances'])
        avg_mse = np.mean(results['mses'])
        performance_data.append({
            'Adjustment_Set': adj_label,
            'Avg_Variance': float(avg_variance),
            'Avg_MSE': float(avg_mse),
        })
    df = pd.DataFrame(performance_data)
    csv_path = os.path.join(base_dir, f"performance_summary_n{n_sample}.csv")
    df.to_csv(csv_path, index=False)

    detailed_data = []
    for adj_set, results in adj_results_dict.items():
        adj_label = ",".join(sorted(adj_set))
        for i in range(len(results['variances'])):
            detailed_data.append({
                'Adjustment_Set': adj_label,
                'Coefficient_Set': i,
                'Variance': float(results['variances'][i]),
                'MSE': float(results['mses'][i])
            })
    detailed_df = pd.DataFrame(detailed_data)
    detailed_csv_path = os.path.join(base_dir, f"performance_detailed_n{n_sample}.csv")
    detailed_df.to_csv(detailed_csv_path, index=False)

    print(f"Average performance results saved to: {csv_path}")
    print(f"Detailed performance results saved to: {detailed_csv_path}")
    return df, detailed_df

# ========================= Main Program =========================
if __name__ == '__main__':
    _enforce_worker_determinism(seed=42)

    start_time = time.time()
    print("Program started...")
    start_str = time.strftime("%Y%m%d_%H%M%S")
    print(f"Start time: {start_str}")

    # Only perform sysctl on Linux root to avoid local failures causing timing differences
    if os.name == "posix" and hasattr(os, "geteuid") and os.geteuid() == 0:
        os.system("sysctl -w net.core.somaxconn=65535")
        os.system("sysctl -w net.ipv4.tcp_max_syn_backlog=65535")
        os.system("sysctl -w vm.swappiness=10")
        os.system("sysctl -w vm.dirty_ratio=40")
        os.system("sysctl -w vm.dirty_background_ratio=10")
    else:
        print("[info] skip sysctl tweaks (not Linux root)")

    # Initialize Ray and configure resources (align with AWS as much as possible, but not exceed 50% of local memory)
    num_cpus = psutil.cpu_count(logical=True)
    total_mem = psutil.virtual_memory().total
    target_oss = int(60 * 1024**3)  # 60GB (fix comment)
    object_store_memory = min(target_oss, int(total_mem * 0.5))
    if object_store_memory < target_oss:
        print(f"[warn] Using {object_store_memory/1024**3:.1f}GB (≤50% of physical memory)")

    # Distribute environment variables to all Ray workers (key)
    ENV_LOCK = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "MKL_SERVICE_FORCE_INTEL": "1",
        "MKL_CBWR": "COMPATIBLE",
        "MKL_DEBUG_CPU_TYPE": "5"
    }

    ray.init(
        num_cpus=num_cpus,
        object_store_memory=object_store_memory,
        ignore_reinit_error=True,
        runtime_env={"env_vars": ENV_LOCK}
    )
    print(f"Ray initialization complete: {num_cpus} CPUs, {object_store_memory/1024**3:.1f}GB object store")


    treat, out = 'A', 'Y'
    base_result_dir = "non-bac"
    os.makedirs(base_result_dir, exist_ok=True)

    G = generate_dag_structure()
    print(f"DAG structure: {G.edges()}")

    all_sets = find_potential_adjustment_sets(G, treat, out)
    all_mediators = get_all_mediator_path_nodes(G, treat, out)
    valid_sets_raw = filter_valid_structured_vas(G, treat, out, all_sets)

    # Normalize to frozenset to avoid "not in" due to type mismatch
    valid_sets = [frozenset(s) for s in valid_sets_raw]
    print(f"All mediator variables: {sorted(all_mediators)}")
    print(f"Found {len(valid_sets)} valid adjustment sets")

    baseline_vas = frozenset({'G1', 'G2'})
    if baseline_vas not in valid_sets:
        print(f"Adding baseline adjustment set: {baseline_vas}")
        valid_sets.append(baseline_vas)

    n_coeff_sets = 50
    n_rep = 100
    n_samples = [250, 500, 1000, 2000]
    print(f"\nExperiment settings:")
    print(f"  - Number of coefficient sets: {n_coeff_sets}")
    print(f"  - Repetitions per set: {n_rep}")
    print(f"  - Sample sizes: {n_samples}")
    print(f"  - Estimator: AIPW for WCDE; counterfactual table lookup; discriminative density ratio (joint vs product); empirical distribution weights")

    print(f"Generating {n_coeff_sets} sets of random coefficients, corresponding nonlinear functions, and true WCDE...")
    coeffs_list = generate_coeffs_list_parallel(G, n_coeff_sets, seed_base=123)

    a_y_vals = np.abs(np.array([coef_tuple[0]['A_Y'] for coef_tuple in coeffs_list], dtype=float))
    avg_a_y = float(a_y_vals.mean())
    std_a_y = float(a_y_vals.std(ddof=1)) if len(a_y_vals) > 1 else 0.0
    print(f"Coefficient |A_Y| mean: {avg_a_y:.6f}  (std={std_a_y:.6f})")

    variance_results = []
    mse_results = []

    for n_sample in n_samples:
        print(f"\n{'='*60}")
        print(f"Start evaluating performance for sample size n={n_sample} ...")
        adj_results_dict = compare_adjusters_fine_grain(
            coeffs_list, n_rep, n_sample,
            treat, out, all_mediators, valid_sets
        )
        print(f"Saving performance results for sample size n={n_sample} ...")
        summary_df, detailed_df = save_performance_results(
            adj_results_dict, n_coeff_sets, n_rep, n_sample,
            os.path.join(base_result_dir, f"n{n_sample}")
        )
        for adj_set, results in adj_results_dict.items():
            adj_label = ",".join(sorted(adj_set))
            avg_variance = float(np.mean(results['variances']))
            avg_mse = float(np.mean(results['mses']))

            variance_results.append({
                'Adjustment_Set': adj_label,
                'n_sample': n_sample,
                'Avg_Variance': avg_variance
            })
            mse_results.append({
                'Adjustment_Set': adj_label,
                'n_sample': n_sample,
                'Avg_MSE': avg_mse
            })

        # Cross-sample aggregation (only one n is generated here)
        variance_df = pd.DataFrame(variance_results)
        variance_table = variance_df.pivot(index='Adjustment_Set', columns='n_sample', values='Avg_Variance')
        variance_table.columns = [f'n={col}' for col in variance_table.columns]

        mse_df = pd.DataFrame(mse_results)
        mse_table = mse_df.pivot(index='Adjustment_Set', columns='n_sample', values='Avg_MSE')
        mse_table.columns = [f'n={col}' for col in mse_table.columns]
        mse_table.insert(0, 'Avg_A_Y', avg_a_y)

        variance_csv = os.path.join(base_result_dir, f"n{n_sample}", f"variance_summary_table_n{n_sample}.csv")
        mse_csv = os.path.join(base_result_dir, f"n{n_sample}", f"mse_summary_table_n{n_sample}.csv")
        variance_table.to_csv(variance_csv)
        mse_table.to_csv(mse_csv)

        print(f"[n={n_sample}] Variance summary table: {variance_csv}")
        print(f"[n={n_sample}] MSE summary table: {mse_csv}")

    print("\nSummary tables have been generated:")
    print(f"1. Variance summary table: {variance_csv}")
    print(f"2. MSE summary table: {mse_csv}")
    print(f"(Per-coefficient-set A_Y–MSE tables can be found in each n subdirectory, e.g., {os.path.join(base_result_dir, 'n100', 'mse_vs_A_Y_n100.csv')})")

    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nExperiment completed!")
    print(f"Total running time: {int(hours):02d}h:{int(minutes):02d}m:{seconds:.2f}s")

    ray.shutdown()
