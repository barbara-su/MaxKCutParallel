"""
Baseline algorithms for Max-3-Cut comparison.
All are inherently serial (no GPU parallelism).

1. SDP relaxation + random rounding (Goemans-Williamson / Frieze-Jerrum style)
2. Greedy (iterative best-node assignment)
3. Random (uniform random cuts)
"""
import os
import sys
import time

os.environ.setdefault("TMPDIR", os.environ.get("TMPDIR", "/tmp"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np


def score_cut(Q, z, K=3):
    """Compute Re(z† Q z) for a cut assignment z ∈ A_K^n."""
    return np.real(z.conj() @ Q @ z)


def random_cut(Q, K=3, num_trials=None, seed=42):
    """Generate random cuts and return the best one.

    Args:
        Q: (n, n) real Laplacian
        K: number of partitions
        num_trials: number of random cuts to try (default: n+1 to match rank-1)
        seed: random seed
    Returns:
        best_score, best_z, elapsed_seconds
    """
    n = Q.shape[0]
    if num_trials is None:
        num_trials = n + 1
    rng = np.random.RandomState(seed)
    roots = np.exp(2j * np.pi * np.arange(K) / K)

    t0 = time.time()
    best_score = -np.inf
    best_z = None

    for _ in range(num_trials):
        k = rng.randint(0, K, size=n)
        z = roots[k]
        s = score_cut(Q, z, K)
        if s > best_score:
            best_score = s
            best_z = z.copy()

    elapsed = time.time() - t0
    return float(best_score), best_z, elapsed


def greedy_cut(Q, K=3, seed=42, init_k=None):
    """Greedy Max-K-Cut: iteratively assign each node to the best partition.

    Repeatedly scans all nodes; for each node, tries all K assignments
    and picks the one maximizing the cut. Repeats until no improvement.

    Args:
        Q: (n, n) real Laplacian
        K: number of partitions
        seed: random seed for initial assignment (used if init_k is None)
        init_k: optional initial partition vector (int array, values 0..K-1).
                 If provided, warm-starts from this assignment instead of random.
    Returns:
        best_score, best_z, elapsed_seconds, iterations
    """
    n = Q.shape[0]
    rng = np.random.RandomState(seed)
    roots = np.exp(2j * np.pi * np.arange(K) / K)

    t0 = time.time()

    # Initialize: warm-start or random
    if init_k is not None:
        k = np.asarray(init_k, dtype=int).copy() % K
    else:
        k = rng.randint(0, K, size=n)
    z = roots[k]
    best_score = score_cut(Q, z, K)

    improved = True
    iterations = 0
    while improved:
        improved = False
        iterations += 1
        for i in range(n):
            current_k = k[i]
            best_local_score = best_score
            best_local_k = current_k

            for trial_k in range(K):
                if trial_k == current_k:
                    continue
                k[i] = trial_k
                z[i] = roots[trial_k]
                s = score_cut(Q, z, K)
                if s > best_local_score:
                    best_local_score = s
                    best_local_k = trial_k

            if best_local_k != current_k:
                k[i] = best_local_k
                z[i] = roots[best_local_k]
                best_score = best_local_score
                improved = True
            else:
                k[i] = current_k
                z[i] = roots[current_k]

    elapsed = time.time() - t0
    return float(best_score), z.copy(), elapsed, iterations


def sdp_max3cut(Q, K=3, num_rounds=100, seed=42):
    """SDP relaxation + random rounding for Max-3-Cut.

    Solves:
        max  tr(Q @ Z)
        s.t. Z_ii = 1 for all i
             Z ⪰ 0

    Then rounds using random hyperplane projection.
    Uses Cholesky factorization of the SDP solution + projection onto roots of unity.

    This is a simplified version that uses eigendecomposition as a proxy for the
    SDP relaxation (the top eigenvectors of Q give the SDP-like relaxation).

    Args:
        Q: (n, n) real Laplacian
        K: number of partitions
        num_rounds: number of random rounding attempts
        seed: random seed
    Returns:
        best_score, best_z, elapsed_seconds, sdp_info
    """
    n = Q.shape[0]
    rng = np.random.RandomState(seed)
    roots = np.exp(2j * np.pi * np.arange(K) / K)

    t0 = time.time()

    # Step 1: SDP relaxation via eigendecomposition
    # The SDP relaxation Z* has the property that its top eigenvectors
    # capture the optimal partition structure. We use the spectral relaxation:
    # embed each node in R^n using the top eigenvectors of Q, then round.
    t_sdp_start = time.time()

    try:
        # Try cvxpy if available for exact SDP
        import cvxpy as cp
        Z = cp.Variable((n, n), hermitian=True)
        constraints = [Z >> 0]  # PSD constraint
        constraints += [Z[i, i] == 1 for i in range(n)]  # diagonal = 1
        objective = cp.Maximize(cp.real(cp.trace(Q @ Z)))
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, max_iters=10000, verbose=False, time_limit_secs=600)

        Z_val = Z.value
        if Z_val is None:
            raise RuntimeError("SDP solver failed")

        # Cholesky-like factorization for rounding
        eigvals, eigvecs = np.linalg.eigh(Z_val)
        eigvals = np.maximum(eigvals, 0)
        V_sdp = eigvecs * np.sqrt(eigvals)  # (n, n) embedding

        sdp_bound = float(prob.value)
        sdp_method = "cvxpy_scs"
        t_sdp = time.time() - t_sdp_start

    except (ImportError, Exception) as e:
        # Fallback: spectral relaxation (not true SDP, but captures the structure)
        eigvals, eigvecs = np.linalg.eigh(Q)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Use top-d eigenvectors (d = min(n, reasonable rank))
        d = min(n, 50)
        V_sdp = eigvecs[:, :d] * np.sqrt(np.maximum(eigvals[:d], 0))

        sdp_bound = float(np.sum(np.maximum(eigvals, 0)))  # trace bound
        sdp_method = "spectral_relaxation"
        t_sdp = time.time() - t_sdp_start

    # Step 2: Random hyperplane rounding
    t_round_start = time.time()
    best_score = -np.inf
    best_z = None

    for _ in range(num_rounds):
        # Random direction in embedding space
        r_vec = rng.randn(V_sdp.shape[1])
        r_vec = r_vec / np.linalg.norm(r_vec)

        # Project each node onto the random direction
        projections = np.real(V_sdp @ r_vec).astype(np.float64)

        # For complex rounding: use 2D projection
        if V_sdp.shape[1] >= 2:
            r_vec2 = rng.randn(V_sdp.shape[1])
            r_vec2 = r_vec2 - np.dot(r_vec2, r_vec) * r_vec
            r_vec2 = r_vec2 / (np.linalg.norm(r_vec2) + 1e-10)
            proj2 = np.real(V_sdp @ r_vec2).astype(np.float64)
            angles = np.arctan2(proj2, projections)
        else:
            angles = np.sign(projections) * np.pi / 3

        # Quantize angles to nearest K-th root of unity
        k = np.round(angles * K / (2 * np.pi)).astype(int) % K
        z = roots[k]
        s = score_cut(Q, z, K)
        if s > best_score:
            best_score = s
            best_z = z.copy()

    t_round = time.time() - t_round_start
    elapsed = time.time() - t0

    sdp_info = {
        "method": sdp_method,
        "sdp_bound": sdp_bound,
        "sdp_time": t_sdp,
        "rounding_time": t_round,
        "num_rounds": num_rounds,
    }
    return float(best_score), best_z, elapsed, sdp_info


def _sparse_update_Qz(Q_csc, Qz, idx, dz):
    """Update Qz += Q[:, idx] * dz using sparse index-based access. O(degree)."""
    start = Q_csc.indptr[idx]
    end = Q_csc.indptr[idx + 1]
    rows = Q_csc.indices[start:end]
    vals = Q_csc.data[start:end]
    Qz[rows] += vals * dz


def _incremental_delta_fast(Qz, Q_diag, idx, old_z, new_z):
    """Compute score change when z[idx] flips. O(1).

    Δ = 2·Re(conj(dz)·Qz[idx]) + |dz|²·Q[idx,idx]
    """
    dz = new_z - old_z
    return 2 * np.real(np.conj(dz) * Qz[idx]) + np.abs(dz)**2 * Q_diag[idx]


def _incremental_delta(Q, Qz, z, idx, old_z, new_z, is_sparse=False):
    """Legacy wrapper. Compute score change when z[idx] flips."""
    dz = new_z - old_z
    Q_ii = Q[idx, idx]
    delta = 2 * np.real(np.conj(dz) * Qz[idx]) + np.abs(dz)**2 * Q_ii
    return delta


def _update_Qz(Q, Qz, idx, dz, is_sparse=False):
    """Legacy wrapper. Update Qz after z[idx] changes by dz."""
    from scipy import sparse
    if is_sparse:
        if sparse.isspmatrix_csc(Q):
            col = np.asarray(Q[:, idx].toarray()).flatten()
        else:
            col = np.asarray(Q.tocsc()[:, idx].toarray()).flatten()
        Qz += col * dz
    else:
        Qz += Q[:, idx] * dz
    return Qz


def greedy_cut_incremental(Q, K=3, seed=42, init_k=None, max_time=None, max_iters=None):
    """Greedy Max-K-Cut with vectorized incremental scoring.

    For each iteration, vectorizes the "best flip" computation across all nodes
    using numpy operations. Only the committed flips do sparse column access.

    Args:
        Q: (n, n) matrix (dense or scipy.sparse)
        K: number of partitions
        seed: random seed
        init_k: optional warm-start partition vector
        max_time: wall-clock time limit in seconds (None = no limit)
        max_iters: max greedy passes (None = until convergence)
    Returns:
        best_score, best_z, elapsed_seconds, iterations
    """
    from scipy import sparse
    is_sparse = sparse.issparse(Q)

    n = Q.shape[0]
    rng = np.random.RandomState(seed)
    roots = np.exp(2j * np.pi * np.arange(K) / K)

    t0 = time.time()

    if init_k is not None:
        k = np.asarray(init_k, dtype=int).copy() % K
    else:
        k = rng.randint(0, K, size=n)
    z = roots[k].astype(np.complex128)

    # Precompute Qz and diagonal
    if is_sparse:
        Qz = np.asarray(Q.dot(z)).flatten()
        Q_diag = np.asarray(Q.diagonal()).flatten()
        Q_csc = Q.tocsc() if not sparse.isspmatrix_csc(Q) else Q
    else:
        Qz = Q @ z
        Q_diag = np.diag(Q).copy()
        Q_csc = None

    score = np.real(z.conj() @ Qz)
    best_score = score
    best_k = k.copy()

    improved = True
    iterations = 0
    while improved:
        if max_time and (time.time() - t0) > max_time:
            break
        if max_iters and iterations >= max_iters:
            break
        improved = False
        iterations += 1

        # Vectorized: compute delta for all nodes × all trial assignments
        # delta[i, trial_k] = 2·Re(conj(root_k - z[i]) · Qz[i]) + |root_k - z[i]|² · Q[i,i]
        for trial_k in range(K):
            new_z_all = roots[trial_k]
            dz_all = new_z_all - z  # (n,)
            delta_all = 2 * np.real(np.conj(dz_all) * Qz) + np.abs(dz_all)**2 * Q_diag

            # Mask: only consider nodes where trial_k != current k
            mask = (k != trial_k)
            delta_all[~mask] = -np.inf

            if trial_k == 0:
                best_delta = delta_all.copy()
                best_trial = np.full(n, trial_k, dtype=int)
            else:
                better = delta_all > best_delta
                best_delta[better] = delta_all[better]
                best_trial[better] = trial_k

        # Find nodes with positive improvement
        improving = best_delta > 1e-10
        if not np.any(improving):
            break

        # Apply all improving flips (greedy: apply in order for correctness)
        # For speed, apply all at once (parallel greedy variant)
        flip_indices = np.where(improving)[0]
        for i in flip_indices:
            if max_time and (time.time() - t0) > max_time:
                break
            new_k = best_trial[i]
            new_z = roots[new_k]
            old_z = z[i]
            dz = new_z - old_z

            # Recompute delta with current Qz (may have changed from prior flips this pass)
            delta = 2 * np.real(np.conj(dz) * Qz[i]) + np.abs(dz)**2 * Q_diag[i]
            if delta <= 0:
                continue

            # Commit flip — O(degree) sparse update
            if Q_csc is not None:
                _sparse_update_Qz(Q_csc, Qz, i, dz)
            else:
                Qz += Q[:, i] * dz
            z[i] = new_z
            k[i] = new_k
            score += delta
            improved = True

        if score > best_score:
            best_score = score
            best_k = k.copy()

    elapsed = time.time() - t0
    best_z = roots[best_k]
    return float(best_score), best_z, elapsed, iterations


def sa_cut(Q, K=3, seed=42, init_k=None, max_iters=None, max_time=None,
           T_init=None, cooling=0.9999):
    """Simulated Annealing for Max-K-Cut with incremental scoring.

    Random single-node flips with Metropolis acceptance criterion.
    Uses O(degree) incremental score updates.

    Args:
        Q: (n, n) matrix (dense or scipy.sparse)
        K: number of partitions
        seed: random seed
        init_k: optional warm-start partition vector
        max_iters: max number of flip attempts (default: 10*n)
        max_time: wall-clock time limit in seconds
        T_init: initial temperature (default: auto from avg edge weight)
        cooling: geometric cooling factor per iteration
    Returns:
        best_score, best_z, elapsed_seconds, accepted_moves
    """
    from scipy import sparse
    is_sparse = sparse.issparse(Q)

    n = Q.shape[0]
    rng = np.random.RandomState(seed)
    roots = np.exp(2j * np.pi * np.arange(K) / K)

    if max_iters is None:
        max_iters = 10 * n

    t0 = time.time()

    if init_k is not None:
        k = np.asarray(init_k, dtype=int).copy() % K
    else:
        k = rng.randint(0, K, size=n)
    z = roots[k].astype(np.complex128)

    if is_sparse:
        Qz = np.asarray(Q.dot(z)).flatten()
        Q_csc = Q.tocsc() if not sparse.isspmatrix_csc(Q) else Q
    else:
        Qz = Q @ z
        Q_csc = None

    score = np.real(z.conj() @ Qz)
    best_score = score
    best_k = k.copy()

    # Auto temperature: calibrate so initial acceptance rate ~50% for typical moves.
    # A single-node flip on a graph with degree d changes score by O(d).
    # For K=3 roots of unity, |dz| ~ 1.7, so typical |delta| ~ 2*d*1.7 ~ 3.4*d.
    # Setting T_init ~ 2*avg_degree gives ~50% acceptance for typical worsening moves.
    if T_init is None:
        if is_sparse:
            avg_degree = Q.nnz / n if n > 0 else 5.0
        else:
            avg_degree = np.count_nonzero(Q) / n if n > 0 else 5.0
        T_init = max(2.0 * avg_degree, 1.0)

    T = T_init
    accepted = 0

    for it in range(max_iters):
        if max_time and it % 1000 == 0 and (time.time() - t0) > max_time:
            break

        # Random node and random new assignment
        i = rng.randint(n)
        old_k = k[i]
        new_k = rng.randint(K - 1)
        if new_k >= old_k:
            new_k += 1
        old_z = z[i]
        new_z = roots[new_k]

        delta = _incremental_delta(Q, Qz, z, i, old_z, new_z, is_sparse)

        # Metropolis criterion
        if delta > 0 or (T > 0 and rng.random() < np.exp(min(delta / T, 0))):
            dz = new_z - old_z
            if Q_csc is not None:
                _sparse_update_Qz(Q_csc, Qz, i, dz)
            else:
                Qz += Q[:, i] * dz
            z[i] = new_z
            k[i] = new_k
            score += delta
            accepted += 1

            if score > best_score:
                best_score = score
                best_k = k.copy()

        T *= cooling

    elapsed = time.time() - t0
    best_z = roots[best_k]
    return float(best_score), best_z, elapsed, accepted


def tabu_cut(Q, K=3, seed=42, init_k=None, max_iters=None, max_time=None,
             tabu_tenure=None):
    """Tabu Search for Max-K-Cut with incremental scoring.

    Best-improvement local search with tabu list preventing recent flips.

    Args:
        Q: (n, n) matrix (dense or scipy.sparse)
        K: number of partitions
        seed: random seed
        init_k: optional warm-start partition vector
        max_iters: max number of iterations (default: 10*n)
        max_time: wall-clock time limit in seconds
        tabu_tenure: number of iterations a flip is forbidden (default: n//10)
    Returns:
        best_score, best_z, elapsed_seconds, iterations
    """
    from scipy import sparse
    is_sparse = sparse.issparse(Q)

    n = Q.shape[0]
    rng = np.random.RandomState(seed)
    roots = np.exp(2j * np.pi * np.arange(K) / K)

    if max_iters is None:
        max_iters = 10 * n
    if tabu_tenure is None:
        tabu_tenure = max(7, n // 10)

    t0 = time.time()

    if init_k is not None:
        k = np.asarray(init_k, dtype=int).copy() % K
    else:
        k = rng.randint(0, K, size=n)
    z = roots[k].astype(np.complex128)

    if is_sparse:
        Qz = np.asarray(Q.dot(z)).flatten()
        Q_csc = Q.tocsc() if not sparse.isspmatrix_csc(Q) else Q
    else:
        Qz = Q @ z
        Q_csc = None

    score = np.real(z.conj() @ Qz)
    best_score = score
    best_k = k.copy()

    # Tabu list: tabu[i] = iteration when node i becomes non-tabu
    tabu = np.zeros(n, dtype=int)

    for it in range(max_iters):
        if max_time and it % 1000 == 0 and (time.time() - t0) > max_time:
            break

        # Find best non-tabu move (or aspiration: allow tabu if improves global best)
        best_move_delta = -np.inf
        best_move_i = -1
        best_move_k = -1

        for i in range(n):
            old_z = z[i]
            for trial_k in range(K):
                if trial_k == k[i]:
                    continue
                new_z = roots[trial_k]
                delta = _incremental_delta(Q, Qz, z, i, old_z, new_z, is_sparse)

                # Accept if: not tabu, OR aspiration (improves best)
                is_tabu = tabu[i] > it
                aspiration = (score + delta) > best_score

                if (not is_tabu or aspiration) and delta > best_move_delta:
                    best_move_delta = delta
                    best_move_i = i
                    best_move_k = trial_k

        if best_move_i < 0:
            break  # No improving move found

        # Apply best move
        i = best_move_i
        old_z = z[i]
        new_z = roots[best_move_k]
        dz = new_z - old_z

        if Q_csc is not None:
            _sparse_update_Qz(Q_csc, Qz, i, dz)
        else:
            Qz += Q[:, i] * dz
        z[i] = new_z
        k[i] = best_move_k
        score += best_move_delta
        tabu[i] = it + tabu_tenure

        if score > best_score:
            best_score = score
            best_k = k.copy()

    elapsed = time.time() - t0
    best_z = roots[best_k]
    return float(best_score), best_z, elapsed, it + 1


def dsatur_cut(Q, K=3, seed=42, improve=True, max_time=None):
    """DSatur-inspired Max-K-Cut heuristic.

    Two-phase algorithm inspired by Apte et al. (arXiv:2602.05956):
    1. Construction: iteratively assign vertices by maximum saturation degree
       (number of distinct labels among neighbors), breaking ties by total
       incident edge weight. Each vertex gets the label maximizing immediate
       cut edges.
    2. Local improvement: 1-opt hill-climbing (flip any vertex whose relabel
       strictly increases the global cut). Equivalent to greedy_cut_incremental.

    Runtime: O(|E| log |V|) construction + O(r|E|) improvement.

    Args:
        Q: (n, n) matrix — Laplacian or weight matrix (dense or scipy.sparse).
           For Laplacian L = D - A, edge weight w(u,v) = -L[u,v].
        K: number of partitions
        seed: random seed (unused currently, deterministic)
        improve: if True, run local improvement phase
        max_time: wall-clock time limit in seconds (None = no limit)
    Returns:
        best_score, best_k, elapsed_seconds, construction_score
    """
    import heapq
    from scipy import sparse

    t0 = time.time()
    is_sparse = sparse.issparse(Q)

    n = Q.shape[0]

    # Build adjacency list from Q.
    # For Laplacian L = D - A: edge weight w(u,v) = -L[u,v] for u != v.
    # For general weight matrix: w(u,v) = Q[u,v].
    # Detect Laplacian: row sums ≈ 0.
    if is_sparse:
        row_sums = np.asarray(Q.sum(axis=1)).ravel()
        is_laplacian = np.max(np.abs(row_sums)) < 1e-8 * max(1.0, np.max(np.abs(Q.diagonal())))
        coo = sparse.triu(Q, k=1).tocoo()
        edge_rows = coo.row
        edge_cols = coo.col
        edge_vals = coo.data
    else:
        row_sums = Q.sum(axis=1)
        is_laplacian = np.max(np.abs(row_sums)) < 1e-8 * max(1.0, np.max(np.abs(np.diag(Q))))
        triu_idx = np.triu_indices(n, k=1)
        edge_rows = triu_idx[0]
        edge_cols = triu_idx[1]
        edge_vals = Q[triu_idx]

    # Build adjacency: list of (neighbor, weight) per vertex
    adj = [[] for _ in range(n)]
    for idx in range(len(edge_rows)):
        u, v = int(edge_rows[idx]), int(edge_cols[idx])
        w = float(edge_vals[idx])
        if is_laplacian:
            w = -w  # Laplacian off-diag = -weight
        if abs(w) < 1e-15:
            continue
        adj[u].append((v, w))
        adj[v].append((u, w))

    # Total incident weight per vertex
    tot_w = np.array([sum(abs(w) for _, w in adj[v]) for v in range(n)])

    # Assignment and weighted label sums
    assign = np.full(n, -1, dtype=np.int32)
    wsum = np.zeros((n, K), dtype=np.float64)  # wsum[v, a] = sum of weights from v to neighbors labeled a

    unassigned = set(range(n))
    cnt = 0
    node_cnt = {}

    def prio(v):
        sat = sum(1 for a in range(K) if wsum[v, a] > 0)
        return (sat, tot_w[v])

    # Initialize priority queue
    heap = []
    for v in range(n):
        p = prio(v)
        heapq.heappush(heap, (-p[0], -p[1], cnt, v))
        node_cnt[v] = cnt
        cnt += 1

    # Phase 1: Construction
    while unassigned:
        if max_time and (time.time() - t0) > max_time:
            # Assign remaining randomly
            for v in list(unassigned):
                assign[v] = 0
                unassigned.discard(v)
            break

        # Pop highest-priority unassigned vertex
        while heap:
            _, _, c, v = heapq.heappop(heap)
            if v in unassigned and c == node_cnt[v]:
                break
        else:
            break

        # Choose label maximizing cut: tot - wsum[v, a] is the weight of
        # edges from v to neighbors NOT in partition a (= cut contribution)
        tot = wsum[v].sum()
        best_a = 0
        best_gain = tot - wsum[v, 0]
        for a in range(1, K):
            g = tot - wsum[v, a]
            if g > best_gain or (g == best_gain and wsum[v, a] < wsum[v, best_a]):
                best_gain = g
                best_a = a

        assign[v] = best_a
        unassigned.discard(v)

        # Update neighbors
        for u, w in adj[v]:
            wsum[u, best_a] += w
            if u in unassigned:
                p = prio(u)
                heapq.heappush(heap, (-p[0], -p[1], cnt, u))
                node_cnt[u] = cnt
                cnt += 1

    # Compute construction score: sum of weights of cut edges
    construction_score = 0.0
    for v in range(n):
        for u, w in adj[v]:
            if v < u and assign[v] != assign[u]:
                construction_score += w

    # For K=3 with Laplacian, the z†Lz score = 3 * weighted_cut
    if is_laplacian and K == 3:
        construction_score *= 3.0

    # Phase 2: Local improvement (1-opt hill climbing)
    if improve:
        improved = True
        while improved:
            if max_time and (time.time() - t0) > max_time:
                break
            improved = False
            for v in range(n):
                if not adj[v]:
                    continue
                cur = assign[v]
                tw = sum(w for _, w in adj[v])
                best_val = tw - wsum[v, cur]
                best_a = cur
                for a in range(K):
                    if a != cur and tw - wsum[v, a] > best_val:
                        best_val = tw - wsum[v, a]
                        best_a = a
                if best_a != cur:
                    for u, w in adj[v]:
                        wsum[u, cur] -= w
                        wsum[u, best_a] += w
                    assign[v] = best_a
                    improved = True

    # Final score
    final_score = 0.0
    for v in range(n):
        for u, w in adj[v]:
            if v < u and assign[v] != assign[u]:
                final_score += w
    if is_laplacian and K == 3:
        final_score *= 3.0

    elapsed = time.time() - t0
    return float(final_score), assign.copy(), elapsed, float(construction_score)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run baseline algorithms for Max-3-Cut")
    parser.add_argument("--q_path", type=str, required=True)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--methods", type=str, default="all",
                        help="Comma-separated: random,greedy,sdp or 'all'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sdp_rounds", type=int, default=100)
    parser.add_argument("--random_trials", type=int, default=0, help="0 = n+1")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    Q = np.load(args.q_path).astype(np.float64)
    n = Q.shape[0]
    print(f"Instance: n={n}, K={args.K}")

    methods = args.methods.split(",") if args.methods != "all" else ["random", "greedy", "sdp"]
    results = {"n": n, "K": args.K, "q_path": args.q_path}

    for method in methods:
        print(f"\n--- {method.upper()} ---")
        if method == "random":
            trials = args.random_trials if args.random_trials > 0 else n + 1
            score, z, elapsed = random_cut(Q, K=args.K, num_trials=trials, seed=args.seed)
            results["random"] = {"score": score, "time": elapsed, "trials": trials}
            print(f"  Score: {score:.0f}, Time: {elapsed:.3f}s, Trials: {trials}")

        elif method == "greedy":
            score, z, elapsed = greedy_cut(Q, K=args.K, seed=args.seed)
            results["greedy"] = {"score": score, "time": elapsed}
            print(f"  Score: {score:.0f}, Time: {elapsed:.3f}s")

        elif method == "sdp":
            score, z, elapsed, info = sdp_max3cut(Q, K=args.K, num_rounds=args.sdp_rounds, seed=args.seed)
            results["sdp"] = {"score": score, "time": elapsed, **info}
            print(f"  Score: {score:.0f}, Time: {elapsed:.3f}s")
            print(f"  SDP method: {info['method']}, SDP time: {info['sdp_time']:.3f}s, Rounding: {info['rounding_time']:.3f}s")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {args.out}")
    else:
        print(f"\nResults: {json.dumps({k: v for k, v in results.items() if k != 'q_path'}, indent=2)}")
