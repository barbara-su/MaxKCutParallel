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


def greedy_cut(Q, K=3, seed=42):
    """Greedy Max-K-Cut: iteratively assign each node to the best partition.

    Repeatedly scans all nodes; for each node, tries all K assignments
    and picks the one maximizing the cut. Repeats until no improvement.

    Args:
        Q: (n, n) real Laplacian
        K: number of partitions
        seed: random seed for initial assignment
    Returns:
        best_score, best_z, elapsed_seconds
    """
    n = Q.shape[0]
    rng = np.random.RandomState(seed)
    roots = np.exp(2j * np.pi * np.arange(K) / K)

    t0 = time.time()

    # Random initial assignment
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
    return float(best_score), z.copy(), elapsed


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
