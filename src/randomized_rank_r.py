"""
Randomized Rank-r solver for Max-K-Cut.

Instead of enumerating ALL O(n^{2r-1}) candidates, randomly samples a fraction.
Key question: how many samples are needed to get within (1-ε) of the exact solution?

Usage:
    python randomized_rank_r.py --q_path Q.npy --v_path V.npy --rank 2 --K 3 \
        --sample_fraction 0.01 --num_gpus 4 --out result.json
"""
import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("TMPDIR", os.environ.get("TMPDIR", "/tmp"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from utils import compute_vtilde


def count_valid_combinations(n, K, r):
    """Total number of valid (2r-1)-tuples."""
    return math.comb(K * n, 2 * r - 1)


def random_valid_combination(n, K, r, rng):
    """Generate one random valid (2r-1)-tuple from [0, Kn)."""
    Kn = K * n
    comb_size = 2 * r - 1
    # Random combination without replacement
    return np.sort(rng.choice(Kn, size=comb_size, replace=False))


def score_candidates_batch(V_tilde, V, Q, candidates, r, K):
    """Score a batch of candidate index sets.

    For each candidate (2r-1)-tuple:
    1. Extract rows from V_tilde
    2. Compute null vector (the direction c)
    3. Project V @ c, quantize to nearest root of unity
    4. Score z†Qz

    Args:
        V_tilde: (Kn, 2r) real matrix
        V: (n, r) complex matrix
        Q: (n, n) real matrix (dense or sparse)
        candidates: (B, 2r-1) int64 array of index sets
        r: rank
        K: alphabet size
    Returns:
        best_score, best_k, best_z across the batch
    """
    from scipy import sparse

    n = V.shape[0]
    roots = np.exp(2j * np.pi * np.arange(K) / K)
    is_sparse = sparse.issparse(Q)

    best_score = -np.inf
    best_k = None
    best_z = None
    feasible = 0

    for idx in range(candidates.shape[0]):
        I = candidates[idx]
        VI = V_tilde[I]  # (2r-1, 2r)

        # Null vector via SVD
        try:
            _, S, Vt = np.linalg.svd(VI)
            c_tilde = Vt[-1]  # last row = null vector
        except np.linalg.LinAlgError:
            continue

        # Check feasibility: phi_{2r-2} in (-pi/K, pi/K]
        # Convert c_tilde to spherical coordinates
        d = len(c_tilde)
        phi = np.zeros(d - 1)
        eps = 1e-10
        for j in range(d - 1):
            if j == 0:
                arg = np.clip(c_tilde[0], -1, 1)
                phi[0] = np.arcsin(arg)
            else:
                prod_cos = 1.0
                for i in range(j):
                    prod_cos *= np.cos(phi[i])
                if abs(prod_cos) < eps:
                    phi[j] = 0
                else:
                    arg = np.clip(c_tilde[j] / prod_cos, -1, 1)
                    phi[j] = np.arcsin(arg)

        if not (-np.pi / K < phi[2 * r - 2] <= np.pi / K):
            continue

        # Sign correction
        j = d - 2
        sign_c = 1.0
        if phi[j] != 0 and c_tilde[j] != 0 and abs(np.cos(phi[j])) >= eps:
            val = np.tan(phi[j]) * c_tilde[j] * c_tilde[j + 1]
            sign_c = np.sign(val)

        c_tilde = c_tilde * sign_c

        # Convert to complex
        c = c_tilde[0:2*r:2] + 1j * c_tilde[1:2*r:2]  # (r,) complex

        # Project and quantize
        y = V[:, :r] @ c  # (n,) complex
        # Nearest root of unity
        k = np.zeros(n, dtype=int)
        for root_id in range(K):
            proj = np.real(np.conj(roots[root_id]) * y)
            if root_id == 0:
                best_proj = proj.copy()
            else:
                better = proj > best_proj
                k[better] = root_id
                best_proj[better] = proj[better]

        z = roots[k]

        # Score
        if is_sparse:
            Qz = Q.dot(z)
        else:
            Qz = Q @ z
        score = np.real(z.conj() @ Qz)
        feasible += 1

        if score > best_score:
            best_score = score
            best_k = k.copy()
            best_z = z.copy()

    return best_score, best_k, best_z, feasible


def randomized_rank_r(Q, V, r=2, K=3, sample_fraction=0.01, max_samples=None,
                       max_time=None, seed=42, batch_size=10000, verbose=True):
    """Randomized rank-r solver.

    Args:
        Q: (n, n) objective matrix (dense or sparse)
        V: (n, r) complex eigenvector matrix
        r: rank
        K: alphabet size
        sample_fraction: fraction of total candidates to sample (0 to 1)
        max_samples: absolute cap on number of samples (overrides fraction if smaller)
        max_time: wall-clock time limit in seconds
        seed: random seed
        batch_size: process this many candidates at a time
        verbose: print progress
    Returns:
        dict with best_score, best_k, best_z, num_samples, feasible_count, elapsed
    """
    n = Q.shape[0]
    rng = np.random.RandomState(seed)

    # Compute V_tilde
    V_tilde = compute_vtilde(V[:, :r]).astype(np.float64)

    total_comb = count_valid_combinations(n, K, r)
    num_samples = int(total_comb * sample_fraction)
    if max_samples is not None:
        num_samples = min(num_samples, max_samples)
    num_samples = max(num_samples, 1)

    if verbose:
        print(f"  Randomized rank-{r}: {num_samples:,} samples of {total_comb:,} total ({sample_fraction*100:.2f}%)")

    t0 = time.time()
    best_score = -np.inf
    best_k = None
    best_z = None
    total_feasible = 0
    total_processed = 0

    Kn = K * n
    comb_size = 2 * r - 1

    while total_processed < num_samples:
        if max_time and (time.time() - t0) > max_time:
            break

        cur_batch = min(batch_size, num_samples - total_processed)

        # Generate random candidates
        candidates = np.zeros((cur_batch, comb_size), dtype=np.int64)
        for i in range(cur_batch):
            candidates[i] = np.sort(rng.choice(Kn, size=comb_size, replace=False))

        score, k, z, feas = score_candidates_batch(
            V_tilde, V[:, :r], Q, candidates, r, K
        )
        total_feasible += feas
        total_processed += cur_batch

        if k is not None and score > best_score:
            best_score = score
            best_k = k
            best_z = z

        if verbose and total_processed % (batch_size * 10) == 0:
            elapsed = time.time() - t0
            rate = total_processed / elapsed if elapsed > 0 else 0
            print(f"    {total_processed:,}/{num_samples:,} ({total_processed/num_samples*100:.1f}%) "
                  f"score={best_score:.0f} feasible={total_feasible}/{total_processed} "
                  f"rate={rate:.0f}/s")

    elapsed = time.time() - t0

    return {
        "best_score": float(best_score) if best_score > -np.inf else None,
        "best_k": best_k,
        "best_z": best_z,
        "num_samples": total_processed,
        "total_candidates": total_comb,
        "sample_fraction": total_processed / total_comb,
        "feasible_count": total_feasible,
        "elapsed": round(elapsed, 2),
        "rate": round(total_processed / elapsed, 1) if elapsed > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Randomized rank-r Max-K-Cut solver")
    parser.add_argument("--q_path", type=str, required=True)
    parser.add_argument("--v_path", type=str, required=True)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--sample_fraction", type=float, default=0.01)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_time", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    Q = np.load(args.q_path).astype(np.float64)
    V = np.load(args.v_path)
    if V.ndim == 1:
        V = V.reshape(-1, 1)
    if not np.iscomplexobj(V):
        V = V.astype(np.complex128)

    n = Q.shape[0]
    print(f"Instance: n={n}, rank={args.rank}, K={args.K}")
    print(f"Sample fraction: {args.sample_fraction*100:.2f}%")

    result = randomized_rank_r(
        Q, V, r=args.rank, K=args.K,
        sample_fraction=args.sample_fraction,
        max_samples=args.max_samples,
        max_time=args.max_time,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    print(f"\nResult: score={result['best_score']:.0f}, "
          f"samples={result['num_samples']:,}/{result['total_candidates']:,}, "
          f"feasible={result['feasible_count']:,}, "
          f"time={result['elapsed']:.1f}s")

    if args.out:
        out = {k: v for k, v in result.items() if k not in ('best_k', 'best_z')}
        out["q_path"] = args.q_path
        if result["best_k"] is not None:
            out["best_k"] = result["best_k"].tolist()
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
