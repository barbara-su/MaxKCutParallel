"""
Hybrid Rank-1 + Local Search solver for Max-K-Cut.

Strategy:
  1. Run the rank-1 phase sweep algorithm with O(n·degree) incremental scoring
  2. Use the rank-1 solution as a warm start for greedy local search
  3. Compare: rank-1 alone, greedy from random, greedy from rank-1 warm start

Key improvements over naive implementation:
  - 2-eigenvector complex rounding for K=3 (avoids real-eigenvector degeneracy)
  - Sparse index-based Qz updates: O(degree) per flip, not O(n)
  - Total sweep cost: O(n·degree) for sparse graphs
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("TMPDIR", os.environ.get("TMPDIR", "/tmp"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from baselines import score_cut, greedy_cut


def _sparse_update_Qz(Q_csc, Qz, idx, dz):
    """Update Qz += Q[:, idx] * dz using sparse index-based access. O(degree)."""
    start = Q_csc.indptr[idx]
    end = Q_csc.indptr[idx + 1]
    rows = Q_csc.indices[start:end]
    vals = Q_csc.data[start:end]
    Qz[rows] += vals * dz


def rank1_phase_sweep(Q, V, K=3):
    """CPU rank-1 phase sweep with O(n·degree) incremental scoring.

    For K=3 with real Laplacians, uses 2-eigenvector complex rounding:
    q[i] = v1[i] + j·v2[i], giving angles that span [0, 2π) and use all K partitions.

    Returns (score, k, z, elapsed).
    """
    from scipy import sparse

    n = Q.shape[0]

    t0 = time.time()

    # 2-eigenvector complex rounding for K≥3 with real Laplacians:
    # A single real eigenvector has angle(q) ∈ {0,π}, mapping to only 2 of K partitions.
    # Using q = v1 + j·v2 gives genuine complex angles spanning [0, 2π).
    if V.ndim == 2 and V.shape[1] >= 2 and K >= 3:
        v1 = np.real(V[:, 0]).astype(np.float64)
        v2 = np.real(V[:, 1]).astype(np.float64)
        v1 = v1 / (np.linalg.norm(v1) + 1e-15)
        v2 = v2 / (np.linalg.norm(v2) + 1e-15)
        q = (v1 + 1j * v2).astype(np.complex128)
    elif V.ndim == 2:
        q = V[:, 0].astype(np.complex128)
    else:
        q = V.astype(np.complex128)

    # Phase computation
    theta = np.angle(q)
    phi = (2 * np.pi / K) * (0.5 + np.floor(K * theta / (2 * np.pi)) - K * theta / (2 * np.pi))
    order = np.argsort(phi)

    k = np.floor(K * theta / (2 * np.pi)).astype(int) % K
    roots = np.exp(2j * np.pi * np.arange(K) / K)

    # Initial candidate
    z = roots[k].astype(np.complex128)

    # Precompute Qz and diagonal
    is_sparse = sparse.issparse(Q)
    if is_sparse:
        Qz = np.asarray(Q.dot(z)).flatten()
        Q_csc = Q.tocsc() if not sparse.isspmatrix_csc(Q) else Q
        Q_diag = np.asarray(Q.diagonal()).flatten()
    else:
        Qz = Q @ z
        Q_csc = None
        Q_diag = np.diag(Q).copy()

    score = np.real(z.conj() @ Qz)
    best_score = score
    best_k = k.copy()

    # Sweep through n boundary points with O(degree) incremental updates
    for idx in order:
        old_z = z[idx]
        k[idx] = (k[idx] + 1) % K
        new_z = roots[k[idx]]
        dz = new_z - old_z

        # Score delta: Δ = 2·Re(conj(dz)·Qz[idx]) + |dz|²·Q[idx,idx]
        delta = 2 * np.real(np.conj(dz) * Qz[idx]) + np.abs(dz)**2 * Q_diag[idx]
        score += delta

        # Update z and Qz with O(degree) sparse update
        z[idx] = new_z
        if is_sparse:
            _sparse_update_Qz(Q_csc, Qz, idx, dz)
        else:
            Qz += Q[:, idx] * dz

        if score > best_score:
            best_score = score
            best_k = k.copy()

    elapsed = time.time() - t0
    best_z = roots[best_k]
    return float(best_score), best_k, best_z, elapsed


def run_hybrid(Q, V, K=3, greedy_seeds=(0, 1, 2)):
    """Run the full hybrid comparison.

    Returns a dict with all results:
      - rank1: score, time
      - greedy_random: best score across seeds, avg time, avg iterations
      - hybrid: score, total time, greedy iterations from warm start
      - improvement: hybrid score vs rank1, hybrid score vs best greedy random
    """
    n = Q.shape[0]
    roots = np.exp(2j * np.pi * np.arange(K) / K)

    # === Phase 1: Rank-1 ===
    r1_score, r1_k, r1_z, r1_time = rank1_phase_sweep(Q, V, K)

    # === Phase 2: Greedy from random (multiple seeds) ===
    greedy_scores = []
    greedy_times = []
    greedy_iters = []
    for seed in greedy_seeds:
        g_score, g_z, g_time, g_iters = greedy_cut(Q, K=K, seed=seed)
        greedy_scores.append(g_score)
        greedy_times.append(g_time)
        greedy_iters.append(g_iters)

    best_greedy = max(greedy_scores)
    avg_greedy_time = np.mean(greedy_times)
    avg_greedy_iters = np.mean(greedy_iters)

    # === Phase 3: Greedy warm-started from rank-1 ===
    hybrid_score, hybrid_z, hybrid_greedy_time, hybrid_iters = greedy_cut(
        Q, K=K, seed=0, init_k=r1_k
    )
    hybrid_total_time = r1_time + hybrid_greedy_time

    # === Results ===
    results = {
        "n": n,
        "K": K,
        "rank1": {
            "score": r1_score,
            "time": round(r1_time, 4),
        },
        "greedy_random": {
            "best_score": best_greedy,
            "scores": greedy_scores,
            "avg_time": round(avg_greedy_time, 4),
            "avg_iterations": round(avg_greedy_iters, 1),
            "seeds": list(greedy_seeds),
        },
        "hybrid": {
            "score": hybrid_score,
            "rank1_time": round(r1_time, 4),
            "greedy_time": round(hybrid_greedy_time, 4),
            "total_time": round(hybrid_total_time, 4),
            "greedy_iterations": hybrid_iters,
        },
        "comparison": {
            "hybrid_vs_rank1": round(hybrid_score - r1_score, 2),
            "hybrid_vs_greedy_best": round(hybrid_score - best_greedy, 2),
            "hybrid_wins_rank1": hybrid_score > r1_score,
            "hybrid_wins_greedy": hybrid_score > best_greedy,
            "hybrid_ties_greedy": hybrid_score == best_greedy,
            "greedy_iter_reduction": round(
                1 - hybrid_iters / avg_greedy_iters, 3
            ) if avg_greedy_iters > 0 else 0,
        },
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Hybrid Rank-1 + Greedy solver")
    parser.add_argument("--q_path", type=str, required=True)
    parser.add_argument("--v_path", type=str, required=True)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--greedy_seeds", type=str, default="0,1,2")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    Q = np.load(args.q_path).astype(np.float64)
    V = np.load(args.v_path)
    if V.ndim == 1:
        V = V.reshape(-1, 1)
    if not np.iscomplexobj(V):
        V = V.astype(np.complex128)

    n = Q.shape[0]
    seeds = tuple(int(s) for s in args.greedy_seeds.split(","))

    print(f"Instance: n={n}, K={args.K}")
    results = run_hybrid(Q, V, K=args.K, greedy_seeds=seeds)

    r = results
    print(f"{'Method':<25} {'Score':>10} {'Time':>10} {'Iters':>8}")
    print("-" * 55)
    print(f"{'Rank-1':<25} {r['rank1']['score']:>10.0f} {r['rank1']['time']:>9.4f}s {'—':>8}")
    print(f"{'Greedy (best of random)':<25} {r['greedy_random']['best_score']:>10.0f} {r['greedy_random']['avg_time']:>9.4f}s {r['greedy_random']['avg_iterations']:>7.0f}")
    print(f"{'Hybrid (R1 + Greedy)':<25} {r['hybrid']['score']:>10.0f} {r['hybrid']['total_time']:>9.4f}s {r['hybrid']['greedy_iterations']:>7.0f}")

    c = r["comparison"]
    if c["hybrid_wins_greedy"]:
        print(f"\n  ✓ Hybrid BEATS greedy by {c['hybrid_vs_greedy_best']:.0f}")
    elif c["hybrid_ties_greedy"]:
        print(f"\n  = Hybrid TIES greedy")
    else:
        print(f"\n  ✗ Greedy wins by {-c['hybrid_vs_greedy_best']:.0f}")

    if args.out:
        results["q_path"] = args.q_path
        results["v_path"] = args.v_path
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
