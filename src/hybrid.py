"""
Hybrid Rank-1 + Local Search solver for Max-K-Cut.

Strategy:
  1. Run the rank-1 phase sweep algorithm (O(n²), captures global spectral structure)
  2. Use the rank-1 solution as a warm start for greedy local search
  3. Compare: rank-1 alone, greedy from random, greedy from rank-1 warm start

The hypothesis: rank-1 provides a spectrally-informed starting point that guides
greedy to a better local optimum (or the same optimum faster).
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


def rank1_phase_sweep(Q, V, K=3):
    """CPU rank-1 phase sweep with incremental scoring. Returns (score, k, z, elapsed).

    Uses O(n) incremental score updates instead of O(n²) full recomputation.
    When vertex idx flips from z_old to z_new, the score change is:
        Δ = 2·Re((z_new - z_old)* · (Qz)[idx]) - Q[idx,idx]·(|z_new|² - |z_old|²)
    Since |z_new| = |z_old| = 1 for roots of unity, the last term vanishes.
    We also update Qz incrementally: Qz += Q[:, idx] · (z_new - z_old).
    """
    from scipy import sparse

    n = Q.shape[0]
    q = V[:, 0] if V.ndim == 2 else V

    t0 = time.time()

    # Phase computation
    theta = np.angle(q)
    phi = (2 * np.pi / K) * (0.5 + np.floor(K * theta / (2 * np.pi)) - K * theta / (2 * np.pi))
    order = np.argsort(phi)

    k = np.floor(K * theta / (2 * np.pi)).astype(int) % K
    roots = np.exp(2j * np.pi * np.arange(K) / K)

    # Initial candidate
    z = roots[k].astype(np.complex128)

    # Precompute Qz = Q @ z (O(nnz) for sparse, O(n²) for dense — done once)
    is_sparse = sparse.issparse(Q)
    if is_sparse:
        Qz = np.asarray(Q.dot(z)).flatten()
        # For column access during updates, need CSC format
        Q_csc = Q.tocsc() if not sparse.isspmatrix_csc(Q) else Q
    else:
        Qz = Q @ z

    score = np.real(z.conj() @ Qz)
    best_score = score
    best_k = k.copy()

    # Sweep through n boundary points with incremental updates
    for idx in order:
        old_z = z[idx]
        k[idx] = (k[idx] + 1) % K
        new_z = roots[k[idx]]
        dz = new_z - old_z

        # Incremental score update: Δ = 2·Re(dz* · Qz[idx]) + Q[idx,idx]·(|new|²-|old|²)
        # Since |new|=|old|=1, the diagonal term vanishes
        # But we also need to account for the self-interaction change:
        # Actually the full formula for Δ(z†Qz) when z[idx] changes by dz:
        #   Δ = 2·Re(conj(dz) · Qz[idx]) - Q[idx,idx]·(|new|²-|old|²) ... no
        # Let's derive carefully:
        #   new_score = (z+δ)† Q (z+δ) where δ is nonzero only at idx
        #   = z†Qz + δ†Qz + z†Qδ + δ†Qδ
        #   = score + conj(dz)·(Qz)[idx] + (Q[idx,:]·z)·conj(dz)... wait
        # Simpler: δ = dz · e_idx, so δ†Qz = conj(dz)·(Qz)[idx]
        # and z†Qδ = conj((Qz)[idx] ... no, z†Q·δ = (Q†z)†·δ = conj(Qz)[idx]·dz...
        # Since Q is real symmetric: z†Qδ = conj(dz)·(Qz)[idx] ... no
        # z†Qδ = sum_j conj(z_j) Q_j,idx dz = dz · sum_j conj(z_j) Q_j,idx = dz · conj(Qz[idx])
        # Wait, (Q†z)[idx] = sum_j Q[idx,j]*z[j] = (Qz)[idx] since Q real symmetric
        # So z†Qδ = dz · conj((Qz)[idx])... hmm let me just be explicit:
        # z†Qδ = sum_j conj(z_j) * Q[j,idx] * dz = dz * sum_j Q[idx,j] * conj(z_j) = dz * conj(Qz[idx])
        # δ†Qz = conj(dz) * (Qz)[idx]
        # δ†Qδ = |dz|² * Q[idx,idx]
        # Total: Δ = conj(dz)·(Qz)[idx] + dz·conj((Qz)[idx]) + |dz|²·Q[idx,idx]
        #          = 2·Re(conj(dz)·(Qz)[idx]) + |dz|²·Q[idx,idx]

        if is_sparse:
            Q_ii = Q[idx, idx]
        else:
            Q_ii = Q[idx, idx]

        delta = 2 * np.real(np.conj(dz) * Qz[idx]) + np.abs(dz)**2 * Q_ii
        score += delta

        # Update z and Qz incrementally
        z[idx] = new_z
        if is_sparse:
            col = np.asarray(Q_csc.getcol(idx).toarray()).flatten()
            Qz += col * dz
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
    t0 = time.time()
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
    parser.add_argument("--rank", type=int, default=1, help="Rank for V (uses first column)")
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
    print(f"Q: {args.q_path}")
    print(f"V: {args.v_path}")
    print()

    results = run_hybrid(Q, V, K=args.K, greedy_seeds=seeds)

    # Print summary
    r = results
    print(f"{'Method':<25} {'Score':>10} {'Time':>10} {'Iters':>8}")
    print("-" * 55)
    print(f"{'Rank-1':<25} {r['rank1']['score']:>10.0f} {r['rank1']['time']:>9.2f}s {'—':>8}")
    print(f"{'Greedy (best of random)':<25} {r['greedy_random']['best_score']:>10.0f} {r['greedy_random']['avg_time']:>9.2f}s {r['greedy_random']['avg_iterations']:>7.0f}")
    print(f"{'Hybrid (R1 + Greedy)':<25} {r['hybrid']['score']:>10.0f} {r['hybrid']['total_time']:>9.2f}s {r['hybrid']['greedy_iterations']:>7.0f}")
    print()

    c = r["comparison"]
    if c["hybrid_wins_greedy"]:
        print(f"  ✓ Hybrid BEATS greedy by {c['hybrid_vs_greedy_best']:.0f}")
    elif c["hybrid_ties_greedy"]:
        print(f"  = Hybrid TIES greedy")
    else:
        print(f"  ✗ Greedy wins by {-c['hybrid_vs_greedy_best']:.0f}")

    print(f"  Hybrid vs Rank-1: +{c['hybrid_vs_rank1']:.0f}")
    print(f"  Greedy iterations reduced by {c['greedy_iter_reduction']*100:.0f}%")

    if args.out:
        results["q_path"] = args.q_path
        results["v_path"] = args.v_path
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
