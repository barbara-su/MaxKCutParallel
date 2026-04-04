"""
Extreme-scale hybrid experiments.
Compares: Pure Greedy vs Hybrid (Rank-1 + Greedy) at n=10K to 1M.
Same graph families and seeds as the extreme-scale baseline experiments.
Reports: score, time, iterations for both.
"""
import json
import os
import sys
import time

os.environ.setdefault("TMPDIR", os.environ.get("TMPDIR", "/tmp"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from baselines import greedy_cut_incremental
from hybrid import rank1_phase_sweep


def generate_graph(n, family, seed=42):
    import networkx as nx
    if family == "regular":
        if n % 2 != 0: n += 1
        G = nx.random_regular_graph(5, n, seed=seed)
    elif family == "erdos_renyi":
        G = nx.erdos_renyi_graph(n, 10.0 / n, seed=seed)
    elif family == "torus":
        q = int(np.sqrt(n / 2))
        p = n // q
        G = nx.grid_2d_graph(p, q, periodic=True)
        n = p * q
    elif family == "sbm":
        bs = n // 3
        sizes = [bs, bs, n - 2 * bs]
        probs = [[0.3, 0.01, 0.01], [0.01, 0.3, 0.01], [0.01, 0.01, 0.3]]
        G = nx.stochastic_block_model(sizes, probs, seed=seed)
    L = nx.laplacian_matrix(G).astype(np.float64).tocsr()
    return L, G.number_of_nodes(), G.number_of_edges()


def run_one(n, family, seed, time_budget):
    print(f"\n{'='*60}")
    print(f"  {family} n={n:,} seed={seed}")
    print(f"{'='*60}")

    t0 = time.time()
    L, actual_n, edges = generate_graph(n, family, seed)
    gen_time = time.time() - t0
    print(f"  Graph: {actual_n:,} nodes, {edges:,} edges ({gen_time:.1f}s)")

    eigval, eigvec = eigsh(L, k=1, which='LM', maxiter=2000, tol=1e-6)
    V = eigvec.astype(np.complex128)
    K = 3

    results = {"family": family, "n": actual_n, "seed": seed, "edges": edges}

    # --- Pure Greedy ---
    print(f"  Pure Greedy...", flush=True)
    t_g0 = time.time()
    g_score, _, g_time, g_iters = greedy_cut_incremental(
        L, K=K, seed=seed, max_time=time_budget, max_iters=20
    )
    g_total = time.time() - t_g0
    results["greedy"] = {"score": g_score, "time": round(g_total, 2), "iterations": g_iters}
    print(f"    Score={g_score:.0f}, Time={g_total:.1f}s, Iters={g_iters}")

    # --- Hybrid: Rank-1 + Greedy ---
    print(f"  Hybrid (Rank-1 + Greedy)...", flush=True)
    t_h0 = time.time()

    # Step 1: Rank-1
    r1_score, r1_k, r1_z, r1_time = rank1_phase_sweep(L, V, K=K)

    # Step 2: Greedy warm-started from rank-1
    hw_score, _, hw_greedy_time, hw_iters = greedy_cut_incremental(
        L, K=K, seed=seed, init_k=r1_k, max_time=time_budget - r1_time, max_iters=20
    )
    h_total = time.time() - t_h0

    results["hybrid"] = {
        "score": hw_score,
        "total_time": round(h_total, 2),
        "rank1_time": round(r1_time, 2),
        "greedy_time": round(hw_greedy_time, 2),
        "rank1_score": r1_score,
        "greedy_iterations": hw_iters,
    }
    print(f"    Score={hw_score:.0f}, Total={h_total:.1f}s (R1={r1_time:.1f}s + G={hw_greedy_time:.1f}s), Iters={hw_iters}")

    # --- Comparison ---
    score_diff = hw_score - g_score
    if g_total > 0:
        speedup = g_total / h_total if h_total > 0 else float('inf')
    else:
        speedup = 1.0
    iter_diff = g_iters - hw_iters

    results["comparison"] = {
        "score_diff": round(score_diff, 1),
        "hybrid_wins_score": hw_score > g_score,
        "hybrid_ties_score": hw_score == g_score,
        "speedup": round(speedup, 2),
        "hybrid_faster": h_total < g_total,
        "iteration_saved": iter_diff,
    }

    winner_score = "Hybrid" if hw_score > g_score else ("Tie" if hw_score == g_score else "Greedy")
    winner_time = "Hybrid" if h_total < g_total else "Greedy"
    print(f"  → Score: {winner_score} ({score_diff:+.0f}), Time: {winner_time} ({speedup:.2f}x), Iters saved: {iter_diff}")

    return results


def main():
    sizes = [10000, 50000, 100000, 500000, 1000000]
    families = ["regular", "erdos_renyi", "torus", "sbm"]
    seeds = [0, 1, 2]
    budgets = {10000: 600, 50000: 600, 100000: 1200, 500000: 1800, 1000000: 7200}

    out_dir = os.path.join(os.path.dirname(__file__), "hybrid_extreme_results")
    os.makedirs(out_dir, exist_ok=True)

    for n in sizes:
        for family in families:
            for seed in seeds:
                tag = f"{family}_{n}_seed_{seed}"
                out_path = os.path.join(out_dir, f"{tag}.json")
                if os.path.exists(out_path):
                    print(f"SKIP: {tag}")
                    continue
                try:
                    result = run_one(n, family, seed, budgets[n])
                    with open(out_path, "w") as f:
                        json.dump(result, f, indent=2)
                except Exception as e:
                    print(f"ERROR: {tag}: {e}")

    print(f"\nAll done.")


if __name__ == "__main__":
    main()
