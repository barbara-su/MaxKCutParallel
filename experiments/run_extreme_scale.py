"""
Extreme-scale rank-1 experiments.
Compares: Random, Rank-1 (incremental), Greedy (incremental), SA, Tabu
on graphs from n=10K to n=1M across 4 families.
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

from baselines import random_cut, greedy_cut_incremental, sa_cut, tabu_cut
from hybrid import rank1_phase_sweep


def generate_graph(n, family, seed=42):
    """Generate sparse Laplacian for a given graph family."""
    import networkx as nx

    if family == "regular":
        if n % 2 != 0:
            n += 1
        G = nx.random_regular_graph(5, n, seed=seed)
    elif family == "erdos_renyi":
        p = 10.0 / n  # ~10 edges per node on average
        G = nx.erdos_renyi_graph(n, p, seed=seed)
    elif family == "torus":
        # Find p, q such that p*q ~ n and p ~ 2*q
        q = int(np.sqrt(n / 2))
        p = n // q
        actual_n = p * q
        G = nx.grid_2d_graph(p, q, periodic=True)
        n = actual_n
    elif family == "sbm":
        block_size = n // 3
        sizes = [block_size, block_size, n - 2 * block_size]
        probs = [[0.3, 0.01, 0.01], [0.01, 0.3, 0.01], [0.01, 0.01, 0.3]]
        G = nx.stochastic_block_model(sizes, probs, seed=seed)
    else:
        raise ValueError(f"Unknown family: {family}")

    L = nx.laplacian_matrix(G).astype(np.float64).tocsr()
    return L, G.number_of_nodes(), G.number_of_edges()


def run_experiment(n, family, seed, time_budget):
    """Run all methods on one instance."""
    print(f"\n{'='*60}")
    print(f"  {family} n={n:,} seed={seed} (budget={time_budget}s)")
    print(f"{'='*60}")

    # Generate graph
    t0 = time.time()
    L, actual_n, num_edges = generate_graph(n, family, seed)
    gen_time = time.time() - t0
    print(f"  Graph: {actual_n:,} nodes, {num_edges:,} edges, generated in {gen_time:.1f}s")

    # Eigensolve for rank-1
    t0 = time.time()
    eigval, eigvec = eigsh(L, k=1, which='LM', maxiter=2000, tol=1e-6)
    eig_time = time.time() - t0
    print(f"  Eigensolve: {eig_time:.1f}s (lambda_1={eigval[0]:.4f})")

    V = eigvec.astype(np.complex128)
    K = 3
    roots = np.exp(2j * np.pi * np.arange(K) / K)

    results = {
        "family": family, "n": actual_n, "seed": seed,
        "edges": num_edges, "gen_time": gen_time, "eig_time": eig_time,
    }

    # --- Random ---
    print(f"  Running Random...", flush=True)
    r_score, _, r_time = random_cut(L.toarray() if actual_n <= 5000 else L, K=K, num_trials=actual_n + 1, seed=seed)
    # For large n, random_cut needs sparse support. Let me compute manually:
    if actual_n > 5000:
        rng = np.random.RandomState(seed)
        best_rs = -np.inf
        for _ in range(min(actual_n + 1, 1000)):  # Cap trials for large n
            k_rand = rng.randint(0, K, size=actual_n)
            z_rand = roots[k_rand]
            Lz = L.dot(z_rand)
            s = np.real(z_rand.conj() @ Lz)
            if s > best_rs:
                best_rs = s
        r_score = float(best_rs)
        r_time = time.time() - t0 - gen_time - eig_time  # rough
    results["random"] = {"score": r_score, "time": round(r_time, 2)}
    print(f"    Score={r_score:.0f}, Time={r_time:.1f}s")

    # --- Rank-1 (incremental) ---
    print(f"  Running Rank-1...", flush=True)
    r1_score, r1_k, r1_z, r1_time = rank1_phase_sweep(L, V, K=K)
    results["rank1"] = {"score": r1_score, "time": round(r1_time, 2)}
    print(f"    Score={r1_score:.0f}, Time={r1_time:.1f}s")

    # --- Greedy (incremental, vectorized) ---
    print(f"  Running Greedy...", flush=True)
    g_score, g_z, g_time, g_iters = greedy_cut_incremental(
        L, K=K, seed=seed, max_time=time_budget, max_iters=20
    )
    results["greedy"] = {"score": g_score, "time": round(g_time, 2), "iterations": g_iters}
    print(f"    Score={g_score:.0f}, Time={g_time:.1f}s, Iters={g_iters}")

    # --- SA (scale iterations with n, generous budget) ---
    print(f"  Running SA...", flush=True)
    sa_iters = max(50 * actual_n, 1000000)
    sa_score, _, sa_time, sa_acc = sa_cut(
        L, K=K, seed=seed, max_iters=sa_iters, max_time=time_budget,
        cooling=0.99999  # slower cooling for better quality
    )
    results["sa"] = {"score": sa_score, "time": round(sa_time, 2), "accepted": sa_acc}
    print(f"    Score={sa_score:.0f}, Time={sa_time:.1f}s, Accepted={sa_acc}")

    # --- Tabu (only for n <= 50K due to O(n) per iteration) ---
    if actual_n <= 50000:
        print(f"  Running Tabu...", flush=True)
        tabu_iters = max(5 * actual_n, 50000)
        tb_score, _, tb_time, tb_iters = tabu_cut(
            L, K=K, seed=seed, max_iters=tabu_iters, max_time=min(time_budget, 300)
        )
        results["tabu"] = {"score": tb_score, "time": round(tb_time, 2), "iterations": tb_iters}
        print(f"    Score={tb_score:.0f}, Time={tb_time:.1f}s, Iters={tb_iters}")
    else:
        print(f"  Tabu: SKIPPED (n too large)")
        results["tabu"] = {"score": None, "time": None, "skipped": True}

    # Summary
    methods = ["random", "rank1", "greedy", "sa", "tabu"]
    scores = {m: results[m]["score"] for m in methods if results[m]["score"] is not None}
    winner = max(scores, key=scores.get)
    print(f"\n  Winner: {winner} (score={scores[winner]:.0f})")

    return results


def main():
    sizes = [10000, 50000, 100000, 500000, 1000000]
    families = ["regular", "erdos_renyi", "torus", "sbm"]
    seeds = [0, 1, 2]

    # Time budgets
    budgets = {
        10000: 600,      # 10 min
        50000: 600,      # 10 min
        100000: 1200,    # 20 min
        500000: 1800,    # 30 min
        1000000: 7200,   # 2 hours
    }

    out_dir = os.path.join(os.path.dirname(__file__), "extreme_results")
    os.makedirs(out_dir, exist_ok=True)

    all_results = []

    for n in sizes:
        for family in families:
            for seed in seeds:
                tag = f"{family}_{n}_seed_{seed}"
                out_path = os.path.join(out_dir, f"{tag}.json")

                if os.path.exists(out_path):
                    print(f"SKIP: {tag} (exists)")
                    continue

                # Skip SBM for n>=50K (too dense, OOM)
                if family == "sbm" and n >= 50000:
                    print(f"SKIP: {tag} (SBM too dense at this scale)")
                    continue

                # Skip ER for n>=500K (graph generation too slow)
                if family == "erdos_renyi" and n >= 500000:
                    print(f"SKIP: {tag} (ER generation too slow at this scale)")
                    continue

                try:
                    result = run_experiment(n, family, seed, budgets[n])
                    with open(out_path, "w") as f:
                        json.dump(result, f, indent=2)
                    all_results.append(result)
                except Exception as e:
                    print(f"ERROR: {tag}: {e}")

    print(f"\n\nAll done. {len(all_results)} experiments completed.")


if __name__ == "__main__":
    main()
