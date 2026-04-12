"""
Phase 3: Large-n frontier for randomized rank-2.

Tests at n=2000, 5000, 10000 where exact rank-2 is infeasible.
Compares randomized rank-2 (with fixed sample budgets) against
greedy, SA, and hybrid (rank-1 + greedy).

Usage:
    python experiments/run_large_n_frontier.py [--family regular] [--num_gpus 4]
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


SAMPLE_BUDGETS = [1_000_000, 10_000_000, 100_000_000]
GRAPH_SEEDS = [0, 1, 2]
RANDOM_SEEDS = [0, 1, 2]

CONFIGS = {
    "regular": [2000, 5000, 10000],
    "torus": [2000, 5000, 10000],
}


def generate_instance(family, n, seed, out_dir):
    """Generate Q.npy and V.npy (top-2 eigenvectors) for a graph instance."""
    import networkx as nx

    q_path = os.path.join(out_dir, f"Q_{family}_{n}_seed_{seed}.npy")
    v_path = os.path.join(out_dir, f"V_{family}_{n}_seed_{seed}.npy")

    if os.path.exists(q_path) and os.path.exists(v_path):
        return q_path, v_path

    print(f"  Generating {family} n={n} seed={seed}...")

    if family == "regular":
        if n % 2 != 0:
            n += 1
        G = nx.random_regular_graph(5, n, seed=seed)
    elif family == "torus":
        q = int(np.sqrt(n / 2))
        p = n // q
        G = nx.grid_2d_graph(p, q, periodic=True)
        n = p * q

    L = nx.laplacian_matrix(G).astype(np.float64).tocsr()
    actual_n = L.shape[0]

    # Top-2 eigenvectors
    eigvals, eigvecs = eigsh(L, k=2, which='LM', maxiter=2000, tol=1e-6)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    V = (eigvecs * np.sqrt(np.maximum(eigvals, 0))).astype(np.complex128)

    # Save as dense Q for GPU (feasible up to n=10K: 800MB in float32)
    Q = L.toarray().astype(np.float64)

    os.makedirs(out_dir, exist_ok=True)
    np.save(q_path, Q)
    np.save(v_path, V)
    print(f"    Saved: n={actual_n}, edges={G.number_of_edges()}")

    return q_path, v_path


def run_baselines(L_sparse, n, K=3, seed=42, time_budget=600):
    """Run rank-1, hybrid, greedy, SA on a sparse Laplacian."""
    from baselines import greedy_cut_incremental, sa_cut
    from hybrid import rank1_phase_sweep

    results = {}
    roots = np.exp(2j * np.pi * np.arange(K) / K)

    # Eigensolve for rank-1
    t0 = time.time()
    eigvals, eigvecs = eigsh(L_sparse, k=2, which='LM', maxiter=2000, tol=1e-6)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    V = (eigvecs * np.sqrt(np.maximum(eigvals, 0))).astype(np.complex128)
    eig_time = time.time() - t0

    # Rank-1
    r1_score, r1_k, r1_z, r1_time = rank1_phase_sweep(L_sparse, V, K=K)
    results["rank1"] = {"score": r1_score, "time": round(eig_time + r1_time, 2)}

    # Greedy
    g_score, _, g_time, g_iters = greedy_cut_incremental(L_sparse, K=K, seed=seed, max_time=time_budget)
    results["greedy"] = {"score": g_score, "time": round(g_time, 2), "iterations": g_iters}

    # Hybrid
    hw_score, _, hw_time, hw_iters = greedy_cut_incremental(L_sparse, K=K, seed=seed, init_k=r1_k, max_time=time_budget)
    results["hybrid"] = {"score": hw_score, "time": round(eig_time + r1_time + hw_time, 2), "iterations": hw_iters}

    # SA
    sa_iters = max(50 * n, 500000)
    sa_score, _, sa_time, sa_acc = sa_cut(L_sparse, K=K, seed=seed, max_iters=sa_iters, max_time=time_budget)
    results["sa"] = {"score": sa_score, "time": round(sa_time, 2)}

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=str, default="regular")
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--instance_dir", type=str, default="large_instances")
    parser.add_argument("--out_dir", type=str, default="large_n_results")
    parser.add_argument("--baseline_budget", type=int, default=600,
                        help="Time budget for greedy/SA in seconds")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    family = args.family

    if family not in CONFIGS:
        print(f"Unknown family: {family}")
        return

    t_start = time.time()
    total_runs = 0

    for n in CONFIGS[family]:
        for gseed in GRAPH_SEEDS:
            tag = f"{family}_{n}_seed{gseed}"

            # Generate instance
            q_path, v_path = generate_instance(family, n, gseed, args.instance_dir)

            # Run baselines (once per instance)
            baseline_path = os.path.join(args.out_dir, f"{tag}_baselines.json")
            if not os.path.exists(baseline_path):
                print(f"\n{'='*60}")
                print(f"  Baselines: {tag}")
                print(f"{'='*60}")

                import networkx as nx
                if family == "regular":
                    nn = n if n % 2 == 0 else n + 1
                    G = nx.random_regular_graph(5, nn, seed=gseed)
                elif family == "torus":
                    q = int(np.sqrt(n / 2))
                    p = n // q
                    G = nx.grid_2d_graph(p, q, periodic=True)
                L_sparse = nx.laplacian_matrix(G).astype(np.float64).tocsr()
                actual_n = L_sparse.shape[0]

                bl = run_baselines(L_sparse, actual_n, seed=gseed, time_budget=args.baseline_budget)
                bl["n"] = actual_n
                bl["family"] = family
                bl["seed"] = gseed
                with open(baseline_path, "w") as f:
                    json.dump(bl, f, indent=2)
                print(f"  R1={bl['rank1']['score']:.0f} G={bl['greedy']['score']:.0f} "
                      f"H={bl['hybrid']['score']:.0f} SA={bl['sa']['score']:.0f}")
            else:
                print(f"SKIP baselines: {tag}")

            # Run randomized rank-2 at each sample budget
            for max_samples in SAMPLE_BUDGETS:
                for rseed in RANDOM_SEEDS:
                    rtag = f"{tag}_samples{max_samples}_rseed{rseed}"
                    out_path = os.path.join(args.out_dir, f"{rtag}.json")

                    if os.path.exists(out_path):
                        continue

                    print(f"\n{'='*60}")
                    print(f"  Randomized: {rtag}")
                    print(f"{'='*60}")

                    cmd = (
                        f"python3 -u src/randomized_rank_r_gpu.py "
                        f"--q_path {q_path} --v_path {v_path} "
                        f"--rank 2 --K 3 "
                        f"--max_samples {max_samples} --seed {rseed} "
                        f"--num_gpus {args.num_gpus} --chunk_size 50000 "
                        f"--out {out_path}"
                    )
                    ret = os.system(cmd)
                    if ret == 0:
                        total_runs += 1
                    else:
                        print(f"  ERROR: exit code {ret}")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"DONE: {total_runs} randomized runs in {elapsed/3600:.1f} hours")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
