"""
Systematic sweep: DSG vs R2G across graph families and sizes.

Tests the hypothesis that spectral methods (R2G) beat DSatur (DSG) on
graphs with global community structure that is invisible to 1-hop
neighborhoods, especially at sparse densities near the spectral
detection threshold.

Families tested:
1. Sparse SBM: planted 3-community structure with decreasing edge density.
   As p_in/p_out → threshold, local signal weakens but spectral signal persists.
2. Random geometric graphs: points in 2D with proximity edges.
   Spatial structure is both local and global.
3. Watts-Strogatz small-world: regular lattice + random rewiring.
   Increasing rewiring disrupts local regularity.
4. Powerlaw cluster: power-law degree distribution with clustering.
   Hub structure dominates.

For each (family, n, parameter), we run:
  - DSG: DSatur construction (no improve) → greedy
  - R2G: Rank-2 spectral sampling (100K samples) → greedy
  - Hybrid: Rank-1 → greedy
  - SA: Simulated annealing (budget-matched)

Seeds: 5 random seeds per configuration for variance estimation.

Usage:
    python experiments/run_dsatur_vs_r2g_sweep.py [--num_workers 8]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from baselines import dsatur_cut, greedy_cut_incremental, sa_cut
from hybrid import rank1_phase_sweep
from randomized_rank_r_cpu_sparse import (
    score_batch_cpu_sparse, generate_random_indices,
    extract_edge_list, load_laplacian,
)
from utils import compute_vtilde


def quick_r2g(L, V, n_samples=100000, seed=0):
    """Run rank-2 sampling + greedy. Returns (r2g_score, r2_score)."""
    n = L.shape[0]
    K = 3
    roots = np.exp(2j * np.pi * np.arange(K) / K)
    roots_conj = np.conj(roots)
    V_tilde = compute_vtilde(V).astype(np.float64)
    edge_list = extract_edge_list(L)
    Kn = K * n
    rng = np.random.default_rng(seed)

    best_r2 = -np.inf
    best_r2_k = None
    batch_size = min(10000, n_samples)
    for _ in range(n_samples // batch_size):
        I_batch = generate_random_indices(Kn, 3, batch_size, rng)
        sc, k, z, feas = score_batch_cpu_sparse(
            L, V, V_tilde, I_batch, 2, K, roots, roots_conj,
            edge_list=edge_list,
        )
        if k is not None and sc > best_r2:
            best_r2 = sc
            best_r2_k = k

    if best_r2_k is not None:
        r2g_score, _, _, _ = greedy_cut_incremental(L, K=K, seed=0, init_k=best_r2_k)
        return r2g_score, best_r2
    return 0, 0


def generate_graph(family, n, param, seed):
    """Generate a graph and return its sparse Laplacian."""
    import networkx as nx

    if family == "sbm_sparse":
        # Sparse SBM with 3 communities
        # param = (p_in, p_out)
        p_in, p_out = param
        block_size = n // 3
        sizes = [block_size, block_size, n - 2 * block_size]
        probs = [
            [p_in, p_out, p_out],
            [p_out, p_in, p_out],
            [p_out, p_out, p_in],
        ]
        G = nx.stochastic_block_model(sizes, probs, seed=seed)

    elif family == "random_geometric":
        # Random geometric graph in 2D
        # param = radius
        radius = param
        G = nx.random_geometric_graph(n, radius, seed=seed)
        # Take largest connected component
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            G = nx.convert_node_labels_to_integers(G)

    elif family == "watts_strogatz":
        # Small-world graph
        # param = (k_neighbors, p_rewire)
        k_ws, p_ws = param
        if k_ws % 2 == 1:
            k_ws -= 1
        G = nx.watts_strogatz_graph(n, max(k_ws, 2), p_ws, seed=seed)

    elif family == "powerlaw_cluster":
        # Powerlaw cluster graph
        # param = (m_edges, p_triangle)
        m, p_tri = param
        G = nx.powerlaw_cluster_graph(n, m, p_tri, seed=seed)

    else:
        raise ValueError(f"Unknown family: {family}")

    L = nx.laplacian_matrix(G).astype(np.float64).tocsr()
    return L, G.number_of_nodes(), G.number_of_edges()


# Configuration: (family, n_values, param_values, description)
CONFIGS = [
    # Sparse SBM: sweep p_in with p_out = p_in/10
    ("sbm_sparse", [500, 1000, 3000, 5000, 10000],
     [(0.05, 0.005), (0.03, 0.003), (0.02, 0.002), (0.015, 0.0015), (0.01, 0.001)],
     "SBM: decreasing density, fixed community SNR"),

    # Random geometric: sweep radius
    ("random_geometric", [500, 1000, 3000, 5000],
     [0.12, 0.08, 0.06],
     "Random geometric: decreasing radius (sparser)"),

    # Watts-Strogatz: sweep rewiring probability
    ("watts_strogatz", [1000, 3000, 5000, 10000],
     [(6, 0.05), (6, 0.1), (6, 0.3), (6, 0.5)],
     "Small-world: increasing rewiring"),

    # Powerlaw cluster: sweep parameters
    ("powerlaw_cluster", [1000, 3000, 5000, 10000],
     [(3, 0.3), (3, 0.5), (5, 0.3)],
     "Powerlaw cluster: varying density and clustering"),
]

SEEDS = [0, 1, 2, 3, 4]
R2_SAMPLES = 100000  # per run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="dsatur_sweep_results")
    parser.add_argument("--families", default="all",
                        help="Comma-separated: sbm_sparse,random_geometric,watts_strogatz,powerlaw_cluster")
    parser.add_argument("--r2_samples", type=int, default=R2_SAMPLES)
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--sa_budget", type=int, default=120,
                        help="SA time budget in seconds")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    families_filter = args.families.split(",") if args.families != "all" else None

    for family, n_values, param_values, desc in CONFIGS:
        if families_filter and family not in families_filter:
            continue

        print(f"\n{'='*70}")
        print(f"  {desc}")
        print(f"{'='*70}")

        for param in param_values:
            for n in n_values:
                tag = f"{family}_n{n}_p{param}"
                out_path = os.path.join(args.out_dir, f"{tag}.json")

                if os.path.exists(out_path):
                    print(f"SKIP {tag}")
                    continue

                if args.dry_run:
                    print(f"  WOULD RUN: {tag}")
                    continue

                dsg_scores = []
                r2g_scores = []
                hyb_scores = []
                sa_scores = []
                actual_ns = []

                for seed in seeds:
                    try:
                        L, actual_n, num_edges = generate_graph(family, n, param, seed)
                    except Exception as e:
                        print(f"  ERROR generating {tag} seed={seed}: {e}")
                        continue

                    actual_ns.append(actual_n)

                    # Eigensolve
                    try:
                        eigvals, eigvecs = eigsh(L, k=2, which="LM",
                                                 maxiter=3000, tol=1e-6)
                    except Exception:
                        eigvals, eigvecs = eigsh(L, k=2, which="LM",
                                                 maxiter=5000, tol=1e-4)
                    idx = np.argsort(eigvals)[::-1]
                    eigvals = eigvals[idx]
                    eigvecs = eigvecs[:, idx]
                    V = (eigvecs * np.sqrt(np.maximum(eigvals, 0))).astype(
                        np.complex128
                    )

                    # DSG
                    _, ds_k, _, _ = dsatur_cut(L, K=3, improve=False)
                    dsg, _, _, _ = greedy_cut_incremental(L, K=3, seed=0,
                                                          init_k=ds_k)
                    dsg_scores.append(float(dsg))

                    # R2G
                    r2g, _ = quick_r2g(L, V, n_samples=args.r2_samples,
                                       seed=seed)
                    r2g_scores.append(float(r2g))

                    # Hybrid
                    V1 = eigvecs[:, 0:1].astype(np.complex128)
                    _, r1_k, _, _ = rank1_phase_sweep(L, V1, K=3)
                    hyb, _, _, _ = greedy_cut_incremental(L, K=3, seed=0,
                                                          init_k=r1_k)
                    hyb_scores.append(float(hyb))

                    # SA
                    sa_iters = max(50 * actual_n, 500000)
                    sa_sc, _, _, _ = sa_cut(L, K=3, seed=seed,
                                           max_iters=sa_iters,
                                           max_time=args.sa_budget)
                    sa_scores.append(float(sa_sc))

                if not dsg_scores:
                    continue

                result = {
                    "family": family,
                    "n_requested": n,
                    "n_actual": int(np.mean(actual_ns)),
                    "param": str(param),
                    "seeds": len(seeds),
                    "dsg_mean": float(np.mean(dsg_scores)),
                    "dsg_std": float(np.std(dsg_scores)),
                    "r2g_mean": float(np.mean(r2g_scores)),
                    "r2g_std": float(np.std(r2g_scores)),
                    "hybrid_mean": float(np.mean(hyb_scores)),
                    "hybrid_std": float(np.std(hyb_scores)),
                    "sa_mean": float(np.mean(sa_scores)),
                    "sa_std": float(np.std(sa_scores)),
                    "r2g_over_dsg": float(np.mean(r2g_scores) /
                                          max(np.mean(dsg_scores), 1) - 1),
                    "r2g_wins": int(sum(1 for r, d in
                                        zip(r2g_scores, dsg_scores) if r > d)),
                }

                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)

                r2g_m = np.mean(r2g_scores)
                dsg_m = np.mean(dsg_scores)
                gap = (r2g_m / max(dsg_m, 1) - 1) * 100
                wins = result["r2g_wins"]
                print(
                    f"  {tag}: DSG={dsg_m:,.0f} R2G={r2g_m:,.0f} "
                    f"gap={gap:+.2f}% R2G wins {wins}/{len(seeds)} seeds",
                    flush=True,
                )

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    results = []
    for fname in sorted(os.listdir(args.out_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(args.out_dir, fname)) as f:
            results.append(json.load(f))

    r2g_wins_total = sum(r["r2g_wins"] for r in results)
    total_seeds = sum(r["seeds"] for r in results)
    r2g_better_configs = sum(1 for r in results if r["r2g_over_dsg"] > 0)
    print(f"  R2G wins {r2g_wins_total}/{total_seeds} individual seeds")
    print(f"  R2G better on {r2g_better_configs}/{len(results)} configurations (mean)")

    # Per-family breakdown
    for family in ["sbm_sparse", "random_geometric", "watts_strogatz",
                    "powerlaw_cluster"]:
        fam_results = [r for r in results if r["family"] == family]
        if not fam_results:
            continue
        wins = sum(r["r2g_wins"] for r in fam_results)
        total = sum(r["seeds"] for r in fam_results)
        avg_gap = np.mean([r["r2g_over_dsg"] * 100 for r in fam_results])
        print(f"  {family}: R2G wins {wins}/{total} seeds, "
              f"avg gap {avg_gap:+.2f}%")


if __name__ == "__main__":
    main()
