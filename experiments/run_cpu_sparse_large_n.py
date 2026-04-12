"""
Phase 3B: Large-n randomized rank-2 via CPU sparse solver.

Runs randomized rank-2 at n=100K, 500K, 1M on the SAME instances as
`fixed_results/` (regular + torus, 3 graph seeds each). This is the regime
where GPU dense Q matrix OOMs, so CPU sparse is the only viable path.

Compares the randomized rank-2 score against rank-1 / hybrid / greedy / SA
baselines from `fixed_results/regular_{n}_seed_{s}.json` and
`fixed_results/torus_{n}_seed_{s}.json`.

Usage:
    python experiments/run_cpu_sparse_large_n.py [--family regular] \\
        [--num_workers 8] [--sizes 100000,500000,1000000] \\
        [--sample_budgets 1000000]

The script:
    1. (Re)generates L for each (family, n, seed) — matching `run_extreme_scale.py`
    2. Saves L as scipy.sparse.npz (reused across random seeds)
    3. Computes top-2 eigenvectors once, saves V.npy
    4. For each sample budget and random seed, runs the CPU sparse solver
    5. Writes a result JSON per run and a summary

Outputs:
    large_instances_sparse/L_{family}_{n}_seed_{s}.npz
    large_instances_sparse/V_{family}_{n}_seed_{s}.npy
    cpu_sparse_results/{tag}.json
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


DEFAULT_SIZES = [100000, 500000, 1000000]
DEFAULT_FAMILIES = ["regular", "torus"]
DEFAULT_GRAPH_SEEDS = [0, 1, 2]
DEFAULT_RANDOM_SEEDS = [0, 1, 2]
DEFAULT_SAMPLE_BUDGETS = [1_000_000]


def generate_graph(n, family, seed):
    """Match run_extreme_scale.generate_graph exactly so instances reproduce."""
    import networkx as nx

    if family == "regular":
        if n % 2 != 0:
            n += 1
        G = nx.random_regular_graph(5, n, seed=seed)
    elif family == "torus":
        q = int(np.sqrt(n / 2))
        p = n // q
        G = nx.grid_2d_graph(p, q, periodic=True)
    elif family == "erdos_renyi":
        p = 10.0 / n  # ~10 edges per node average (matches run_extreme_scale)
        G = nx.erdos_renyi_graph(n, p, seed=seed)
    else:
        raise ValueError(f"Unsupported family: {family}")

    L = nx.laplacian_matrix(G).astype(np.float64).tocsr()
    return L, G.number_of_nodes(), G.number_of_edges()


def ensure_instance(family, n, seed, inst_dir):
    """Generate (or load) L.npz and V.npy for a single instance.

    Returns (l_path, v_path, actual_n).
    """
    tag = f"{family}_{n}_seed_{seed}"
    l_path = os.path.join(inst_dir, f"L_{tag}.npz")
    v_path = os.path.join(inst_dir, f"V_{tag}.npy")
    meta_path = os.path.join(inst_dir, f"meta_{tag}.json")

    if os.path.exists(l_path) and os.path.exists(v_path) and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return l_path, v_path, meta["n"]

    os.makedirs(inst_dir, exist_ok=True)
    print(f"  Generating {tag}...", flush=True)

    t0 = time.time()
    L, actual_n, num_edges = generate_graph(n, family, seed)
    t_gen = time.time() - t0
    print(
        f"    Graph: n={actual_n:,}, edges={num_edges:,} "
        f"(gen {t_gen:.1f}s)",
        flush=True,
    )

    sparse.save_npz(l_path, L)

    t0 = time.time()
    eigvals, eigvecs = eigsh(L, k=2, which="LM", maxiter=2000, tol=1e-6)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    V = (eigvecs * np.sqrt(np.maximum(eigvals, 0))).astype(np.complex128)
    t_eig = time.time() - t0
    print(
        f"    Eigensolve: {t_eig:.1f}s, top eigvals={eigvals.tolist()}",
        flush=True,
    )

    np.save(v_path, V)
    with open(meta_path, "w") as f:
        json.dump(
            {
                "family": family,
                "n": actual_n,
                "edges": num_edges,
                "eigvals": eigvals.tolist(),
                "gen_time": round(t_gen, 2),
                "eig_time": round(t_eig, 2),
            },
            f,
            indent=2,
        )
    return l_path, v_path, actual_n


def run_solver(l_path, v_path, max_samples, rseed, num_workers, out_path,
               batch_size=0, solver_script=None):
    """Launch the CPU sparse solver via subprocess.

    batch_size=0 means auto-scale (recommended — the solver picks a safe
    value based on n to bound memory).
    """
    if solver_script is None:
        here = os.path.dirname(os.path.abspath(__file__))
        solver_script = os.path.join(here, "..", "src",
                                     "randomized_rank_r_cpu_sparse.py")

    cmd = [
        sys.executable,
        "-u",
        solver_script,
        "--q_path", l_path,
        "--v_path", v_path,
        "--rank", "2",
        "--K", "3",
        "--max_samples", str(max_samples),
        "--seed", str(rseed),
        "--num_workers", str(num_workers),
        "--batch_size", str(batch_size),
        "--out", out_path,
    ]
    print("    $ " + " ".join(cmd), flush=True)
    # Cap BLAS threads so workers don't oversubscribe the cores
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    ret = subprocess.call(cmd, env=env)
    return ret == 0


def load_baseline(family, n, seed, baseline_dir):
    """Load the matching baseline JSON from fixed_results/."""
    path = os.path.join(baseline_dir, f"{family}_{n}_seed_{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=str, default="all",
                        choices=["all", "regular", "torus", "erdos_renyi"])
    parser.add_argument("--sizes", type=str,
                        default=",".join(str(s) for s in DEFAULT_SIZES),
                        help="Comma-separated list of n values")
    parser.add_argument("--graph_seeds", type=str,
                        default=",".join(str(s) for s in DEFAULT_GRAPH_SEEDS))
    parser.add_argument("--random_seeds", type=str,
                        default=",".join(str(s) for s in DEFAULT_RANDOM_SEEDS))
    parser.add_argument("--sample_budgets", type=str,
                        default=",".join(str(s) for s in DEFAULT_SAMPLE_BUDGETS))
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=0,
                        help="0 = auto (let the solver pick based on n)")
    parser.add_argument("--inst_dir", type=str, default="large_instances_sparse")
    parser.add_argument("--out_dir", type=str, default="cpu_sparse_results")
    parser.add_argument("--baseline_dir", type=str, default="fixed_results")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would run without executing")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    graph_seeds = [int(s) for s in args.graph_seeds.split(",") if s.strip()]
    random_seeds = [int(s) for s in args.random_seeds.split(",") if s.strip()]
    sample_budgets = [int(s) for s in args.sample_budgets.split(",") if s.strip()]

    families = DEFAULT_FAMILIES if args.family == "all" else [args.family]

    os.makedirs(args.out_dir, exist_ok=True)

    plan = []
    for family in families:
        for n in sizes:
            for gseed in graph_seeds:
                for budget in sample_budgets:
                    for rseed in random_seeds:
                        tag = (f"{family}_{n}_seed{gseed}"
                               f"_samples{budget}_rseed{rseed}")
                        out_path = os.path.join(args.out_dir, f"{tag}.json")
                        if os.path.exists(out_path):
                            continue
                        plan.append((family, n, gseed, budget, rseed, out_path))

    print(f"Plan: {len(plan)} runs to execute")
    for family, n, gseed, budget, rseed, out_path in plan[:10]:
        print(f"  {family} n={n} gseed={gseed} budget={budget:,} rseed={rseed}")
    if len(plan) > 10:
        print(f"  ... and {len(plan) - 10} more")

    if args.dry_run:
        print("Dry run; exiting.")
        return

    t_start = time.time()
    successes = 0
    failures = 0

    # Group by instance so we only generate each one once
    instance_cache = {}

    for family, n, gseed, budget, rseed, out_path in plan:
        inst_key = (family, n, gseed)
        if inst_key not in instance_cache:
            l_path, v_path, actual_n = ensure_instance(
                family, n, gseed, args.inst_dir
            )
            instance_cache[inst_key] = (l_path, v_path, actual_n)
        else:
            l_path, v_path, actual_n = instance_cache[inst_key]

        print(f"\n{'='*70}")
        print(
            f"  Running {family} n={actual_n:,} gseed={gseed} "
            f"budget={budget:,} rseed={rseed}"
        )
        print(f"{'='*70}", flush=True)

        t_run = time.time()
        ok = run_solver(
            l_path=l_path,
            v_path=v_path,
            max_samples=budget,
            rseed=rseed,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            out_path=out_path,
        )
        elapsed = time.time() - t_run

        if ok and os.path.exists(out_path):
            successes += 1
            # Annotate with baseline comparison
            with open(out_path) as f:
                result = json.load(f)
            baseline = load_baseline(
                family, actual_n, gseed, args.baseline_dir
            )
            if baseline is None:
                # Try also the parameter n (sometimes fixed_results uses
                # requested rather than actual n; for torus they may differ)
                baseline = load_baseline(
                    family, n, gseed, args.baseline_dir
                )
            if baseline is not None:
                result["baseline"] = {
                    k: baseline.get(k, {}).get("score")
                    for k in ("rank1", "greedy", "hybrid", "sa")
                }
                result["baseline"]["n"] = baseline.get("n")
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)
                print(
                    f"  score={result['best_score']:.0f}  "
                    f"vs R1={result['baseline'].get('rank1')}  "
                    f"H={result['baseline'].get('hybrid')}  "
                    f"G={result['baseline'].get('greedy')}  "
                    f"SA={result['baseline'].get('sa')}  "
                    f"({elapsed:.0f}s)",
                    flush=True,
                )
            else:
                print(
                    f"  score={result['best_score']:.0f} "
                    f"[no baseline found] ({elapsed:.0f}s)",
                    flush=True,
                )
        else:
            failures += 1
            print(f"  FAILED after {elapsed:.0f}s", flush=True)

    total_elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(
        f"DONE: {successes} succeeded, {failures} failed "
        f"in {total_elapsed/3600:.2f} hours"
    )
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
