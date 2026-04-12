"""
Phase 3C: CPU sparse randomized rank-2 on REAL-WORLD graphs.

Runs rand-R2 on delaunay_n10 through delaunay_n19, roadNet-PA, roadNet-TX
using pre-computed L and V from realworld_data/processed/.

Compares against rank-1/greedy/hybrid/SA from fixed_results/realworld_*.json.

This is the honest test of whether rank-2 helps where rank-1 fails:
rank-1 gets only 66-80% of greedy on these graphs.

Usage:
    python experiments/run_cpu_sparse_realworld.py \\
        [--num_workers 32] [--sample_budgets 1000000]
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


DEFAULT_GRAPHS = [
    "delaunay_n10",
    "delaunay_n13",
    "delaunay_n16",
    "delaunay_n19",
    "roadNet-PA",
    "roadNet-TX",
]
DEFAULT_SAMPLE_BUDGETS = [1_000_000]
DEFAULT_RANDOM_SEEDS = [0]


def run_solver(l_path, v_path, max_samples, rseed, num_workers, out_path,
               batch_size=0, solver_script=None):
    """Launch the CPU sparse solver via subprocess."""
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
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    ret = subprocess.call(cmd, env=env)
    return ret == 0


def load_baseline(graph_name, baseline_dir):
    """Load the matching baseline JSON from fixed_results/realworld_*.json."""
    path = os.path.join(baseline_dir, f"realworld_{graph_name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=str,
                        default=",".join(DEFAULT_GRAPHS),
                        help="Comma-separated graph names")
    parser.add_argument("--random_seeds", type=str,
                        default=",".join(str(s) for s in DEFAULT_RANDOM_SEEDS))
    parser.add_argument("--sample_budgets", type=str,
                        default=",".join(str(s) for s in DEFAULT_SAMPLE_BUDGETS))
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--data_dir", type=str,
                        default="realworld_data/processed")
    parser.add_argument("--out_dir", type=str, default="cpu_sparse_realworld")
    parser.add_argument("--baseline_dir", type=str, default="fixed_results")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    graphs = [g.strip() for g in args.graphs.split(",") if g.strip()]
    random_seeds = [int(s) for s in args.random_seeds.split(",") if s.strip()]
    sample_budgets = [int(s) for s in args.sample_budgets.split(",") if s.strip()]

    os.makedirs(args.out_dir, exist_ok=True)

    plan = []
    for gname in graphs:
        l_path = os.path.join(args.data_dir, f"L_{gname}.npz")
        v_path = os.path.join(args.data_dir, f"V_{gname}.npy")
        if not (os.path.exists(l_path) and os.path.exists(v_path)):
            print(f"SKIP: {gname} (missing files)")
            continue
        for budget in sample_budgets:
            for rseed in random_seeds:
                tag = f"{gname}_samples{budget}_rseed{rseed}"
                out_path = os.path.join(args.out_dir, f"{tag}.json")
                if os.path.exists(out_path):
                    continue
                plan.append((gname, l_path, v_path, budget, rseed, out_path))

    print(f"Plan: {len(plan)} runs to execute")
    for gname, _, _, budget, rseed, _ in plan[:20]:
        print(f"  {gname} budget={budget:,} rseed={rseed}")

    if args.dry_run:
        return

    t_start = time.time()
    successes = 0

    for gname, l_path, v_path, budget, rseed, out_path in plan:
        print(f"\n{'='*70}")
        print(f"  {gname} budget={budget:,} rseed={rseed}")
        print(f"{'='*70}", flush=True)
        t_run = time.time()
        ok = run_solver(
            l_path=l_path, v_path=v_path,
            max_samples=budget, rseed=rseed,
            num_workers=args.num_workers, batch_size=args.batch_size,
            out_path=out_path,
        )
        elapsed = time.time() - t_run
        if ok and os.path.exists(out_path):
            successes += 1
            with open(out_path) as f:
                result = json.load(f)
            baseline = load_baseline(gname, args.baseline_dir)
            if baseline is not None:
                result["baseline"] = {
                    "rank1": baseline.get("rank1", {}).get("score"),
                    "greedy": baseline.get("greedy", {}).get("score"),
                    "hybrid": baseline.get("hybrid", {}).get("score"),
                    "sa": baseline.get("sa", {}).get("score"),
                    "rank1_time": baseline.get("rank1", {}).get("time"),
                    "greedy_time": baseline.get("greedy", {}).get("time"),
                    "hybrid_time": baseline.get("hybrid", {}).get("total_time")
                                   or baseline.get("hybrid", {}).get("time"),
                    "sa_time": baseline.get("sa", {}).get("time"),
                }
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)
                print(
                    f"  score={result['best_score']:.0f}  "
                    f"R1={result['baseline']['rank1']}  "
                    f"G={result['baseline']['greedy']}  "
                    f"H={result['baseline']['hybrid']}  "
                    f"SA={result['baseline']['sa']}  "
                    f"({elapsed:.0f}s)",
                    flush=True,
                )
        else:
            print(f"  FAILED after {elapsed:.0f}s")

    total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"DONE: {successes}/{len(plan)} in {total/60:.1f} min")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
