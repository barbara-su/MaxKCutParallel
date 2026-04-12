"""
Phase 2: Sample count sweep for randomized rank-2 on GPU.

Sweeps sample counts (1K to 100M) across graph families and sizes.
Produces data for the "diminishing returns" hero figure.

Usage:
    python experiments/run_randomized_sweep_gpu.py [--family regular] [--num_gpus 4]
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

SAMPLE_COUNTS = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
RANDOM_SEEDS = [0, 1, 2]

INSTANCES = {
    "regular": [
        ("regular", 250, [0, 1, 2]),
        ("regular", 500, [0, 1, 2]),
        ("regular", 1000, [0, 1, 2]),
        ("regular", 1500, [0, 1, 2]),
    ],
    "torus": [
        ("torus", 252, [0, 1, 2]),
        ("torus", 504, [0, 1, 2]),
        ("torus", 1008, [0, 1, 2]),
    ],
    "sbm": [
        ("sbm", 250, [0, 1, 2]),
        ("sbm", 500, [0, 1, 2]),
    ],
}


def find_instance_paths(base_dir, family, n, seed):
    """Find Q and V paths for a given instance."""
    patterns = [
        (f"{base_dir}/{family}/Q_{family}_{n}_seed_{seed}.npy",
         f"{base_dir}/{family}/V_{family}_{n}_seed_{seed}.npy"),
    ]
    for q_path, v_path in patterns:
        if os.path.exists(q_path) and os.path.exists(v_path):
            return q_path, v_path
    return None, None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=str, default="all",
                        help="Graph family: regular, torus, sbm, or all")
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--base_dir", type=str, default="blog_instances")
    parser.add_argument("--out_dir", type=str, default="randomized_gpu_results")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--chunk_size", type=int, default=50000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    families = list(INSTANCES.keys()) if args.family == "all" else [args.family]

    total_runs = 0
    skipped = 0
    errors = 0
    t_start = time.time()

    for family in families:
        if family not in INSTANCES:
            print(f"Unknown family: {family}")
            continue

        for fam, n, graph_seeds in INSTANCES[family]:
            for gseed in graph_seeds:
                q_path, v_path = find_instance_paths(args.base_dir, fam, n, gseed)
                if q_path is None:
                    print(f"SKIP: {fam} n={n} seed={gseed} — instance not found")
                    continue

                Kn = args.K * n
                comb_size = 2 * args.rank - 1
                total_comb = math.comb(Kn, comb_size)

                for max_samples in SAMPLE_COUNTS:
                    # Skip if max_samples > total candidates
                    if max_samples > total_comb:
                        continue

                    for rseed in RANDOM_SEEDS:
                        tag = f"{fam}_{n}_gseed{gseed}_samples{max_samples}_rseed{rseed}"
                        out_path = os.path.join(args.out_dir, f"{tag}.json")

                        if os.path.exists(out_path):
                            skipped += 1
                            continue

                        print(f"\n{'='*60}")
                        print(f"  {tag}")
                        print(f"  {max_samples:,} samples of {total_comb:,} ({max_samples/total_comb*100:.4f}%)")
                        print(f"{'='*60}")

                        cmd = (
                            f"python3 -u src/randomized_rank_r_gpu.py "
                            f"--q_path {q_path} --v_path {v_path} "
                            f"--rank {args.rank} --K {args.K} "
                            f"--max_samples {max_samples} --seed {rseed} "
                            f"--num_gpus {args.num_gpus} --chunk_size {args.chunk_size} "
                            f"--out {out_path}"
                        )

                        try:
                            ret = os.system(cmd)
                            if ret != 0:
                                print(f"  ERROR: exit code {ret}")
                                errors += 1
                            else:
                                total_runs += 1
                        except Exception as e:
                            print(f"  ERROR: {e}")
                            errors += 1

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"DONE: {total_runs} runs, {skipped} skipped, {errors} errors")
    print(f"Total time: {elapsed/3600:.1f} hours")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
