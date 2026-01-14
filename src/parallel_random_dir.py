#!/usr/bin/env python3
"""
parallel_random_dir.py

Random-cut baseline over a directory of (Q,V) instances, within ONE Ray session.

Key behavior:
- For each instance with n vertices, rank-1 enumeration considers (n + 1) candidates (prefix lengths).
- Here, num_candidates defaults to (n + 1) if not explicitly specified.
- You can override with --num_candidates X (or --random_candidates X for backward naming).

Outputs JSON in the same schema style as parallel_rank_r_dir.py.
"""

import argparse
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import ray
import random

from utils import set_numpy_precision, opt_K_cut

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def discover_instances(qv_dir: Path) -> List[Tuple[Path, Path]]:
    """
    For every file starting with 'Q' and ending with '.npy',
    pair it with the corresponding 'V' file obtained by
    replacing the leading 'Q' with 'V'.
    """
    q_files = sorted(qv_dir.glob("Q*.npy"))
    out: List[Tuple[Path, Path]] = []

    for q_path in q_files:
        v_name = "V" + q_path.name[1:]
        v_path = q_path.parent / v_name
        if not v_path.exists():
            log.warning(f"Missing V for Q={q_path.name}, expected {v_name}. Skipping.")
            continue
        out.append((q_path, v_path))

    return out


def result_already_exists(results_dir: Path, q_path: Path, x_used: int) -> bool:
    stem = q_path.stem
    out_path = results_dir / f"{stem}_rand{int(x_used)}.json"
    return out_path.exists()


@ray.remote
def process_random_batch(
    num_samples: int,
    n: int,
    Q: np.ndarray,
    roots: np.ndarray,
    K: int,
    seed: int,
    batch_id: int,
):
    """
    One Ray task: evaluate `num_samples` random assignments.
    Returns best (score, k, z) within this task.
    """
    rng = np.random.default_rng(seed + batch_id)

    best_score = float("-inf")
    best_k = None
    best_z = None

    for _ in range(int(num_samples)):
        k = rng.integers(0, K, size=n, endpoint=False)
        z = roots[k]
        score = np.einsum("i,ij,j->", z.conj(), Q, z).real

        if score > best_score:
            best_score = float(score)
            best_k = k
            best_z = z

    return best_score, best_k, best_z, batch_id


def process_random_parallel(
    Q: np.ndarray,
    K: int,
    num_candidates: int,
    candidates_per_task: int,
    seed: int,
    complex_dtype,
):
    """
    Ray-parallel random baseline: sample `num_candidates` random K-cuts and return the best.
    Returns (best_score, best_k, best_z).
    """
    n = int(Q.shape[0])

    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive after defaulting")
    if candidates_per_task <= 0:
        raise ValueError("--candidates_per_task must be positive")
    if K <= 1:
        raise ValueError("--K must be >= 2")

    roots = np.exp(2 * np.pi * 1j * np.arange(K) / K).astype(complex_dtype, copy=False)

    resources = ray.available_resources()
    num_cpus = max(1, int(resources.get("CPU", 1)))
    max_in_flight = max(2 * num_cpus, 1)

    Q_ref = ray.put(Q)
    roots_ref = ray.put(roots)

    total_tasks = (num_candidates + candidates_per_task - 1) // candidates_per_task
    log.info(
        f"Random baseline: n={n}, K={K}, X={num_candidates}, "
        f"candidates_per_task={candidates_per_task}, total_tasks={total_tasks}, Ray CPUs={num_cpus}"
    )

    start_time = time.time()

    in_flight = []
    submitted = 0
    completed = 0

    best_score = float("-inf")
    best_k = None
    best_z = None

    def submit_one(batch_size: int, batch_id: int):
        return process_random_batch.remote(
            batch_size, n, Q_ref, roots_ref, K, seed, batch_id
        )

    batch_id = 0
    remaining = int(num_candidates)

    while remaining > 0:
        batch_size = min(int(candidates_per_task), remaining)
        in_flight.append(submit_one(batch_size, batch_id))
        submitted += 1
        batch_id += 1
        remaining -= batch_size

        if len(in_flight) >= max_in_flight:
            done, in_flight = ray.wait(in_flight, num_returns=1)
            batch_score, batch_k, batch_z, b_id = ray.get(done[0])
            completed += 1

            if batch_k is not None and batch_score > best_score:
                best_score = float(batch_score)
                best_k = batch_k
                best_z = batch_z
                log.info(f"New best random score from batch {b_id}: {best_score}")

    while in_flight:
        done, in_flight = ray.wait(in_flight, num_returns=1)
        batch_score, batch_k, batch_z, b_id = ray.get(done[0])
        completed += 1

        if batch_k is not None and batch_score > best_score:
            best_score = float(batch_score)
            best_k = batch_k
            best_z = batch_z
            log.info(f"New best random score from batch {b_id}: {best_score}")

    elapsed = time.time() - start_time
    log.info(
        f"Random search complete in {elapsed:.4f}s; submitted={submitted}, completed={completed}, max_in_flight={max_in_flight}"
    )

    if best_k is None or best_z is None:
        raise RuntimeError("Random baseline found no candidate")

    return best_score, best_k, best_z


def parse_args():
    ap = argparse.ArgumentParser(description="Run random-cut baseline over a directory without restarting Ray.")
    ap.add_argument("--qv_dir", type=str, required=True, help="Directory containing Q_* and V_* .npy files")
    ap.add_argument("--results_dir", type=str, required=True, help="Directory to store outputs (json)")

    ap.add_argument(
        "--num_candidates",
        "--random_candidates",
        dest="num_candidates",
        type=int,
        default=0,
        help="X: number of random cuts per instance. Default is (n + 1), matching rank-1 candidate count.",
    )

    ap.add_argument("--K", type=int, default=3, help="Number of partitions (default 3)")
    ap.add_argument("--precision", type=int, default=64, choices=[16, 32, 64], help="Numeric precision")
    ap.add_argument("--candidates_per_task", type=int, default=1000, help="Random cuts per Ray task")
    ap.add_argument("--seed", type=int, default=42, help="Base seed (deterministic across runs)")
    ap.add_argument("--debug", action="store_true", help="Compute opt_K_cut (only feasible for tiny n)")
    ap.add_argument("--max_instances", type=int, default=0, help="If >0, cap number of instances processed")
    ap.add_argument("--skip_existing", action="store_true", help="Skip if matching result json exists")
    ap.add_argument("--start_index", type=int, default=0, help="Start from this index in sorted instance list")
    return ap.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    qv_dir = Path(args.qv_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    float_dtype, complex_dtype = set_numpy_precision(args.precision)
    log.info(f"precision={args.precision} -> float={float_dtype.__name__}, complex={complex_dtype.__name__}")

    ray.init(address="auto", ignore_reinit_error=True)
    resources = ray.available_resources()
    num_workers = int(resources.get("CPU", 1))
    log.info(f"Ray connected. Detected CPU slots: {num_workers}")

    instances = discover_instances(qv_dir)
    if not instances:
        raise SystemExit(f"No instances found in {qv_dir} matching Q*.npy")

    if args.start_index < 0 or args.start_index >= len(instances):
        raise SystemExit(f"--start_index out of range: {args.start_index} (0..{len(instances)-1})")

    instances = instances[args.start_index:]
    if args.max_instances and args.max_instances > 0:
        instances = instances[: args.max_instances]

    log.info(f"Discovered {len(instances)} instances to run (after slicing) from {qv_dir}")

    for idx, (q_path, v_path) in enumerate(instances):
        log.info("============================================================")
        log.info(f"[{idx+1}/{len(instances)}] Q: {q_path}")
        log.info(f"V: {v_path} (recorded only, not used)")

        Q = np.asarray(np.load(q_path), dtype=float_dtype)
        n = int(Q.shape[0])

        # Default X matches rank-1 enumeration count: n + 1.
        x_used = int(args.num_candidates) if int(args.num_candidates) > 0 else int(n + 1)

        if args.skip_existing and result_already_exists(results_dir, q_path, x_used):
            log.info(f"Skip: result file already exists for X={x_used}.")
            del Q
            continue

        t0 = time.time()
        best_score, best_k, best_z = process_random_parallel(
            Q,
            K=int(args.K),
            num_candidates=x_used,
            candidates_per_task=int(args.candidates_per_task),
            seed=int(args.seed),  # uniform seed for all instances
            complex_dtype=complex_dtype,
        )
        elapsed = time.time() - t0

        log.info(f"Done: score={best_score}, time={elapsed:.4f}s, X={x_used}")

        output: Dict[str, object] = {
            "rank": 0,
            "precision": int(args.precision),
            "candidates_per_task": int(args.candidates_per_task),
            "num_candidates": int(x_used),
            "K": int(args.K),
            "best_score": float(best_score),
            "time_seconds": float(elapsed),
            "best_k": np.asarray(best_k).tolist(),
            "best_z_real": np.real(best_z).tolist(),
            "best_z_imag": np.imag(best_z).tolist(),
            "num_workers": int(num_workers),
            "q_file": str(q_path),
            "v_file": str(v_path),
            "seed": int(args.seed),
        }

        if args.debug:
            opt_score, _ = opt_K_cut(Q.astype(np.float64, copy=False))
            output["opt_score"] = float(opt_score)
            log.info(f"opt_score={opt_score}")

        stem = q_path.stem
        fname = f"{stem}_rand{int(x_used)}.json"
        out_path = results_dir / fname
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        log.info(f"Saved: {out_path}")

        del Q, best_z

    log.info("All instances complete.")
    ray.shutdown()


if __name__ == "__main__":
    main()
