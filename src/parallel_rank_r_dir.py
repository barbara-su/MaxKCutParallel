#!/usr/bin/env python3
"""
parallel_rank_r_dir.py

Run MANY (Q,V) instances from a directory inside ONE Ray cluster session,
so you do NOT restart Ray per instance.

Expected filenames (flexible, as long as these patterns match):
  Q_<n>_seed_<seed>*.npy
  V_<n>_seed_<seed>*.npy

Example:
  Q_20_seed_7.npy
  V_20_seed_7.npy
  Q_20000_seed_42_low0_high0.npy
  V_20000_seed_42_low0_high0.npy

Usage (inside your symmetric_run cluster):
  python -u src/parallel_rank_r_dir.py \
    --qv_dir graphs/erdos_renyi/rank_2/p01/n20 \
    --results_dir results/erdos_renyi/rank_2/p01/n20 \
    --rank 2 --precision 32 --candidates_per_task 1000

Notes:
- This script calls ray.init(address="auto") ONCE.
- It loops through instances and runs rank-1 or recursive rank-r solver per instance.
- It loads Q/V directly from the seeded filenames; no symlink hacks needed.
"""

import argparse
import itertools
import json
import logging
import math
import os
import re
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import ray
import random

from utils import (
    set_numpy_precision,
    compute_vtilde,
    get_row_mapping,
    find_intersection,
    determine_phi_sign_c,
    find_intersection_fixed_angle,
    convert_ctilde_to_complex,
    complex_to_partition,
    opt_K_cut,
    generate_debug_QV,
)
from parallel_rank_1 import process_rank_1_parallel  # returns (best_score, best_k, best_z, best_l)

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------
# Rank-r Ray remote worker
# ---------------------------
@ray.remote
def process_combination_batch(
    V_tilde,
    V,
    K,
    r,
    Q,
    combinations_batch,
    batch_id,
    row_mapping,
    inverse_mapping,
    s,
):
    n = V.shape[0]
    best_score = float("-inf")
    best_candidate = None

    for combo in combinations_batch:
        try:
            I = np.array(combo, dtype=int)
            VI = V_tilde[I]  # (2r-1, 2r)

            c_tilde = find_intersection(VI)
            phi, sign_c = determine_phi_sign_c(c_tilde)

            if not (-np.pi / K < phi[2 * r - 2] <= np.pi / K):
                continue

            candidate = np.zeros(n, dtype=V.dtype)

            c_tilde = c_tilde * sign_c
            c = convert_ctilde_to_complex(c_tilde, r)

            v_indices_used = set()
            for idx in I:
                v_row, _ = row_mapping[idx]
                v_indices_used.add(v_row)

            # vertices not in selected hyperplanes
            for k in range(n):
                if k in v_indices_used:
                    continue
                v_c = V[k] @ c
                metric = np.real(np.conj(s) * v_c)
                s_idx = int(np.argmax(metric))
                candidate[k] = s[s_idx]

            # vertices in selected hyperplanes
            for v_idx in v_indices_used:
                vtilde_rows_for_v = [idx for idx in inverse_mapping[v_idx] if idx in I]

                assigned = False
                for vtilde_idx in vtilde_rows_for_v:
                    pos = int(np.where(I == vtilde_idx)[0][0])
                    VI_minus = np.delete(VI, pos, axis=0)
                    try:
                        new_c_tilde = find_intersection_fixed_angle(VI_minus, r, K)
                        new_c = convert_ctilde_to_complex(new_c_tilde, r)
                        v_c = V[v_idx] @ new_c
                        metric = np.real(np.conj(s) * v_c)
                        s_idx = int(np.argmax(metric))
                        candidate[v_idx] = s[s_idx]
                        assigned = True
                        break
                    except ValueError:
                        continue

                if (not assigned) or (np.abs(candidate[v_idx]) < 1e-10):
                    v_c = V[v_idx] @ c
                    metric = np.real(np.conj(s) * v_c)
                    s_idx = int(np.argmax(metric))
                    candidate[v_idx] = s[s_idx]

            score = np.einsum("i,ij,j->", candidate.conj(), Q, candidate).real

            if score > best_score:
                best_score = float(score)
                best_candidate = candidate.copy()

        except (ValueError, np.linalg.LinAlgError):
            continue

    return best_score, best_candidate, batch_id


def process_rankr_single(V: np.ndarray, Q: np.ndarray, K: int = 3, candidates_per_task: int = 1000):
    n, r = V.shape
    log.info(f"Rank r subroutine (single rank): n={n}, r={r}, K={K}")

    if candidates_per_task <= 0:
        raise ValueError("--candidates_per_task must be positive")

    log.info("Computing V_tilde")
    V_tilde = compute_vtilde(V)

    log.info("Computing row mappings for V_tilde")
    row_mapping, inverse_mapping = get_row_mapping(n, K)

    s = np.exp(1j * 2 * np.pi * np.arange(K) / K).astype(V.dtype, copy=False)

    num_vtilde_rows = K * n
    comb_size = 2 * r - 1
    if comb_size > num_vtilde_rows:
        raise ValueError("Combination size 2r-1 exceeds K*n")

    num_combinations = math.comb(num_vtilde_rows, comb_size)
    log.info(f"Total (2r-1)-tuples: C({num_vtilde_rows},{comb_size}) = {num_combinations}")

    resources = ray.available_resources()
    num_cpus = max(1, int(resources.get("CPU", 1)))
    total_tasks = (num_combinations + candidates_per_task - 1) // candidates_per_task if num_combinations > 0 else 0
    log.info(f"Ray CPUs={num_cpus}, candidates_per_task={candidates_per_task}, total_tasks={total_tasks}")

    V_tilde_ref = ray.put(V_tilde)
    V_ref = ray.put(V)
    Q_ref = ray.put(Q)
    row_mapping_ref = ray.put(row_mapping)
    inverse_mapping_ref = ray.put(inverse_mapping)
    s_ref = ray.put(s)

    def batched_combinations():
        iterator = itertools.combinations(range(num_vtilde_rows), comb_size)
        while True:
            batch = list(itertools.islice(iterator, candidates_per_task))
            if not batch:
                break
            yield batch

    start_time = time.time()

    max_in_flight = max(2 * num_cpus, 1)
    in_flight = []
    submitted = 0
    completed = 0

    best_score = float("-inf")
    best_candidate = None

    def submit_one(batch, batch_id):
        return process_combination_batch.remote(
            V_tilde_ref,
            V_ref,
            K,
            r,
            Q_ref,
            batch,
            batch_id,
            row_mapping_ref,
            inverse_mapping_ref,
            s_ref,
        )

    batch_id = 0
    for comb_batch in batched_combinations():
        in_flight.append(submit_one(comb_batch, batch_id))
        batch_id += 1
        submitted += 1

        if len(in_flight) >= max_in_flight:
            done, in_flight = ray.wait(in_flight, num_returns=1)
            batch_score, batch_candidate, b_id = ray.get(done[0])
            completed += 1

            if batch_candidate is not None and batch_score > best_score:
                best_score = float(batch_score)
                best_candidate = batch_candidate
                log.info(f"New best score from batch {b_id}: {best_score}")

            if completed % 1000 == 0:
                log.info(f"Progress: submitted={submitted}, completed={completed}, in_flight={len(in_flight)}")

    while in_flight:
        done, in_flight = ray.wait(in_flight, num_returns=1)
        batch_score, batch_candidate, b_id = ray.get(done[0])
        completed += 1

        if batch_candidate is not None and batch_score > best_score:
            best_score = float(batch_score)
            best_candidate = batch_candidate
            log.info(f"New best score from batch {b_id}: {best_score}")

        if completed % 1000 == 0:
            log.info(f"Progress: submitted={submitted}, completed={completed}, in_flight={len(in_flight)}")

    elapsed = time.time() - start_time
    log.info(f"Rank r single-rank search complete in {elapsed:.4f}s; submitted={submitted}, completed={completed}")

    if best_candidate is None:
        raise RuntimeError("Rank r algorithm found no feasible candidate")

    best_k = complex_to_partition(best_candidate, K)
    best_z = best_candidate
    return best_score, best_k, best_z


def process_rankr_recursive(V: np.ndarray, Q: np.ndarray, K: int = 3, candidates_per_task: int = 1000):
    n, r = V.shape
    log.info(f"Recursive rank solver at r={r}")

    if r == 1:
        log.info("Base case r=1: process_rank_1_parallel")
        best_score, best_k, best_z, _ = process_rank_1_parallel(
            V[:, 0], Q, K, candidates_per_task=candidates_per_task
        )
        return best_score, best_k, best_z

    best_score, best_k, best_z = process_rankr_single(V, Q, K, candidates_per_task=candidates_per_task)

    log.info(f"Recursing to lower rank r={r-1}")
    lower_score, lower_k, lower_z = process_rankr_recursive(
        V[:, : r - 1], Q, K, candidates_per_task=candidates_per_task
    )

    if lower_score > best_score:
        log.info(f"Lower rank {r-1} improved score {best_score} -> {lower_score}")
        best_score, best_k, best_z = lower_score, lower_k, lower_z

    return best_score, best_k, best_z


def discover_instances(qv_dir: Path):
    """
    For every file starting with 'Q' and ending with '.npy',
    pair it with the corresponding 'V' file obtained by
    replacing the leading 'Q' with 'V'.

    Returns list of (q_path, v_path), sorted by filename.
    """
    q_files = sorted(qv_dir.glob("Q*.npy"))
    out = []

    for q_path in q_files:
        v_name = "V" + q_path.name[1:]
        v_path = q_path.parent / v_name

        if not v_path.exists():
            log.warning(f"Missing V for Q={q_path.name}, expected {v_name}. Skipping.")
            continue

        out.append((q_path, v_path))

    return out



def parse_args():
    ap = argparse.ArgumentParser(description="Run parallel_rank_r over a directory without restarting Ray.")
    ap.add_argument("--qv_dir", type=str, required=True, help="Directory containing Q_*_seed_*.npy and V_*_seed_*.npy")
    ap.add_argument("--results_dir", type=str, required=True, help="Directory to store outputs (json)")
    ap.add_argument("--rank", type=int, default=2, help="Rank r (1 uses rank-1 routine)")
    ap.add_argument("--precision", type=int, default=64, choices=[16, 32, 64], help="Numeric precision")
    ap.add_argument("--candidates_per_task", type=int, default=1000, help="Ray batch size per task")
    ap.add_argument("--debug", action="store_true", help="Compute opt_K_cut (only feasible for tiny n)")
    ap.add_argument("--max_instances", type=int, default=0, help="If >0, cap number of instances processed")
    ap.add_argument("--skip_existing", action="store_true", help="Skip if matching result json exists")
    ap.add_argument("--start_index", type=int, default=0, help="Start from this index in sorted instance list")
    return ap.parse_args()


def main():
    args = parse_args()
    
    np.random.seed(42)
    random.seed(42)

    qv_dir = Path(args.qv_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    float_dtype, complex_dtype = set_numpy_precision(args.precision)
    log.info(f"precision={args.precision} -> float={float_dtype.__name__}, complex={complex_dtype.__name__}")

    # Connect to the already-running Ray cluster ONCE (symmetric_run started it)
    ray.init(address="auto", ignore_reinit_error=True)
    resources = ray.available_resources()
    num_workers = int(resources.get("CPU", 1))
    log.info(f"Ray connected. Detected CPU slots: {num_workers}")

    instances = discover_instances(qv_dir)
    if not instances:
        raise SystemExit(f"No instances found in {qv_dir} matching Q_*_seed_*.npy")

    if args.start_index < 0 or args.start_index >= len(instances):
        raise SystemExit(f"--start_index out of range: {args.start_index} (0..{len(instances)-1})")

    instances = instances[args.start_index :]
    if args.max_instances and args.max_instances > 0:
        instances = instances[: args.max_instances]

    log.info(f"Discovered {len(instances)} instances to run (after slicing) from {qv_dir}")

    for idx, (q_path, v_path) in enumerate(instances):
        log.info("============================================================")
        log.info(f"[{idx+1}/{len(instances)}] rank={args.rank}")
        log.info(f"Q: {q_path}")
        log.info(f"V: {v_path}")

        # if args.skip_existing and result_already_exists(results_dir, n, seed, args.rank, args.precision, args.candidates_per_task):
        #     log.info("Skip: existing result json detected.")
        #     continue

        # Load
        Q = np.asarray(np.load(q_path), dtype=float_dtype)
        V_full = np.asarray(np.load(v_path), dtype=complex_dtype)

        if V_full.ndim == 1:
            V_full = V_full.reshape(-1, 1)

        if V_full.shape[1] < args.rank:
            raise ValueError(f"V has {V_full.shape[1]} cols but rank={args.rank}")

        V = V_full[:, : args.rank]

        # Solve
        t0 = time.time()
        if args.rank == 1:
            best_score, best_k, best_z, best_l = process_rank_1_parallel(
                V[:, 0], Q, K=3, candidates_per_task=args.candidates_per_task
            )
        else:
            best_score, best_k, best_z = process_rankr_recursive(
                V, Q, K=3, candidates_per_task=args.candidates_per_task
            )
            best_l = None
        elapsed = time.time() - t0

        log.info(f"Done: score={best_score}, time={elapsed:.4f}s")

        output: Dict[str, object] = {
            "rank": args.rank,
            "precision": args.precision,
            "candidates_per_task": args.candidates_per_task,
            "best_score": float(best_score),
            "time_seconds": float(elapsed),
            "best_k": np.asarray(best_k).tolist(),
            "best_z_real": np.real(best_z).tolist(),
            "best_z_imag": np.imag(best_z).tolist(),
            "num_workers": int(num_workers),
            "q_file": str(q_path),
            "v_file": str(v_path),
        }
        if best_l is not None:
            output["best_l"] = int(best_l)

        if args.debug:
            opt_score, _ = opt_K_cut(Q.astype(np.float64, copy=False))
            output["opt_score"] = float(opt_score)
            log.info(f"opt_score={opt_score}")

        stem = q_path.stem 
        fname = f"{stem}_r{args.rank}.json"
        out_path = results_dir / fname
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        log.info(f"Saved: {out_path}")

        # Help GC a bit between instances
        del Q, V_full, V, best_z

    log.info("All instances complete.")
    ray.shutdown()


if __name__ == "__main__":
    main()
