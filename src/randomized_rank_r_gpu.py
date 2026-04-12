"""
GPU-accelerated randomized rank-r solver for Max-K-Cut.

Instead of enumerating ALL O(n^{2r-1}) candidates, randomly samples a subset
on GPU. Uses the same GPUKernel from worker.py for scoring, but replaces
combination unranking with torch.randint for index generation.

Expected throughput: ~300K candidates/sec per P100 (1000x faster than CPU).

Usage:
    python randomized_rank_r_gpu.py --q_path Q.npy --v_path V.npy \
        --rank 2 --K 3 --max_samples 10000000 --num_gpus 4 --seed 42 \
        --out result.json
"""
import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("TMPDIR", os.environ.get("TMPDIR", "/tmp"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.multiprocessing as mp

from worker import GPUKernel
from utils import compute_vtilde


def generate_random_indices(Kn, comb_size, batch_size, seed, device):
    """Generate a batch of random sorted (2r-1)-tuples from [0, Kn).

    Uses rejection sampling to avoid duplicate indices within each tuple.
    For large Kn (e.g., 3000+), duplicates are extremely rare (<0.1%).
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # Generate random indices
    I = torch.randint(0, Kn, (batch_size, comb_size), device=device, generator=gen)

    # Sort each row
    I, _ = torch.sort(I, dim=1)

    # Reject rows with duplicate indices (where sorted[i] == sorted[i+1])
    if comb_size > 1:
        dups = (I[:, 1:] == I[:, :-1]).any(dim=1)
        # For large Kn, dup rate is tiny. Retry only the bad rows.
        dup_count = dups.sum().item()
        retries = 0
        while dup_count > 0 and retries < 5:
            new_I = torch.randint(0, Kn, (dup_count, comb_size), device=device, generator=gen)
            new_I, _ = torch.sort(new_I, dim=1)
            I[dups] = new_I
            dups = (I[:, 1:] == I[:, :-1]).any(dim=1)
            dup_count = dups.sum().item()
            retries += 1
        # Drop any remaining duplicates
        if dup_count > 0:
            I = I[~dups]

    return I


def gpu_random_worker(
    gpu_id, V_np, Q_np, V_tilde_np, max_samples, r, K,
    chunk_size, result_dict, precision=32, seed=42
):
    """Run randomized sampling on a single GPU."""
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    kernel = GPUKernel(device, K, precision)
    kernel.set_instance(V_np, Q_np, V_tilde_np)

    Kn = K * V_np.shape[0]
    comb_size = 2 * r - 1

    best_score = float("-inf")
    best_k = None
    best_z = None
    total_feasible = 0
    total_processed = 0

    t0 = time.time()
    batch_id = 0
    # Use different sub-seeds per GPU to avoid overlap
    rng_seed = seed * 1000 + gpu_id

    while total_processed < max_samples:
        cur = min(chunk_size, max_samples - total_processed)

        # Generate random indices on GPU, then move to CPU for score_batch
        I_gpu = generate_random_indices(Kn, comb_size, cur, rng_seed + batch_id, device)
        I_np = I_gpu.cpu().numpy()

        if I_np.shape[0] == 0:
            total_processed += cur
            batch_id += 1
            continue

        score, k, z, feas = kernel.score_batch(I_np, r)
        total_feasible += feas
        total_processed += I_np.shape[0]

        if k is not None and score > best_score:
            best_score = score
            best_k = k.copy()
            best_z = z.copy()

        batch_id += 1

        if batch_id % 20 == 0:
            elapsed = time.time() - t0
            rate = total_processed / elapsed if elapsed > 0 else 0
            pct = total_processed / max_samples * 100
            print(
                f"  GPU {gpu_id}: {pct:.1f}% ({total_processed:,}/{max_samples:,}), "
                f"score={best_score:.0f}, rate={rate:,.0f}/s, "
                f"feasible={total_feasible}/{total_processed}",
                flush=True,
            )

    elapsed = time.time() - t0
    result_dict[gpu_id] = {
        "best_score": best_score,
        "best_k": best_k,
        "best_z": best_z,
        "feasible": total_feasible,
        "processed": total_processed,
        "elapsed": elapsed,
    }
    print(
        f"  GPU {gpu_id}: DONE in {elapsed:.1f}s, "
        f"score={best_score:.0f}, "
        f"rate={total_processed/elapsed:,.0f}/s, "
        f"processed={total_processed:,}, feasible={total_feasible:,}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="GPU randomized rank-r Max-K-Cut solver")
    parser.add_argument("--q_path", type=str, required=True)
    parser.add_argument("--v_path", type=str, required=True)
    parser.add_argument("--vtilde_path", type=str, default="")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--max_samples", type=int, required=True,
                        help="Total number of random candidates to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_gpus", type=int, default=-1, help="-1 = all visible")
    parser.add_argument("--chunk_size", type=int, default=50000)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    # Load data
    Q = np.load(args.q_path).astype(np.float64 if args.precision == 64 else np.float32)
    V = np.load(args.v_path)
    if V.ndim == 1:
        V = V.reshape(-1, 1)
    V = V[:, :args.rank]
    if not np.iscomplexobj(V):
        V = V.astype(np.complex128 if args.precision == 64 else np.complex64)

    n = Q.shape[0]
    r = args.rank
    K = args.K

    # V_tilde
    if args.vtilde_path and os.path.exists(args.vtilde_path):
        V_tilde = np.load(args.vtilde_path).astype(np.float64 if args.precision == 64 else np.float32)
    else:
        V_tilde = compute_vtilde(V).astype(np.float64 if args.precision == 64 else np.float32)

    Kn = K * n
    comb_size = 2 * r - 1
    total_comb = math.comb(Kn, comb_size)
    fraction = args.max_samples / total_comb

    num_gpus = args.num_gpus
    if num_gpus <= 0:
        num_gpus = torch.cuda.device_count()
    num_gpus = min(num_gpus, torch.cuda.device_count())

    samples_per_gpu = args.max_samples // num_gpus
    remainder = args.max_samples % num_gpus

    print(f"Instance: n={n}, r={r}, K={K}")
    print(f"Total candidates: C({Kn},{comb_size}) = {total_comb:,}")
    print(f"Sampling: {args.max_samples:,} ({fraction*100:.4f}%)")
    print(f"GPUs: {num_gpus}, seed: {args.seed}")

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_dict = manager.dict()

    processes = []
    t_total = time.time()

    for g in range(num_gpus):
        gpu_samples = samples_per_gpu + (1 if g < remainder else 0)
        p = mp.Process(
            target=gpu_random_worker,
            args=(g, V, Q, V_tilde, gpu_samples, r, K,
                  args.chunk_size, result_dict, args.precision, args.seed),
        )
        p.start()
        processes.append(p)
        print(f"  Launched GPU {g}: {gpu_samples:,} samples")

    for p in processes:
        p.join()

    total_elapsed = time.time() - t_total

    # Merge results across GPUs
    best_score = float("-inf")
    best_k = None
    best_z = None
    total_feasible = 0
    total_processed = 0

    for g in range(num_gpus):
        if g not in result_dict:
            continue
        res = result_dict[g]
        total_feasible += res["feasible"]
        total_processed += res["processed"]
        if res["best_k"] is not None and res["best_score"] > best_score:
            best_score = res["best_score"]
            best_k = res["best_k"]
            best_z = res["best_z"]

    rate = total_processed / total_elapsed if total_elapsed > 0 else 0

    print(f"\nResult: score={best_score:.0f}")
    print(f"  Samples: {total_processed:,} / {total_comb:,} ({total_processed/total_comb*100:.4f}%)")
    print(f"  Feasible: {total_feasible:,} ({total_feasible/max(total_processed,1)*100:.1f}%)")
    print(f"  Time: {total_elapsed:.1f}s ({rate:,.0f} cand/s)")

    if args.out:
        out = {
            "n": n,
            "r": r,
            "K": K,
            "best_score": float(best_score) if best_score > float("-inf") else None,
            "max_samples": args.max_samples,
            "total_processed": total_processed,
            "total_candidates": total_comb,
            "sample_fraction": total_processed / total_comb,
            "feasible_count": total_feasible,
            "elapsed": round(total_elapsed, 2),
            "rate": round(rate, 1),
            "num_gpus": num_gpus,
            "seed": args.seed,
            "q_path": args.q_path,
        }
        if best_k is not None:
            out["best_k"] = best_k.tolist()
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
