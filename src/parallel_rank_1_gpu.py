#!/usr/bin/env python3
"""
parallel_rank_1_gpu.py

GPU-accelerated rank-1 solver with Ray CPU orchestration.

Same CLI shape as your original parallel_rank_1.py, plus:
- --K
- --gpus (optional cap; default uses all Ray-visible GPUs)

Core idea:
- CPU computes k0 and sorted_idx.
- CPU Ray tasks build batched integer assignments k_batch for a list of prefix lengths l.
- GPU actor(s) keep dense Q on device and score candidates in batch:
    score(z) = Re(conj(z)^T Q z),  z_i = exp(2πj k_i / K).

Design A (multi-GPU):
- Spawn one Rank1GPUActor per GPU.
- Round-robin batches across actors.
"""

import argparse
import itertools
import json
import logging
import os
import time
import warnings
from datetime import datetime
from typing import List

import numpy as np
import ray

from utils import (
    set_numpy_precision,
    generate_Q,
    low_rank_matrix,
    generate_debug_QV,
    opt_K_cut,
)

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _torch_dtype_names_from_precision(precision: int):
    if precision in (16, 32):
        return "float32", "complex64"
    if precision == 64:
        return "float64", "complex128"
    raise ValueError("precision must be one of {16,32,64}")


@ray.remote(num_gpus=1)
class Rank1GPUActor:
    """
    Owns GPU, keeps dense Q resident, scores batches of assignments k.

    Input:
      k_batch_np: (B,n) int64 entries in [0,K)
    Output:
      scores_np: (B,) real numpy
    """

    def __init__(self, K: int, precision: int):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        self.device = "cuda"
        self.K = int(K)
        self.precision = int(precision)

        rdtype_name, cdtype_name = _torch_dtype_names_from_precision(self.precision)
        self.rdtype = getattr(torch, rdtype_name)
        self.cdtype = getattr(torch, cdtype_name)

        kk = torch.arange(self.K, device=self.device, dtype=torch.float32)
        self.roots = torch.exp(2j * torch.pi * kk / self.K).to(self.cdtype)  # (K,)

        self.Q = None  # (n,n) real float
        self.n = None

    def set_instance(self, Q_np: np.ndarray):
        import torch

        Q_arr = np.asarray(Q_np)
        if (not Q_arr.flags.writeable) or (not Q_arr.flags.c_contiguous):
            Q_arr = np.array(Q_arr, copy=True, order="C")
        Q_t = torch.as_tensor(Q_arr)
        if Q_t.dtype == torch.float16:
            Q_t = Q_t.to(torch.float32)
        # Assume dense real Q (as in your code)
        self.Q = Q_t.to(device=self.device, dtype=self.rdtype).contiguous()
        self.n = int(self.Q.shape[0])

    def score_k_batch(self, k_batch_np: np.ndarray) -> np.ndarray:
        import torch

        with torch.inference_mode():
            if self.Q is None:
                raise RuntimeError("Call set_instance(Q) before score_k_batch")

            k_arr = np.asarray(k_batch_np)
            if (not k_arr.flags.writeable) or (not k_arr.flags.c_contiguous):
                k_arr = np.array(k_arr, copy=True, order="C")
            k = torch.as_tensor(k_arr, device=self.device, dtype=torch.int64)
            if k.ndim != 2:
                raise ValueError("k_batch must have shape (B,n)")

            # z: (B,n) complex
            z = self.roots[k]

            # Q @ z^T via two real GEMMs (Q is real)
            zT = z.T  # (n,B)
            Qzr = torch.matmul(self.Q, zT.real)
            Qzi = torch.matmul(self.Q, zT.imag)
            Qz = Qzr + 1j * Qzi  # (n,B) complex

            # score_b = sum_i conj(z_i) * (Qz)_i
            scores = torch.sum(torch.conj(zT) * Qz, dim=0).real  # (B,)

            return scores.to("cpu").numpy()


@ray.remote
def process_rank_1_batch_hybrid(
    l_values: List[int],
    k0: np.ndarray,
    sorted_idx: np.ndarray,
    K: int,
    batch_id: int,
    gpu_actor: "ray.actor.ActorHandle",
):
    """
    One CPU Ray task:
    - build k_batch for this list of l values
    - GPU score them
    - return best (score, l) within this task
    """
    B = int(len(l_values))
    if B == 0:
        return float("-inf"), None, int(batch_id)

    # Build (B,n)
    k_batch = np.tile(k0[None, :], (B, 1)).astype(np.int64, copy=False)

    for b, l in enumerate(l_values):
        l_int = int(l)
        if l_int > 0:
            idx = sorted_idx[:l_int]
            k_batch[b, idx] = (k_batch[b, idx] + 1) % int(K)

    scores = ray.get(gpu_actor.score_k_batch.remote(k_batch))  # (B,)
    best_pos = int(np.argmax(scores))
    best_score = float(scores[best_pos])
    best_l = int(l_values[best_pos])
    return best_score, best_l, int(batch_id)


def process_rank_1_parallel_gpu(
    V: np.ndarray,
    Q: np.ndarray,
    K: int = 3,
    candidates_per_task: int = 10,
    gpu_actors: List["ray.actor.ActorHandle"] = None,
):
    """
    GPU-accelerated version of process_rank_1_parallel.

    Returns:
      best_score, best_k, best_z, best_l
    """
    if V.ndim == 2 and V.shape[1] == 1:
        V = V[:, 0]
    n = int(V.shape[0])

    log.info(f"Rank 1 subroutine (hybrid GPU): received eigenvector of length {n}")

    if candidates_per_task <= 0:
        raise ValueError("--candidates_per_task must be a positive integer")
    if gpu_actors is None or len(gpu_actors) == 0:
        raise ValueError("gpu_actors must be a non-empty list")

    real_q1 = np.real(V).flatten()
    im_q1 = np.imag(V).flatten()

    thetas = np.arctan2(im_q1, real_q1)
    thetas = np.where(thetas < 0, thetas + 2 * np.pi, thetas)

    b = K * thetas / (2 * np.pi)
    b_floor = np.floor(b).astype(int)
    k0 = (b_floor % K).astype(np.int64)
    log.info("Initial assignment k0 computed")

    phi_hat = 0.5 - b + b_floor
    phis = 2 * np.pi * phi_hat / K
    sorted_idx = np.argsort(phis).astype(np.int64)

    num_candidates = n + 1
    total_tasks = (num_candidates + candidates_per_task - 1) // candidates_per_task

    resources = ray.available_resources()
    num_cpus = max(1, int(resources.get("CPU", 1)))

    log.info(f"Total candidates (prefix lengths): {num_candidates}")
    log.info(f"Detected CPUs: {num_cpus}")
    log.info(f"candidates_per_task: {candidates_per_task}")
    log.info(f"Total Ray tasks to submit (ceil(n/candidates_per_task)): {total_tasks}")

    k0_ref = ray.put(k0)
    sorted_idx_ref = ray.put(sorted_idx)

    def batched_l_values():
        iterator = iter(range(num_candidates))
        while True:
            batch = list(itertools.islice(iterator, candidates_per_task))
            if not batch:
                break
            yield batch

    start_time = time.time()

    max_in_flight = max(2 * num_cpus, 4 * len(gpu_actors))
    in_flight = []
    submitted = 0
    completed = 0

    best_score = float("-inf")
    best_l = 0

    num_gpu = len(gpu_actors)

    batch_id = 0
    for batch in batched_l_values():
        actor = gpu_actors[batch_id % num_gpu]
        in_flight.append(
            process_rank_1_batch_hybrid.remote(
                batch, k0_ref, sorted_idx_ref, int(K), int(batch_id), actor
            )
        )
        submitted += 1
        batch_id += 1

        if len(in_flight) >= max_in_flight:
            done, in_flight = ray.wait(in_flight, num_returns=1)
            batch_score, batch_best_l, b_id = ray.get(done[0])
            completed += 1

            log.info(f"Completed task {b_id}")
            if batch_best_l is not None and batch_score > best_score:
                best_score = float(batch_score)
                best_l = int(batch_best_l)
                log.info(f"New best from task {b_id}: score={best_score} (l={best_l})")

    while in_flight:
        done, in_flight = ray.wait(in_flight, num_returns=1)
        batch_score, batch_best_l, b_id = ray.get(done[0])
        completed += 1

        log.info(f"Completed task {b_id}")
        if batch_best_l is not None and batch_score > best_score:
            best_score = float(batch_score)
            best_l = int(batch_best_l)
            log.info(f"New best from task {b_id}: score={best_score} (l={best_l})")

    elapsed = time.time() - start_time
    log.info(f"Rank 1 (hybrid GPU) search complete in {elapsed:.4f} seconds")
    log.info(f"Submitted={submitted}, completed={completed}, max_in_flight={max_in_flight}")
    log.info(f"Best prefix l = {best_l}")

    best_k = k0.copy()
    if best_l > 0:
        best_k[sorted_idx[:best_l]] = (best_k[sorted_idx[:best_l]] + 1) % K

    roots = np.exp(2 * np.pi * 1j * np.arange(K) / K)
    best_z = roots[best_k]

    return best_score, best_k, best_z, best_l


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel MAX k CUT experiment (rank 1, GPU)")
    parser.add_argument("--n", type=int, default=10000, help="Problem size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--precision",
        type=int,
        default=64,
        choices=[16, 32, 64],
        help="Numeric precision: 16, 32, or 64 (default: 64)",
    )
    parser.add_argument(
        "--candidates_per_task",
        type=int,
        default=64,
        help="How many candidates (l values) each CPU Ray task evaluates (batched on GPU).",
    )
    parser.add_argument("--K", type=int, default=3, help="Number of partitions (default: 3)")
    parser.add_argument("--debug", action="store_true", help="Compute correctness with opt_K_cut")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to store outputs")
    parser.add_argument(
        "--graph_dir",
        type=str,
        default=None,
        help="Directory containing Q_{n}.npy and V_{n}.npy",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="If >0, cap number of GPU actors to this many. Default uses all Ray-visible GPUs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    float_dtype, complex_dtype = set_numpy_precision(args.precision)
    log.info(f"Using precision={args.precision} -> float={float_dtype.__name__}, complex={complex_dtype.__name__}")

    np.random.seed(args.seed)

    log.info("Starting MAX k CUT experiment (rank 1, GPU)")
    ray.init(address="auto", ignore_reinit_error=True)
    log.info("Ray initialized")

    resources = ray.available_resources()
    num_workers = int(resources.get("CPU", 1))
    num_gpus_visible = int(resources.get("GPU", 0))
    log.info(f"Detected {num_workers} Ray workers (CPU slots)")
    log.info(f"Detected {num_gpus_visible} Ray GPUs")

    if num_gpus_visible < 1:
        raise SystemExit("No GPU detected by Ray. Ensure CUDA_VISIBLE_DEVICES is set and Ray sees GPUs.")

    num_gpu_actors = num_gpus_visible if int(args.gpus) <= 0 else min(int(args.gpus), num_gpus_visible)
    gpu_actors = [Rank1GPUActor.remote(K=int(args.K), precision=int(args.precision)) for _ in range(num_gpu_actors)]
    log.info(f"Spawned {len(gpu_actors)} Rank1GPUActor(s).")

    # Load Q and V (same behavior as your original)
    if not args.debug:
        log.info("Loading Q and V...")
        if args.graph_dir is not None:
            q_path = os.path.join(args.graph_dir, f"Q_{args.n}.npy")
            v_path = os.path.join(args.graph_dir, f"V_{args.n}.npy")

            log.info(f"Loading Q from {q_path}")
            log.info(f"Loading V from {v_path}")

            Q = np.load(q_path).astype(float_dtype, copy=False)
            V = np.asarray(np.load(v_path), dtype=complex_dtype)
        else:
            Q = np.asarray(generate_Q(0.5, args.n, "erdos_renyi", seed=args.seed), dtype=float_dtype)
            log.info("Random graph Laplacian generated")
            eigvals, eigvecs = np.linalg.eigh(Q.astype(np.float64, copy=False))
            _, V = low_rank_matrix(Q, eigvals, eigvecs, r=1)
            V = np.asarray(V, dtype=complex_dtype)
            log.info("Eigen decomposition complete and top eigenvector extracted")
    else:
        log.info("Generating debug low rank Q, V (rank 1)")
        Q, V = generate_debug_QV(n=args.n, rank=1, seed=args.seed)
        Q = np.asarray(Q, dtype=float_dtype)
        V = np.asarray(V, dtype=complex_dtype)

    # Broadcast Q to all GPU actors
    ray.get([a.set_instance.remote(Q) for a in gpu_actors])

    log.info("Executing parallel rank 1 algorithm (GPU scoring)")
    start = time.time()

    # V can be (n,1) or (n,), handle both
    v1 = V[:, 0] if V.ndim == 2 else V
    best_score, best_k, best_z, best_l = process_rank_1_parallel_gpu(
        v1,
        Q,
        K=int(args.K),
        candidates_per_task=int(args.candidates_per_task),
        gpu_actors=gpu_actors,
    )

    elapsed = time.time() - start

    log.info(f"Rank 1 result: score = {best_score}")
    log.info(f"Execution time: {elapsed:.4f} seconds")
    log.info(f"candidates_per_task={args.candidates_per_task}")
    log.info(f"best_l={best_l}")

    output = {
        "n": int(args.n),
        "seed": int(args.seed),
        "rank": 1,
        "precision": int(args.precision),
        "candidates_per_task": int(args.candidates_per_task),
        "K": int(args.K),
        "best_l": int(best_l),
        "best_score": float(best_score),
        "time_seconds": float(elapsed),
        "best_k": np.asarray(best_k).tolist(),
        "best_z_real": np.real(best_z).tolist(),
        "best_z_imag": np.imag(best_z).tolist(),
        "num_workers": int(num_workers),
        "num_gpus": int(len(gpu_actors)),
    }

    if args.debug:
        log.info("Computing optimal K-cut...")
        opt_score, _ = opt_K_cut(Q.astype(np.float64, copy=False), K=int(args.K))
        log.info(f"Correct score: {opt_score}")
        output["opt_score"] = float(opt_score)

    os.makedirs(args.results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_result_n{args.n}_r1_p{args.precision}_K{args.K}_cpt{args.candidates_per_task}_gpu.json"
    path = os.path.join(args.results_dir, filename)

    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved results to {path}")

    ray.shutdown()


if __name__ == "__main__":
    main()
