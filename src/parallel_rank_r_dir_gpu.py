import argparse
import itertools
import json
import logging
import math
import random
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import ray

from utils import (
    set_numpy_precision,
    compute_vtilde,
    get_row_mapping,
    find_intersection,
    determine_phi_sign_c,
    find_intersection_fixed_angle,
    convert_ctilde_to_complex,
    opt_K_cut,
)
from parallel_rank_1_gpu import process_rank_1_parallel_gpu  # (best_score, best_k, best_z, best_l)

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
    pair it with the corresponding 'V' file obtained by replacing the leading 'Q' with 'V'.
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


def result_already_exists(results_dir: Path, q_path: Path, rank: int) -> bool:
    stem = q_path.stem
    out_path = results_dir / f"{stem}_r{rank}.json"
    return out_path.exists()


def _torch_dtype_names_from_precision(precision: int) -> Tuple[str, str]:
    if precision in (16, 32):
        return "complex64", "float32"
    if precision == 64:
        return "complex128", "float64"
    raise ValueError("precision must be one of {16,32,64}")


def _roots_numpy(K: int, complex_dtype) -> np.ndarray:
    return np.exp(1j * 2 * np.pi * np.arange(K) / K).astype(complex_dtype, copy=False)


def _nearest_root_id(v_c: complex, roots: np.ndarray) -> int:
    metric = np.real(np.conj(roots) * v_c)
    return int(np.argmax(metric))

@ray.remote(num_gpus=1)
class RankRGPUActor:
    """
    GPU actor that does scoring
    score(z) = Re(conj(z)^T Q z)
    """
    def __init__(self, K: int, precision: int):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available (torch.cuda.is_available() is False)")

        self.device = "cuda"
        self.K = int(K)
        self.precision = int(precision)

        cdtype_name, qdtype_name = _torch_dtype_names_from_precision(self.precision)
        self.cdtype = getattr(torch, cdtype_name)
        
        # Q in your code is real; keep it real on GPU for speed/memory.
        self.qdtype = getattr(torch, qdtype_name)

        kk = torch.arange(self.K, device=self.device, dtype=torch.float32)
        self.roots = torch.exp(2j * torch.pi * kk / self.K).to(self.cdtype)  # (K,)

        self.V = None  # (n,r) complex
        self.Q = None  # (n,n) real (float32/float64)
        self.n = None
        self.r = None

    def set_instance(self, V_np: np.ndarray, Q_np: np.ndarray):
        import torch

        # V
        V_arr = np.asarray(V_np)
        if (not V_arr.flags.writeable) or (not V_arr.flags.c_contiguous):
            V_arr = np.array(V_arr, copy=True, order="C")
        V_t = torch.as_tensor(V_arr)
        if V_t.dtype not in (torch.complex64, torch.complex128):
            V_t = V_t.to(torch.complex64)
        self.V = V_t.to(dtype=self.cdtype, device=self.device).contiguous()
        self.n, self.r = self.V.shape

        # Q (assume real PSD / Laplacian style)
        Q_arr = np.asarray(Q_np)
        if (not Q_arr.flags.writeable) or (not Q_arr.flags.c_contiguous):
            Q_arr = np.array(Q_arr, copy=True, order="C")
        Q_t = torch.as_tensor(Q_arr)
        if Q_t.dtype in (torch.complex64, torch.complex128):
            # If someone passes complex Q, keep it complex, but this is unusual.
            Q_t = Q_t.to(self.cdtype)
        else:
            if Q_t.dtype == torch.float16:
                Q_t = Q_t.to(torch.float32)
            self.Q = Q_t.to(dtype=self.qdtype, device=self.device).contiguous()

    def score_batch(self, C_np: np.ndarray, overrides: List[Tuple[np.ndarray, np.ndarray]]):
        """
        C_np: (B,r) complex on CPU
        overrides: length B list of (idxs, root_ids):
          idxs: shape (t,) vertex indices to overwrite
          root_ids: shape (t,) in [0,K)

        Returns: (best_score, best_k (n,), best_z (n,))
        """
        import torch
        with torch.inference_mode():
            if self.V is None or self.Q is None:
                raise RuntimeError("Call set_instance(V, Q) before score_batch(...)")

            C_arr = np.asarray(C_np)
            if (not C_arr.flags.writeable) or (not C_arr.flags.c_contiguous):
                C_arr = np.array(C_arr, copy=True, order="C")
            C = torch.as_tensor(C_arr, device=self.device, dtype=self.cdtype)  # (B,r)
            B = int(C.shape[0])

            # Y = V @ C^T : (n,r) @ (r,B) -> (n,B)
            Y = torch.matmul(self.V, C.T)  # (n,B) complex

            # Quantize by phase rounding
            theta = torch.angle(Y)  # (n,B) float
            k = torch.round(theta * (self.K / (2 * torch.pi))).to(torch.int64) % self.K  # (n,B)

            # Apply overrides
            for b in range(B):
                idxs, root_ids = overrides[b]
                if idxs is None or len(idxs) == 0:
                    continue
                idxs_t = torch.tensor(idxs, device=self.device, dtype=torch.int64)
                roots_t = torch.tensor(root_ids, device=self.device, dtype=torch.int64)
                k[idxs_t, b] = roots_t

            z = self.roots[k]  # (n,B) complex

            # DenseQ scoring: score_b = Re( conj(z_b)^T Q z_b )
            Qz = torch.matmul(self.Q, z.real) + 1j * torch.matmul(self.Q, z.imag)
            
            scores = torch.sum(torch.conj(z) * Qz, dim=0).real  # (B,)
            best_b = torch.argmax(scores)
            best_score = float(scores[best_b].item())
            best_k = k[:, best_b].to("cpu").numpy()
            best_z = z[:, best_b].to("cpu").numpy()
            return best_score, best_k, best_z

    def score_k_batch(self, k_batch_np: np.ndarray) -> np.ndarray:
        """
        Rank-1 scoring helper: scores batches of integer assignments k.

        Input:
          k_batch_np: (B,n) int64 entries in [0,K)
        Output:
          scores_np: (B,) real numpy
        """
        import torch

        with torch.inference_mode():
            if self.Q is None:
                raise RuntimeError("Call set_instance(V, Q) before score_k_batch(...)")

            k_arr = np.asarray(k_batch_np)
            if (not k_arr.flags.writeable) or (not k_arr.flags.c_contiguous):
                k_arr = np.array(k_arr, copy=True, order="C")
            k = torch.as_tensor(k_arr, device=self.device, dtype=torch.int64)
            if k.ndim != 2:
                raise ValueError("k_batch must have shape (B,n)")

            # z: (B,n) complex
            z = self.roots[k]
            zT = z.T  # (n,B)

            # score_b = Re( conj(z_b)^T Q z_b )
            Qzr = torch.matmul(self.Q, zT.real)
            Qzi = torch.matmul(self.Q, zT.imag)
            Qz = Qzr + 1j * Qzi
            
            scores = torch.sum(torch.conj(zT) * Qz, dim=0).real
            return scores.to("cpu").numpy()


# -----------------------------
# CPU Ray task: build batch payload, call GPU once
# -----------------------------
@ray.remote
def process_combination_batch_hybrid(
    V_tilde: np.ndarray,
    V: np.ndarray,
    K: int,
    r: int,
    combinations_batch: List[Tuple[int, ...]],
    batch_id: int,
    row_mapping: Dict[int, Tuple[int, int]],
    inverse_mapping: Dict[int, List[int]],
    gpu_actor: "ray.actor.ActorHandle",
):
    """
    CPU: construct c and used-vertex overrides for each combo.
    GPU: quantize, apply overrides, score, select best in batch.
    """
    t_build_start = time.perf_counter()
    n = V.shape[0]
    complex_dtype = V.dtype
    roots = _roots_numpy(K, complex_dtype)
    C_list: List[np.ndarray] = []
    overrides: List[Tuple[np.ndarray, np.ndarray]] = []
    for combo in combinations_batch:
        try:
            I = np.array(combo, dtype=int)
            VI = V_tilde[I]  # (2r-1, 2r)

            c_tilde = find_intersection(VI)
            phi, sign_c = determine_phi_sign_c(c_tilde)

            if not (-np.pi / K < phi[2 * r - 2] <= np.pi / K):
                continue
            c_tilde = c_tilde * sign_c
            c = convert_ctilde_to_complex(c_tilde, r)  # (r,)
            v_used = set()
            for idx in I:
                v_row, _ = row_mapping[idx]
                v_used.add(v_row)
            idxs_out: List[int] = []
            roots_out: List[int] = []

            for v_idx in v_used:
                vtilde_rows_for_v = [idx for idx in inverse_mapping[v_idx] if idx in I]
                assigned = False
                for vtilde_idx in vtilde_rows_for_v:
                    pos = int(np.where(I == vtilde_idx)[0][0])
                    VI_minus = np.delete(VI, pos, axis=0)
                    try:
                        new_c_tilde = find_intersection_fixed_angle(VI_minus, r, K)
                        new_c = convert_ctilde_to_complex(new_c_tilde, r)
                        v_c = V[v_idx] @ new_c
                        root_id = _nearest_root_id(v_c, roots)
                        idxs_out.append(int(v_idx))
                        roots_out.append(int(root_id))
                        assigned = True
                        break
                    except ValueError:
                        continue
                if not assigned:
                    v_c = V[v_idx] @ c
                    root_id = _nearest_root_id(v_c, roots)
                    idxs_out.append(int(v_idx))
                    roots_out.append(int(root_id))
            C_list.append(np.asarray(c, dtype=complex_dtype))
            overrides.append(
                (np.asarray(idxs_out, dtype=np.int64), np.asarray(roots_out, dtype=np.int64))
            )

        except (ValueError, np.linalg.LinAlgError):
            continue

    build_sec = time.perf_counter() - t_build_start
    if len(C_list) == 0:
        return float("-inf"), None, None, int(batch_id), float(build_sec), 0.0, int(len(combinations_batch)), 0

    C_np = np.stack(C_list, axis=0)  # (B,r)
    t_gpu_start = time.perf_counter()
    best_score, best_k, best_z = ray.get(gpu_actor.score_batch.remote(C_np, overrides))
    gpu_score_sec = time.perf_counter() - t_gpu_start
    return (
        float(best_score),
        best_k,
        best_z,
        int(batch_id),
        float(build_sec),
        float(gpu_score_sec),
        int(len(combinations_batch)),
        int(len(C_list)),
    )

def process_rankr_single_hybrid_gpu(
    V: np.ndarray,
    K: int,
    candidates_per_task: int,
    gpu_actors: List["ray.actor.ActorHandle"],
):
    n, r = V.shape
    log.info(f"Rank r subroutine (hybrid GPU): n={n}, r={r}, K={K}")

    if candidates_per_task <= 0:
        raise ValueError("--candidates_per_task must be positive")

    log.info("Computing V_tilde")
    V_tilde = compute_vtilde(V)

    log.info("Computing row mappings for V_tilde")
    row_mapping, inverse_mapping = get_row_mapping(n, K)

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
    row_mapping_ref = ray.put(row_mapping)
    inverse_mapping_ref = ray.put(inverse_mapping)

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
    best_k = None
    best_z = None
    total_build_sec = 0.0
    total_gpu_score_sec = 0.0
    total_combos_seen = 0
    total_feasible = 0

    num_gpu_actors = len(gpu_actors)
    if num_gpu_actors < 1:
        raise RuntimeError("gpu_actors list is empty")
    
    def submit_one(batch, batch_id):
        # Round-robin dispatch. This is enough to get near-linear GPU scaling early on.
        actor = gpu_actors[batch_id % num_gpu_actors]
        return process_combination_batch_hybrid.remote(
            V_tilde_ref,
            V_ref,
            K,
            r,
            batch,
            batch_id,
            row_mapping_ref,
            inverse_mapping_ref,
            actor,
        )

    batch_id = 0
    for comb_batch in batched_combinations():
        in_flight.append(submit_one(comb_batch, batch_id))
        batch_id += 1
        submitted += 1

        if len(in_flight) >= max_in_flight:
            done, in_flight = ray.wait(in_flight, num_returns=1)
            (
                batch_score,
                batch_k,
                batch_z,
                b_id,
                build_sec,
                gpu_score_sec,
                combos_seen,
                feasible_count,
            ) = ray.get(done[0])
            completed += 1
            total_build_sec += float(build_sec)
            total_gpu_score_sec += float(gpu_score_sec)
            total_combos_seen += int(combos_seen)
            total_feasible += int(feasible_count)

            if batch_k is not None and batch_score > best_score:
                best_score = float(batch_score)
                best_k = batch_k
                best_z = batch_z
                log.info(f"New best score from batch {b_id}: {best_score}")

            if completed % 500 == 0:
                log.info(
                    "Progress: submitted=%d, completed=%d, in_flight=%d, avg_cpu_build=%.4fs, avg_gpu_score=%.4fs, feasible_ratio=%.4f",
                    submitted,
                    completed,
                    len(in_flight),
                    total_build_sec / completed,
                    total_gpu_score_sec / completed,
                    (total_feasible / total_combos_seen) if total_combos_seen > 0 else 0.0,
                )

    while in_flight:
        done, in_flight = ray.wait(in_flight, num_returns=1)
        (
            batch_score,
            batch_k,
            batch_z,
            b_id,
            build_sec,
            gpu_score_sec,
            combos_seen,
            feasible_count,
        ) = ray.get(done[0])
        completed += 1
        total_build_sec += float(build_sec)
        total_gpu_score_sec += float(gpu_score_sec)
        total_combos_seen += int(combos_seen)
        total_feasible += int(feasible_count)

        if batch_k is not None and batch_score > best_score:
            best_score = float(batch_score)
            best_k = batch_k
            best_z = batch_z
            log.info(f"New best score from batch {b_id}: {best_score}")

        if completed % 500 == 0:
            log.info(
                "Progress: submitted=%d, completed=%d, in_flight=%d, avg_cpu_build=%.4fs, avg_gpu_score=%.4fs, feasible_ratio=%.4f",
                submitted,
                completed,
                len(in_flight),
                total_build_sec / completed,
                total_gpu_score_sec / completed,
                (total_feasible / total_combos_seen) if total_combos_seen > 0 else 0.0,
            )

    elapsed = time.time() - start_time
    log.info(f"Hybrid GPU rank-r search complete in {elapsed:.4f}s; submitted={submitted}, completed={completed}")
    if completed > 0:
        log.info(
            "Rank-r timing summary: total_cpu_build=%.4fs, total_gpu_score=%.4fs, avg_cpu_build=%.4fs/task, avg_gpu_score=%.4fs/task, feasible_ratio=%.4f (%d/%d)",
            total_build_sec,
            total_gpu_score_sec,
            total_build_sec / completed,
            total_gpu_score_sec / completed,
            (total_feasible / total_combos_seen) if total_combos_seen > 0 else 0.0,
            total_feasible,
            total_combos_seen,
        )

    if best_z is None:
        raise RuntimeError("Hybrid GPU rank-r algorithm found no feasible candidate")

    return best_score, np.asarray(best_k), np.asarray(best_z)


def process_rankr_recursive_hybrid_gpu(
    V: np.ndarray,
    Q: np.ndarray,
    K: int,
    candidates_per_task: int,
    gpu_actors: List["ray.actor.ActorHandle"],
):
    n, r = V.shape
    log.info(f"Recursive hybrid GPU solver at r={r}")

    if r == 1:
        log.info("Base case r=1: process_rank_1_parallel_gpu (GPU path)")
        best_score, best_k, best_z, _ = process_rank_1_parallel_gpu(
            V[:, 0],
            Q,
            K,
            candidates_per_task=candidates_per_task,
            gpu_actors=gpu_actors,
        )
        return best_score, best_k, best_z

    best_score, best_k, best_z = process_rankr_single_hybrid_gpu(
        V, K=K, candidates_per_task=candidates_per_task, gpu_actors=gpu_actors
    )

    log.info(f"Recursing to lower rank r={r-1}")
    lower_score, lower_k, lower_z = process_rankr_recursive_hybrid_gpu(
        V[:, : r - 1], Q, K=K, candidates_per_task=candidates_per_task, gpu_actors=gpu_actors
    )

    if lower_score > best_score:
        log.info(f"Lower rank {r-1} improved score {best_score} -> {lower_score}")
        best_score, best_k, best_z = lower_score, lower_k, lower_z

    return best_score, best_k, best_z

def parse_args():
    ap = argparse.ArgumentParser(description="Run parallel_rank_r over a directory (hybrid GPU) without restarting Ray.")
    ap.add_argument("--qv_dir", type=str, required=True, help="Directory containing Q*.npy and V*.npy")
    ap.add_argument("--results_dir", type=str, required=True, help="Directory to store outputs (json)")
    ap.add_argument("--rank", type=int, default=2, help="Rank r (1 uses rank-1 routine)")
    ap.add_argument("--K", type=int, default=3, help="Number of partitions (default 3)")
    ap.add_argument("--precision", type=int, default=32, choices=[16, 32, 64], help="Numeric precision")
    ap.add_argument("--candidates_per_task", type=int, default=256, help="Ray batch size (number of combos per CPU task)")
    ap.add_argument("--debug", action="store_true", help="Compute opt_K_cut (only feasible for tiny n)")
    ap.add_argument("--max_instances", type=int, default=0, help="If >0, cap number of instances processed")
    ap.add_argument("--skip_existing", action="store_true", help="Skip if matching result json exists")
    ap.add_argument("--start_index", type=int, default=0, help="Start from this index in sorted instance list")
    ap.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="If >0, cap number of GPU actors to this many. Default uses all Ray-visible GPUs.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    qv_dir = Path(args.qv_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    float_dtype, complex_dtype = set_numpy_precision(args.precision)
    log.info(f"precision={args.precision} -> float={float_dtype.__name__}, complex={complex_dtype.__name__}")

    ray.init(address="auto", ignore_reinit_error=True)
    resources = ray.available_resources()
    num_workers = int(resources.get("CPU", 1))
    num_gpus_visible = int(resources.get("GPU", 0))
    log.info(f"Ray connected. Detected CPU slots: {num_workers}, GPU slots: {num_gpus_visible}")
    if num_gpus_visible < 1:
        raise SystemExit("No GPU detected by Ray. Ensure CUDA_VISIBLE_DEVICES is set and Ray sees GPUs.")

    instances = discover_instances(qv_dir)
    if not instances:
        raise SystemExit(f"No instances found in {qv_dir} matching Q*.npy")

    if args.start_index < 0 or args.start_index >= len(instances):
        raise SystemExit(f"--start_index out of range: {args.start_index} (0..{len(instances)-1})")

    instances = instances[args.start_index:]
    if args.max_instances and args.max_instances > 0:
        instances = instances[: args.max_instances]

    log.info(f"Discovered {len(instances)} instances to run (after slicing) from {qv_dir}")

    # Create one GPU actor per visible GPU (Design A), optionally capped by --gpus
    num_gpu_actors = num_gpus_visible if int(args.gpus) <= 0 else min(int(args.gpus), num_gpus_visible)
    gpu_actors = [
        RankRGPUActor.remote(K=int(args.K), precision=int(args.precision))
        for _ in range(num_gpu_actors)
    ]
    log.info(f"Spawned {len(gpu_actors)} GPU actors (visible GPUs: {num_gpus_visible}).")
    
    for idx, (q_path, v_path) in enumerate(instances):
        log.info("============================================================")
        log.info(f"[{idx+1}/{len(instances)}] rank={args.rank}, K={args.K}")
        
        log.info(f"Q: {q_path}")
        log.info(f"V: {v_path}")

        if args.skip_existing and result_already_exists(results_dir, q_path, args.rank):
            log.info("Skip: result file already exists.")
            continue

        t_load_start = time.perf_counter()
        Q = np.asarray(np.load(q_path), dtype=float_dtype)
        V_full = np.asarray(np.load(v_path), dtype=complex_dtype)
        if V_full.ndim == 1:
            V_full = V_full.reshape(-1, 1)

        if V_full.shape[1] < args.rank:
            raise ValueError(f"V has {V_full.shape[1]} cols but rank={args.rank}")

        V = V_full[:, : args.rank]
        load_sec = time.perf_counter() - t_load_start

        # Broadcast instance to all GPU actors (each keeps its own V and Q resident)
        t_broadcast_start = time.perf_counter()
        ray.get([a.set_instance.remote(V, Q) for a in gpu_actors])
        broadcast_sec = time.perf_counter() - t_broadcast_start

        t0 = time.time()
        if args.rank == 1:
            best_score, best_k, best_z, best_l = process_rank_1_parallel_gpu(
                V[:, 0],
                Q,
                K=int(args.K),
                candidates_per_task=int(args.candidates_per_task),
                gpu_actors=gpu_actors,
            )
        else:
            best_score, best_k, best_z = process_rankr_recursive_hybrid_gpu(
                V,
                Q,
                K=int(args.K),
                candidates_per_task=int(args.candidates_per_task),
                gpu_actors=gpu_actors,
            )
            best_l = None
        elapsed = time.time() - t0

        log.info(f"Done: score={best_score}, time={elapsed:.4f}s")
        log.info(
            "Phase timing: load_qv=%.4fs, broadcast_to_gpu_actors=%.4fs, solve=%.4fs",
            load_sec,
            broadcast_sec,
            elapsed,
        )

        output: Dict[str, object] = {
            "rank": int(args.rank),
            "K": int(args.K),
            "precision": int(args.precision),
            "candidates_per_task": int(args.candidates_per_task),
            "best_score": float(best_score),
            "time_seconds": float(elapsed),
            "best_k": np.asarray(best_k).tolist(),
            "best_z_real": np.real(best_z).tolist(),
            "best_z_imag": np.imag(best_z).tolist(),
            "num_workers": int(num_workers),
            "num_gpus": int(len(gpu_actors)),
            "q_file": str(q_path),
            "v_file": str(v_path),
        }
        
        if best_l is not None:
            output["best_l"] = int(best_l)

        if args.debug:
            opt_score, _ = opt_K_cut(Q.astype(np.float64, copy=False), K=int(args.K))
            output["opt_score"] = float(opt_score)
            log.info(f"opt_score={opt_score}")

        stem = q_path.stem
        fname = f"{stem}_r{args.rank}.json"
        out_path = results_dir / fname
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        log.info(f"Saved: {out_path}")

        del Q, V_full, V, best_z

    log.info("All instances complete.")
    ray.shutdown()


if __name__ == "__main__":
    main()