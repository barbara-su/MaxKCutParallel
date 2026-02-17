import argparse
import itertools
import json
import logging
import math
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import ray

from utils import (
    set_numpy_precision,
    compute_vtilde,
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
    """
    Convert precision to torch dtype
    """
    if precision in (16, 32):
        return "complex64", "float32"
    if precision == 64:
        return "complex64", "float32"
    raise ValueError("precision must be one of {16,32,64}")

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
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        cdtype_name, qdtype_name = _torch_dtype_names_from_precision(self.precision)
        self.cdtype = getattr(torch, cdtype_name)
        
        # Q in your code is real; keep it real on GPU for speed/memory.
        self.qdtype = getattr(torch, qdtype_name)

        kk = torch.arange(self.K, device=self.device, dtype=torch.float32)
        self.roots = torch.exp(2j * torch.pi * kk / self.K).to(self.cdtype)  # (K,)

        self.V = None  # (n,r) complex
        self.Q = None  # (n,n) real (float32)
        self.V_tilde = None  # (K*n, 2r) real (float32)
        self.n = None
        self.r = None
        self._zcat_buf = None  # (n, >=2B) real workspace for one-GEMM scoring

    def set_instance(self, V_np: np.ndarray, Q_np: np.ndarray, V_tilde_np: np.ndarray = None):
        """
        Initialize Q and V (and optionally V_tilde for full-GPU candidate generation).
        """
        import torch
        
        # build V as writable and contiguous
        V_arr = np.asarray(V_np)
        if (not V_arr.flags.writeable) or (not V_arr.flags.c_contiguous):
            V_arr = np.array(V_arr, copy=True, order="C")
        V_t = torch.as_tensor(V_arr)
        self.V = V_t.to(dtype=self.cdtype, device=self.device).contiguous()
        self.n, self.r = self.V.shape

        # build Q as writable and contiguous
        Q_arr = np.asarray(Q_np)
        if (not Q_arr.flags.writeable) or (not Q_arr.flags.c_contiguous):
            Q_arr = np.array(Q_arr, copy=True, order="C")
        Q_t = torch.as_tensor(Q_arr)
        self.Q = Q_t.to(dtype=self.qdtype, device=self.device).contiguous()

        if V_tilde_np is None:
            self.V_tilde = None
        else:
            VT_arr = np.asarray(V_tilde_np)
            if (not VT_arr.flags.writeable) or (not VT_arr.flags.c_contiguous):
                VT_arr = np.array(VT_arr, copy=True, order="C")
            VT_t = torch.as_tensor(VT_arr)
            self.V_tilde = VT_t.to(dtype=self.qdtype, device=self.device).contiguous()

    def _build_null_vectors_pivot(self, VI):
        """
        VI: (B, m, d) real, with m = d-1.
        Returns:
          c_tilde: (B, d) real
          valid: (B,) bool
        """
        import torch

        B, m, d = VI.shape
        if m != d - 1:
            raise ValueError(f"Expected VI with m=d-1, got m={m}, d={d}")

        inf_val = torch.full((B,), float("inf"), device=self.device, dtype=self.qdtype)
        best_res = inf_val.clone()
        best_c = torch.zeros((B, d), device=self.device, dtype=self.qdtype)
        any_success = torch.zeros((B,), device=self.device, dtype=torch.bool)

        all_cols = torch.arange(d, device=self.device, dtype=torch.int64)
        for pivot in range(d):
            cols = all_cols[all_cols != pivot]
            A = VI[:, :, cols]  # (B,m,m)
            b = -VI[:, :, pivot : pivot + 1]  # (B,m,1)
            x, info = torch.linalg.solve_ex(A, b)  # (B,m,1), (B,)
            x = x.squeeze(-1)

            c = torch.zeros((B, d), device=self.device, dtype=self.qdtype)
            c[:, pivot] = 1.0
            c[:, cols] = x

            res = torch.linalg.norm(torch.matmul(VI, c.unsqueeze(-1)).squeeze(-1), dim=1)
            ok = info == 0
            any_success = any_success | ok
            res = torch.where(ok, res, inf_val)
            better = res < best_res
            best_res = torch.where(better, res, best_res)
            best_c = torch.where(better.unsqueeze(1), c, best_c)

        eps = 1e-8
        nrm = torch.linalg.norm(best_c, dim=1, keepdim=True)
        good_nrm = nrm.squeeze(1) > eps
        best_c = torch.where(good_nrm.unsqueeze(1), best_c / torch.clamp(nrm, min=eps), best_c)
        valid = any_success & good_nrm & torch.isfinite(best_res)
        return best_c, valid

    def _determine_phi_sign_torch(self, c_tilde):
        """
        Batched equivalent of determine_phi_sign_c logic for small dimensions (r<=3).
        Input:
          c_tilde: (B, d) real
        Returns:
          phi: (B, d-1) real
          sign_c: (B,) real
        """
        import torch

        B, d = c_tilde.shape
        phi = torch.zeros((B, d - 1), device=self.device, dtype=self.qdtype)
        eps = 1e-10

        for phi_ind in range(d - 1):
            if phi_ind > 0:
                prod_cos = torch.ones((B,), device=self.device, dtype=self.qdtype)
                tiny = torch.zeros((B,), device=self.device, dtype=torch.bool)
                for i in range(phi_ind):
                    ci = torch.cos(phi[:, i])
                    tiny = tiny | (torch.abs(ci) < eps)
                    prod_cos = prod_cos * ci
                safe = (~tiny) & (torch.abs(prod_cos) > eps)
                arg = torch.zeros((B,), device=self.device, dtype=self.qdtype)
                arg = torch.where(safe, c_tilde[:, phi_ind] / prod_cos, arg)
                arg = torch.clamp(arg, -1.0, 1.0)
                phi[:, phi_ind] = torch.where(tiny, torch.zeros_like(arg), torch.asin(arg))
            else:
                arg = torch.clamp(c_tilde[:, 0], -1.0, 1.0)
                phi[:, 0] = torch.asin(arg)

        j = d - 2
        sign_c = torch.ones((B,), device=self.device, dtype=self.qdtype)
        base = (phi[:, j] != 0.0) & (c_tilde[:, j] != 0.0)
        cos_ok = torch.abs(torch.cos(phi[:, j])) >= eps
        m = base & cos_ok
        val = torch.tan(phi[:, j]) * c_tilde[:, j] * c_tilde[:, j + 1]
        sign_c = torch.where(m, torch.sign(val), sign_c)
        return phi, sign_c

    def _ctilde_to_complex_torch(self, c_tilde, r):
        import torch

        re = c_tilde[:, 0 : 2 * r : 2]
        im = c_tilde[:, 1 : 2 * r : 2]
        return re.to(self.cdtype) + (1j * im.to(self.cdtype))

    def score_index_batch(self, I_batch_np: np.ndarray, r: int):
        """
        Full-GPU path for r>=2:
        CPU provides only index batch I; GPU builds candidates and scores.

        Input:
          I_batch_np: (B, 2r-1) int64 indices into V_tilde rows.
        Returns:
          best_score, best_k, best_z, feasible_count
        """
        import torch

        with torch.inference_mode():
            if self.V is None or self.Q is None or self.V_tilde is None:
                raise RuntimeError("Call set_instance(V, Q, V_tilde) before score_index_batch(...)")

            I_arr = np.asarray(I_batch_np)
            if (not I_arr.flags.writeable) or (not I_arr.flags.c_contiguous):
                I_arr = np.array(I_arr, copy=True, order="C")
            I = torch.as_tensor(I_arr, device=self.device, dtype=torch.int64)
            if I.ndim != 2:
                raise ValueError("I_batch must have shape (B, 2r-1)")

            B = int(I.shape[0])
            if B == 0:
                return float("-inf"), None, None, 0

            # Gather VI and compute batched null vectors.
            VI = self.V_tilde[I]  # (B, 2r-1, 2r)
            c_tilde, valid_null = self._build_null_vectors_pivot(VI)
            phi, sign_c = self._determine_phi_sign_torch(c_tilde)
            feasible_phi = (-torch.pi / self.K < phi[:, 2 * r - 2]) & (phi[:, 2 * r - 2] <= torch.pi / self.K)

            c_tilde = c_tilde * sign_c.unsqueeze(1)
            C = self._ctilde_to_complex_torch(c_tilde, int(r))  # (B,r)

            # Quantize from Y = V @ C^T
            Y = torch.matmul(self.V[:, : int(r)], C.T)  # (n,B) complex
            theta = torch.angle(Y)
            k = torch.round(theta * (self.K / (2 * torch.pi))).to(torch.int64) % self.K
            z = self.roots[k]  # (n,B)

            # Score with one GEMM.
            zr = z.real
            zi = z.imag
            if self._zcat_buf is None or self._zcat_buf.shape[0] != self.n or self._zcat_buf.shape[1] < 2 * B:
                self._zcat_buf = torch.empty((self.n, 2 * B), device=self.device, dtype=self.qdtype)
            Zcat = self._zcat_buf[:, : 2 * B]
            Zcat[:, :B] = zr
            Zcat[:, B : 2 * B] = zi
            QZcat = torch.matmul(self.Q, Zcat)  # (n,2B)
            Qzr, Qzi = QZcat[:, :B], QZcat[:, B:]
            scores = torch.sum(zr * Qzr + zi * Qzi, dim=0)

            valid = valid_null & feasible_phi & torch.isfinite(scores)
            feasible_count = int(valid.sum().item())
            if feasible_count == 0:
                return float("-inf"), None, None, 0

            neg_inf = torch.full_like(scores, float("-inf"))
            scores = torch.where(valid, scores, neg_inf)
            best_b = torch.argmax(scores)
            best_score = float(torch.round(scores[best_b]).item())
            best_k = k[:, best_b].to("cpu").numpy()
            best_z = z[:, best_b].to("cpu").numpy()
            return best_score, best_k, best_z, feasible_count

    def score_batch(
        self,
        C_np: np.ndarray,
        override_triplets_np: np.ndarray,
    ):
        """
        C_np: (B,r) complex on CPU
        override_triplets_np:
          shape (3, M), flattened override triplets so we can apply all overrides in one scatter:
            k[override_rows, override_cols] = override_vals

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

            # quantize by phase rounding
            theta = torch.angle(Y)  # (n,B) float
            k = torch.round(theta * (self.K / (2 * torch.pi))).to(torch.int64) % self.K  # (n,B)

            # apply overrides with one scatter
            if override_triplets_np is not None and override_triplets_np.size > 0:
                triplets_arr = np.asarray(override_triplets_np)
                if (not triplets_arr.flags.writeable) or (not triplets_arr.flags.c_contiguous):
                    triplets_arr = np.array(triplets_arr, copy=True, order="C")
                triplets_t = torch.as_tensor(triplets_arr, device=self.device, dtype=torch.int64)
                k[triplets_t[0], triplets_t[1]] = triplets_t[2]

            z = self.roots[k]  # (n,B)

            # DenseQ scoring with real Q using one GEMM:
            # score_b = zr_b^T Q zr_b + zi_b^T Q zi_b
            zr = z.real
            zi = z.imag
            if self._zcat_buf is None or self._zcat_buf.shape[0] != self.n or self._zcat_buf.shape[1] < 2 * B:
                self._zcat_buf = torch.empty((self.n, 2 * B), device=self.device, dtype=self.qdtype)
            Zcat = self._zcat_buf[:, : 2 * B]
            Zcat[:, :B] = zr
            Zcat[:, B : 2 * B] = zi
            QZcat = torch.matmul(self.Q, Zcat)  # (n, 2B)
            Qzr, Qzi = QZcat[:, :B], QZcat[:, B:]
            scores = torch.sum(zr * Qzr + zi * Qzi, dim=0)  # (B,)
            best_b = torch.argmax(scores)
            
            # account for precision of tf32
            best_score = float(torch.round(scores[best_b]).item())
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

            # score_b = zr_b^T Q zr_b + zi_b^T Q zi_b, with one GEMM.
            zr = zT.real
            zi = zT.imag
            B = int(zr.shape[1])
            if self._zcat_buf is None or self._zcat_buf.shape[0] != self.n or self._zcat_buf.shape[1] < 2 * B:
                self._zcat_buf = torch.empty((self.n, 2 * B), device=self.device, dtype=self.qdtype)
            Zcat = self._zcat_buf[:, : 2 * B]
            Zcat[:, :B] = zr
            Zcat[:, B : 2 * B] = zi
            QZcat = torch.matmul(self.Q, Zcat)  # (n, 2B)
            Qzr, Qzi = QZcat[:, : zr.shape[1]], QZcat[:, zr.shape[1] :]
            scores = torch.sum(zr * Qzr + zi * Qzi, dim=0)
            return scores.to("cpu").numpy()

@ray.remote
def process_combination_index_batch(
    combinations_batch: List[Tuple[int, ...]],
    batch_id: int,
    r: int,
    gpu_actor: "ray.actor.ActorHandle",
):
    """
    CPU does indexing only: tuples -> ndarray.
    GPU actor does candidate generation + scoring.
    """
    t_index_start = time.perf_counter()
    I_batch = np.asarray(combinations_batch, dtype=np.int64)
    if I_batch.ndim != 2:
        I_batch = I_batch.reshape(len(combinations_batch), -1)
    index_sec = time.perf_counter() - t_index_start

    t_gpu_start = time.perf_counter()
    best_score, best_k, best_z, feasible_count = ray.get(gpu_actor.score_index_batch.remote(I_batch, int(r)))
    gpu_sec = time.perf_counter() - t_gpu_start
    return (
        float(best_score),
        best_k,
        best_z,
        int(batch_id),
        float(index_sec),
        float(gpu_sec),
        int(len(combinations_batch)),
        int(feasible_count),
    )

def process_rankr_single_fullgpu(
    V: np.ndarray,
    Q: np.ndarray,
    K: int,
    candidates_per_task: int,
    max_in_flight_cpu: int,
    gpu_actors: List["ray.actor.ActorHandle"],
):
    n, r = V.shape
    log.info(f"Rank r subroutine (full-GPU candidate generation): n={n}, r={r}, K={K}")
    log.info("Full-GPU path currently skips used-vertex override refinement.")

    if candidates_per_task <= 0:
        raise ValueError("--candidates_per_task must be positive")

    log.info("Computing V_tilde")
    V_tilde = compute_vtilde(V)

    # Broadcast current-r instance to all GPU actors.
    t_set_start = time.perf_counter()
    ray.get([a.set_instance.remote(V, Q, V_tilde) for a in gpu_actors])
    t_set_sec = time.perf_counter() - t_set_start
    log.info("Broadcast to GPU actors (V/Q/V_tilde) took %.4fs", t_set_sec)

    num_vtilde_rows = K * n
    comb_size = 2 * r - 1
    if comb_size > num_vtilde_rows:
        raise ValueError("Combination size 2r-1 exceeds K*n")

    num_combinations = math.comb(num_vtilde_rows, comb_size)
    log.info(f"Total (2r-1)-tuples: C({num_vtilde_rows},{comb_size}) = {num_combinations}")

    resources = ray.available_resources()
    num_cpus = max(1, int(resources.get("CPU", 1)))
    total_tasks = (num_combinations + candidates_per_task - 1) // candidates_per_task if num_combinations > 0 else 0
    total_gpu_tasks = total_tasks
    log.info(
        "Ray CPUs=%d, candidates_per_task=%d, total_cpu_tasks=%d, total_gpu_tasks=%d",
        num_cpus,
        candidates_per_task,
        total_tasks,
        total_gpu_tasks,
    )

    def batched_combinations():
        iterator = itertools.combinations(range(num_vtilde_rows), comb_size)
        while True:
            batch = list(itertools.islice(iterator, candidates_per_task))
            if not batch:
                break
            yield batch

    start_time = time.time()
    if max_in_flight_cpu <= 0:
        max_in_flight = max(2 * num_cpus, 1)
    else:
        max_in_flight = int(max_in_flight_cpu)
    in_flight = []
    submitted = 0
    completed = 0

    best_score = float("-inf")
    best_k = None
    best_z = None
    total_index_sec = 0.0
    total_gpu_sec = 0.0
    total_combos_seen = 0
    total_feasible = 0

    num_gpu_actors = len(gpu_actors)
    if num_gpu_actors < 1:
        raise RuntimeError("gpu_actors list is empty")

    def submit_one(batch, batch_id):
        actor = gpu_actors[batch_id % num_gpu_actors]
        return process_combination_index_batch.remote(
            batch,
            batch_id,
            r,
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
                index_sec,
                gpu_sec,
                combos_seen,
                feasible_count,
            ) = ray.get(done[0])
            completed += 1
            total_index_sec += float(index_sec)
            total_gpu_sec += float(gpu_sec)
            total_combos_seen += int(combos_seen)
            total_feasible += int(feasible_count)

            if batch_k is not None and batch_score > best_score:
                best_score = float(batch_score)
                best_k = batch_k
                best_z = batch_z
                log.info(f"New best score from batch {b_id}: {best_score}")

            if completed % 10 == 0:
                log.info(
                    "Progress: submitted=%d, completed=%d, in_flight=%d, avg_cpu_index=%.4fs, avg_gpu_batch=%.4fs, feasible_ratio=%.4f",
                    submitted,
                    completed,
                    len(in_flight),
                    total_index_sec / completed,
                    total_gpu_sec / completed,
                    (total_feasible / total_combos_seen) if total_combos_seen > 0 else 0.0,
                )

    while in_flight:
        done, in_flight = ray.wait(in_flight, num_returns=1)
        (
            batch_score,
            batch_k,
            batch_z,
            b_id,
            index_sec,
            gpu_sec,
            combos_seen,
            feasible_count,
        ) = ray.get(done[0])
        completed += 1
        total_index_sec += float(index_sec)
        total_gpu_sec += float(gpu_sec)
        total_combos_seen += int(combos_seen)
        total_feasible += int(feasible_count)

        if batch_k is not None and batch_score > best_score:
            best_score = float(batch_score)
            best_k = batch_k
            best_z = batch_z
            log.info(f"New best score from batch {b_id}: {best_score}")

        if completed % 10 == 0:
            log.info(
                "Progress: submitted=%d, completed=%d, in_flight=%d, avg_cpu_index=%.4fs, avg_gpu_batch=%.4fs, feasible_ratio=%.4f",
                submitted,
                completed,
                len(in_flight),
                total_index_sec / completed,
                total_gpu_sec / completed,
                (total_feasible / total_combos_seen) if total_combos_seen > 0 else 0.0,
            )

    elapsed = time.time() - start_time
    log.info(f"Full-GPU rank-r search complete in {elapsed:.4f}s; submitted={submitted}, completed={completed}")
    if completed > 0:
        log.info(
            "Rank-r timing summary: total_cpu_index=%.4fs, total_gpu_batch=%.4fs, avg_cpu_index=%.4fs/task, avg_gpu_batch=%.4fs/task, feasible_ratio=%.4f (%d/%d)",
            total_index_sec,
            total_gpu_sec,
            total_index_sec / completed,
            total_gpu_sec / completed,
            (total_feasible / total_combos_seen) if total_combos_seen > 0 else 0.0,
            total_feasible,
            total_combos_seen,
        )

    if best_z is None:
        raise RuntimeError("Full-GPU rank-r algorithm found no feasible candidate")

    return best_score, np.asarray(best_k), np.asarray(best_z)


def process_rankr_recursive_fullgpu(
    V: np.ndarray,
    Q: np.ndarray,
    K: int,
    candidates_per_task: int,
    max_in_flight_cpu: int,
    gpu_actors: List["ray.actor.ActorHandle"],
):
    n, r = V.shape
    log.info(f"Recursive full-GPU solver at r={r}")

    if r == 1:
        log.info("Base case r=1: process_rank_1_parallel_gpu (GPU path)")
        # rank-1 path uses Q only; keep existing implementation.
        ray.get([a.set_instance.remote(V[:, :1], Q, None) for a in gpu_actors])
        best_score, best_k, best_z, _ = process_rank_1_parallel_gpu(
            V[:, 0],
            Q,
            K,
            candidates_per_task=candidates_per_task,
            gpu_actors=gpu_actors,
        )
        return best_score, best_k, best_z

    best_score, best_k, best_z = process_rankr_single_fullgpu(
        V,
        Q,
        K=K,
        candidates_per_task=candidates_per_task,
        max_in_flight_cpu=max_in_flight_cpu,
        gpu_actors=gpu_actors,
    )

    log.info(f"Recursing to lower rank r={r-1}")
    lower_score, lower_k, lower_z = process_rankr_recursive_fullgpu(
        V[:, : r - 1],
        Q,
        K=K,
        candidates_per_task=candidates_per_task,
        max_in_flight_cpu=max_in_flight_cpu,
        gpu_actors=gpu_actors,
    )

    if lower_score > best_score:
        log.info(f"Lower rank {r-1} improved score {best_score} -> {lower_score}")
        best_score, best_k, best_z = lower_score, lower_k, lower_z

    return best_score, best_k, best_z

def parse_args():
    ap = argparse.ArgumentParser(description="Run parallel_rank_r over a directory (full-GPU candidate generation) without restarting Ray.")
    ap.add_argument("--qv_dir", type=str, required=True, help="Directory containing Q*.npy and V*.npy")
    ap.add_argument("--results_dir", type=str, required=True, help="Directory to store outputs (json)")
    ap.add_argument("--rank", type=int, default=2, help="Rank r (1 uses rank-1 routine)")
    ap.add_argument("--K", type=int, default=3, help="Number of partitions (default 3)")
    ap.add_argument("--precision", type=int, default=32, choices=[16, 32, 64], help="Numeric precision")
    ap.add_argument(
        "--candidates_per_task",
        type=int,
        default=256,
        help="Batch size. For rank>=2 this is index-batch size; for rank=1 this is l-candidates per CPU task.",
    )
    ap.add_argument(
        "--max_in_flight_cpu",
        type=int,
        default=0,
        help="Max in-flight CPU index tasks for rank>=2. 0 uses auto=max(2*CPUs,1).",
    )
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
    log.info("Full-GPU mode: CPU does index batching only for rank>=2")

    ray.init(address="auto", ignore_reinit_error=True)
    resources = ray.available_resources()
    num_workers = int(resources.get("CPU", 1))
    num_gpus_visible = int(resources.get("GPU", 0))
    log.info(f"Ray connected. Detected CPU slots: {num_workers}, GPU slots: {num_gpus_visible}")
    
    instances = discover_instances(qv_dir)
    instances = instances[args.start_index:]
    if args.max_instances and args.max_instances > 0:
        instances = instances[: args.max_instances]

    log.info(f"Discovered {len(instances)} instances to run (after slicing) from {qv_dir}")

    # Create one GPU actor per visible GPU
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

        broadcast_sec = 0.0
        t0 = time.time()

        if args.rank == 1:
            # rank-1 path expects Q already resident in actor.
            t_broadcast_start = time.perf_counter()
            ray.get([a.set_instance.remote(V[:, :1], Q, None) for a in gpu_actors])
            broadcast_sec = time.perf_counter() - t_broadcast_start
            best_score, best_k, best_z, best_l = process_rank_1_parallel_gpu(
                V[:, 0],
                Q,
                K=int(args.K),
                candidates_per_task=int(args.candidates_per_task),
                gpu_actors=gpu_actors,
            )
            
        else:
            best_score, best_k, best_z = process_rankr_recursive_fullgpu(
                V,
                Q,
                K=int(args.K),
                candidates_per_task=int(args.candidates_per_task),
                max_in_flight_cpu=int(args.max_in_flight_cpu),
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
            "max_in_flight_cpu": int(args.max_in_flight_cpu),
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
        
        # compute optimal cut if debug enabled
        if args.debug:
            opt_score, _ = opt_K_cut(Q.astype(np.float32, copy=False), K=int(args.K))
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
