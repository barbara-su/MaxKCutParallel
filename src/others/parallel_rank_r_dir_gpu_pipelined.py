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
from numba import njit

from utils import (
    set_numpy_precision,
    compute_vtilde,
    find_intersection,
    find_intersection_fixed_angle,
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

def _roots_numpy(K: int, complex_dtype) -> np.ndarray:
    return np.exp(1j * 2 * np.pi * np.arange(K) / K).astype(complex_dtype, copy=False)


@njit(cache=True)
def _determine_phi_sign_c_numba(c_tilde):
    D = len(c_tilde)
    phi = np.zeros(D - 1)
    for phi_ind in range(D - 1):
        if phi_ind > 0:
            prod_cos = 1.0
            tiny = False
            for i in range(phi_ind):
                c = np.cos(phi[i])
                if abs(c) < 1e-10:
                    tiny = True
                    break
                prod_cos *= c
            if tiny:
                phi[phi_ind] = 0.0
                continue
        else:
            prod_cos = 1.0

        arg = c_tilde[phi_ind] / prod_cos if abs(prod_cos) > 1e-10 else 0.0
        if arg < -1.0:
            arg = -1.0
        elif arg > 1.0:
            arg = 1.0
        phi[phi_ind] = np.arcsin(arg)

    if phi[D - 2] == 0.0 or c_tilde[D - 2] == 0.0:
        sign_c = 1.0
    else:
        if abs(np.cos(phi[D - 2])) < 1e-10:
            sign_c = 1.0
        else:
            val = np.tan(phi[D - 2]) * c_tilde[D - 2] * c_tilde[D - 1]
            if val > 0.0:
                sign_c = 1.0
            elif val < 0.0:
                sign_c = -1.0
            else:
                sign_c = 0.0

    return phi, sign_c


@njit(cache=True)
def _convert_ctilde_to_complex_numba(c_tilde, r):
    out = np.zeros(r, dtype=np.complex64)
    for j in range(r):
        idx = 2 * j
        if idx + 1 < len(c_tilde):
            out[j] = c_tilde[idx] + 1j * c_tilde[idx + 1]
    return out


@njit(cache=True)
def _nearest_root_id_numba(vr, vi, roots_r, roots_i):
    best_idx = 0
    best_metric = roots_r[0] * vr + roots_i[0] * vi
    for i in range(1, roots_r.shape[0]):
        metric = roots_r[i] * vr + roots_i[i] * vi
        if metric > best_metric:
            best_metric = metric
            best_idx = i
    return best_idx

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
        self.n = None
        self.r = None
        self._zcat_buf = None  # (n, >=2B) real workspace for one-GEMM scoring

    def set_instance(self, V_np: np.ndarray, Q_np: np.ndarray):
        """
        Initialize Q and V.
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
def build_candidates_chunk(
    V_tilde: np.ndarray,
    V: np.ndarray,
    K: int,
    r: int,
    combinations_batch: List[Tuple[int, ...]],
    batch_id: int,
):
    """
    CPU producer only.
    Builds feasible candidates and packed override triplets; does not call GPU.
    """
    t_build_start = time.perf_counter()
    complex_dtype = V.dtype
    roots = _roots_numpy(K, complex_dtype)
    roots_r = roots.real.astype(np.float32, copy=False)
    roots_i = roots.imag.astype(np.float32, copy=False)
    num_vtilde_rows = int(V_tilde.shape[0])
    pos_lookup = np.full(num_vtilde_rows, -1, dtype=np.int64)
    combo_positions = np.arange(2 * r - 1, dtype=np.int64)
    C_list: List[np.ndarray] = []
    rows_flat: List[int] = []
    cols_flat: List[int] = []
    vals_flat: List[int] = []

    for combo in combinations_batch:
        I = None
        try:
            I = np.array(combo, dtype=np.int64)
            VI = V_tilde[I]  # (2r-1, 2r)
            pos_lookup[I] = combo_positions
            VI_minus = np.empty((VI.shape[0] - 1, VI.shape[1]), dtype=VI.dtype)

            c_tilde = find_intersection(VI)
            phi, sign_c = _determine_phi_sign_c_numba(c_tilde)

            if not (-np.pi / K < phi[2 * r - 2] <= np.pi / K):
                pos_lookup[I] = -1
                continue
            c_tilde = c_tilde * sign_c
            c = _convert_ctilde_to_complex_numba(c_tilde, r).astype(complex_dtype, copy=False)

            # Mapping from V_tilde row to vertex is fixed by construction:
            # v_tilde_idx = v_idx * K + rotation  =>  v_idx = v_tilde_idx // K.
            v_used = np.unique(I // K)
            idxs_out: List[int] = []
            roots_out: List[int] = []

            for v_idx in v_used:
                v_idx = int(v_idx)
                assigned = False
                for rotation in range(K):
                    vtilde_idx = v_idx * K + rotation
                    pos = pos_lookup[vtilde_idx]
                    if pos < 0:
                        continue
                    if pos > 0:
                        VI_minus[:pos, :] = VI[:pos, :]
                    if pos < VI.shape[0] - 1:
                        VI_minus[pos:, :] = VI[pos + 1 :, :]
                    try:
                        new_c_tilde = find_intersection_fixed_angle(VI_minus, r, K)
                        new_c = _convert_ctilde_to_complex_numba(new_c_tilde, r).astype(complex_dtype, copy=False)
                        v_c = V[v_idx] @ new_c
                        root_id = _nearest_root_id_numba(float(np.real(v_c)), float(np.imag(v_c)), roots_r, roots_i)
                        idxs_out.append(int(v_idx))
                        roots_out.append(int(root_id))
                        assigned = True
                        break
                    except ValueError:
                        continue
                if not assigned:
                    v_c = V[v_idx] @ c
                    root_id = _nearest_root_id_numba(float(np.real(v_c)), float(np.imag(v_c)), roots_r, roots_i)
                    idxs_out.append(int(v_idx))
                    roots_out.append(int(root_id))

            col_idx = len(C_list)
            C_list.append(np.asarray(c, dtype=complex_dtype))
            if len(idxs_out) > 0:
                rows_flat.extend(idxs_out)
                cols_flat.extend([col_idx] * len(idxs_out))
                vals_flat.extend(roots_out)
            pos_lookup[I] = -1
        except (ValueError, np.linalg.LinAlgError):
            if I is not None:
                pos_lookup[I] = -1
            continue

    build_sec = time.perf_counter() - t_build_start
    combos_seen = int(len(combinations_batch))
    feasible_count = int(len(C_list))

    if feasible_count == 0:
        C_np = np.empty((0, r), dtype=complex_dtype)
        override_triplets = np.empty((3, 0), dtype=np.int64)
        return C_np, override_triplets, int(batch_id), float(build_sec), combos_seen, feasible_count

    C_np = np.stack(C_list, axis=0)  # (B,r)
    if len(rows_flat) > 0:
        override_triplets = np.empty((3, len(rows_flat)), dtype=np.int64)
        override_triplets[0, :] = np.asarray(rows_flat, dtype=np.int64)
        override_triplets[1, :] = np.asarray(cols_flat, dtype=np.int64)
        override_triplets[2, :] = np.asarray(vals_flat, dtype=np.int64)
    else:
        override_triplets = np.empty((3, 0), dtype=np.int64)

    return C_np, override_triplets, int(batch_id), float(build_sec), combos_seen, feasible_count

def process_rankr_single_pipelined_gpu(
    V: np.ndarray,
    Q: np.ndarray,
    K: int,
    cpu_combos_per_task: int,
    target_gpu_batch: int,
    max_in_flight_cpu: int,
    max_pending_gpu: int,
    gpu_actors: List["ray.actor.ActorHandle"],
):
    n, r = V.shape
    log.info(f"Rank r subroutine (pipelined GPU): n={n}, r={r}, K={K}")

    if cpu_combos_per_task <= 0:
        raise ValueError("--cpu_combos_per_task must be positive")
    if target_gpu_batch <= 0:
        raise ValueError("--target_gpu_batch must be positive")
    if max_pending_gpu <= 0:
        raise ValueError("--max_pending_gpu must be positive")

    log.info("Computing V_tilde")
    V_tilde = compute_vtilde(V)

    # IMPORTANT:
    # Recursion changes r (and thus V column count). Refresh actor state per rank
    # so score_batch sees matching shapes for V and C.
    t_set_start = time.perf_counter()
    ray.get([a.set_instance.remote(V, Q) for a in gpu_actors])
    t_set_sec = time.perf_counter() - t_set_start
    log.info("Broadcasted rank-specific V/Q to GPU actors in %.4fs", t_set_sec)

    num_vtilde_rows = K * n
    comb_size = 2 * r - 1
    if comb_size > num_vtilde_rows:
        raise ValueError("Combination size 2r-1 exceeds K*n")

    num_combinations = math.comb(num_vtilde_rows, comb_size)
    total_tasks = (num_combinations + cpu_combos_per_task - 1) // cpu_combos_per_task if num_combinations > 0 else 0
    total_gpu_tasks = (num_combinations + target_gpu_batch - 1) // target_gpu_batch if num_combinations > 0 else 0
    resources = ray.available_resources()
    num_cpus = max(1, int(resources.get("CPU", 1)))
    if max_in_flight_cpu <= 0:
        max_in_flight_cpu = max(4 * num_cpus, 64)

    log.info(f"Total (2r-1)-tuples: C({num_vtilde_rows},{comb_size}) = {num_combinations}")
    log.info(
        "Pipeline config: Ray CPUs=%d, cpu_combos_per_task=%d, total_cpu_tasks=%d, target_gpu_batch=%d, total_gpu_tasks=%d, max_in_flight_cpu=%d, max_pending_gpu=%d",
        num_cpus,
        cpu_combos_per_task,
        total_tasks,
        target_gpu_batch,
        total_gpu_tasks,
        max_in_flight_cpu,
        max_pending_gpu,
    )

    V_tilde_ref = ray.put(V_tilde)
    V_ref = ray.put(V)

    def batched_combinations():
        iterator = itertools.combinations(range(num_vtilde_rows), comb_size)
        while True:
            batch = list(itertools.islice(iterator, cpu_combos_per_task))
            if not batch:
                break
            yield batch

    num_gpu_actors = len(gpu_actors)
    if num_gpu_actors < 1:
        raise RuntimeError("gpu_actors list is empty")

    start_time = time.time()
    in_flight_cpu = []
    pending_gpu = []
    next_gpu_actor = 0

    C_chunks: List[np.ndarray] = []
    triplet_chunks: List[np.ndarray] = []
    B_accum = 0
    combos_accum = 0
    feasible_accum = 0

    submitted_cpu = 0
    completed_cpu = 0
    submitted_gpu = 0
    completed_gpu = 0
    total_cpu_build_sec = 0.0
    total_gpu_wall_sec = 0.0
    total_combos_seen = 0
    total_feasible = 0

    best_score = float("-inf")
    best_k = None
    best_z = None

    def submit_gpu_batch():
        nonlocal C_chunks, triplet_chunks, B_accum, combos_accum, feasible_accum, submitted_gpu, next_gpu_actor
        if B_accum <= 0:
            return

        C_big = np.concatenate(C_chunks, axis=0)
        if triplet_chunks:
            override_big = np.concatenate(triplet_chunks, axis=1)
        else:
            override_big = np.empty((3, 0), dtype=np.int64)

        actor = gpu_actors[next_gpu_actor % num_gpu_actors]
        next_gpu_actor += 1

        t_submit = time.perf_counter()
        fut = actor.score_batch.remote(C_big, override_big)
        pending_gpu.append((fut, t_submit, B_accum, combos_accum, feasible_accum))
        submitted_gpu += 1

        C_chunks = []
        triplet_chunks = []
        B_accum = 0
        combos_accum = 0
        feasible_accum = 0

    def drain_one_gpu(block: bool = True):
        nonlocal completed_gpu, total_gpu_wall_sec, best_score, best_k, best_z
        if not pending_gpu:
            return False
        futures = [x[0] for x in pending_gpu]
        if block:
            done, _ = ray.wait(futures, num_returns=1)
            if not done:
                return False
        else:
            done, _ = ray.wait(futures, num_returns=1, timeout=0.0)
            if not done:
                return False

        done_ref = done[0]
        idx = next(i for i, x in enumerate(pending_gpu) if x[0] == done_ref)
        fut, t_submit, batch_B, batch_combos, batch_feasible = pending_gpu.pop(idx)
        batch_score, batch_k, batch_z = ray.get(fut)

        gpu_wall = time.perf_counter() - t_submit
        total_gpu_wall_sec += float(gpu_wall)
        completed_gpu += 1

        if batch_k is not None and batch_score > best_score:
            best_score = float(batch_score)
            best_k = batch_k
            best_z = batch_z
            log.info(
                "New best score from GPU super-batch %d: %.1f (B=%d combos=%d feasible=%d gpu_wall=%.3fs)",
                completed_gpu,
                best_score,
                int(batch_B),
                int(batch_combos),
                int(batch_feasible),
                float(gpu_wall),
            )
        return True

    def handle_cpu_result(cpu_result):
        nonlocal completed_cpu, total_cpu_build_sec, total_combos_seen, total_feasible
        nonlocal C_chunks, triplet_chunks, B_accum, combos_accum, feasible_accum

        C_np, override_triplets, b_id, build_sec, combos_seen, feasible_count = cpu_result
        completed_cpu += 1
        total_cpu_build_sec += float(build_sec)
        total_combos_seen += int(combos_seen)
        total_feasible += int(feasible_count)

        if feasible_count > 0:
            C_chunks.append(C_np)
            if override_triplets.shape[1] > 0:
                trip = np.array(override_triplets, copy=True, order="C")
                trip[1, :] += B_accum
                triplet_chunks.append(trip)
            B_accum += int(feasible_count)

        combos_accum += int(combos_seen)
        feasible_accum += int(feasible_count)

        if B_accum >= target_gpu_batch:
            submit_gpu_batch()

        while len(pending_gpu) >= max_pending_gpu:
            drain_one_gpu(block=True)

        if completed_cpu % 100 == 0:
            avg_build = total_cpu_build_sec / completed_cpu
            feasible_ratio = (total_feasible / total_combos_seen) if total_combos_seen > 0 else 0.0
            log.info(
                "Progress: cpu submitted=%d completed=%d in_flight=%d | gpu submitted=%d completed=%d pending=%d | B_buffer=%d | avg_cpu_build=%.4fs feasible_ratio=%.4f",
                submitted_cpu,
                completed_cpu,
                len(in_flight_cpu),
                submitted_gpu,
                completed_gpu,
                len(pending_gpu),
                B_accum,
                avg_build,
                feasible_ratio,
            )
            drain_one_gpu(block=False)

    batch_id = 0
    for comb_batch in batched_combinations():
        in_flight_cpu.append(
            build_candidates_chunk.remote(
                V_tilde_ref,
                V_ref,
                K,
                r,
                comb_batch,
                batch_id,
            )
        )
        batch_id += 1
        submitted_cpu += 1

        if len(in_flight_cpu) >= max_in_flight_cpu:
            done, in_flight_cpu = ray.wait(in_flight_cpu, num_returns=1)
            handle_cpu_result(ray.get(done[0]))

    while in_flight_cpu:
        done, in_flight_cpu = ray.wait(in_flight_cpu, num_returns=1)
        handle_cpu_result(ray.get(done[0]))

    submit_gpu_batch()
    while pending_gpu:
        drain_one_gpu(block=True)

    elapsed = time.time() - start_time
    feasible_ratio = (total_feasible / total_combos_seen) if total_combos_seen > 0 else 0.0
    log.info(
        "Pipelined GPU rank-r search complete in %.4fs; cpu submitted=%d completed=%d, gpu submitted=%d completed=%d",
        elapsed,
        submitted_cpu,
        completed_cpu,
        submitted_gpu,
        completed_gpu,
    )
    if completed_cpu > 0:
        log.info(
            "Pipelined timing summary: total_cpu_build=%.4fs, total_gpu_wall=%.4fs, avg_cpu_build=%.4fs/task, avg_gpu_wall=%.4fs/super_batch, feasible_ratio=%.4f (%d/%d)",
            total_cpu_build_sec,
            total_gpu_wall_sec,
            total_cpu_build_sec / completed_cpu,
            (total_gpu_wall_sec / completed_gpu) if completed_gpu > 0 else 0.0,
            feasible_ratio,
            total_feasible,
            total_combos_seen,
        )

    if best_z is None:
        raise RuntimeError("Pipelined GPU rank-r algorithm found no feasible candidate")

    return best_score, np.asarray(best_k), np.asarray(best_z)


def process_rankr_recursive_pipelined_gpu(
    V: np.ndarray,
    Q: np.ndarray,
    K: int,
    candidates_per_task: int,
    cpu_combos_per_task: int,
    target_gpu_batch: int,
    max_in_flight_cpu: int,
    max_pending_gpu: int,
    gpu_actors: List["ray.actor.ActorHandle"],
):
    n, r = V.shape
    log.info(f"Recursive pipelined GPU solver at r={r}")

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

    best_score, best_k, best_z = process_rankr_single_pipelined_gpu(
        V,
        Q,
        K=K,
        cpu_combos_per_task=cpu_combos_per_task,
        target_gpu_batch=target_gpu_batch,
        max_in_flight_cpu=max_in_flight_cpu,
        max_pending_gpu=max_pending_gpu,
        gpu_actors=gpu_actors,
    )

    log.info(f"Recursing to lower rank r={r-1}")
    lower_score, lower_k, lower_z = process_rankr_recursive_pipelined_gpu(
        V[:, : r - 1],
        Q,
        K=K,
        candidates_per_task=candidates_per_task,
        cpu_combos_per_task=cpu_combos_per_task,
        target_gpu_batch=target_gpu_batch,
        max_in_flight_cpu=max_in_flight_cpu,
        max_pending_gpu=max_pending_gpu,
        gpu_actors=gpu_actors,
    )

    if lower_score > best_score:
        log.info(f"Lower rank {r-1} improved score {best_score} -> {lower_score}")
        best_score, best_k, best_z = lower_score, lower_k, lower_z

    return best_score, best_k, best_z

def parse_args():
    ap = argparse.ArgumentParser(description="Run parallel_rank_r over a directory (pipelined GPU) without restarting Ray.")
    ap.add_argument("--qv_dir", type=str, required=True, help="Directory containing Q*.npy and V*.npy")
    ap.add_argument("--results_dir", type=str, required=True, help="Directory to store outputs (json)")
    ap.add_argument("--rank", type=int, default=2, help="Rank r (1 uses rank-1 routine)")
    ap.add_argument("--K", type=int, default=3, help="Number of partitions (default 3)")
    ap.add_argument("--precision", type=int, default=32, choices=[16, 32, 64], help="Numeric precision")
    ap.add_argument(
        "--candidates_per_task",
        type=int,
        default=256,
        help="Rank-1 path only: number of l-candidates per CPU task.",
    )
    ap.add_argument(
        "--cpu_combos_per_task",
        type=int,
        default=10000,
        help="Rank>1 producer task size: number of (2r-1)-tuples per CPU task.",
    )
    ap.add_argument(
        "--target_gpu_batch",
        type=int,
        default=500000,
        help="Rank>1 consumer target: score when accumulated feasible candidates reaches this size.",
    )
    ap.add_argument(
        "--max_in_flight_cpu",
        type=int,
        default=0,
        help="Rank>1 producer concurrency cap. 0 means auto (~4x Ray CPUs).",
    )
    ap.add_argument(
        "--max_pending_gpu",
        type=int,
        default=2,
        help="Rank>1 backpressure cap for submitted-but-not-finished GPU super-batches.",
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
    log.info("Numba acceleration: enabled")

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
    log.info(
        "Pipeline args: cpu_combos_per_task=%d, target_gpu_batch=%d, max_in_flight_cpu=%d, max_pending_gpu=%d",
        int(args.cpu_combos_per_task),
        int(args.target_gpu_batch),
        int(args.max_in_flight_cpu),
        int(args.max_pending_gpu),
    )

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

        # rank>=2 path re-broadcasts per recursion level with rank-specific V.
        # Keep explicit broadcast here for rank==1 direct runs.
        broadcast_sec = 0.0
        t0 = time.time()

        if args.rank == 1:
            t_broadcast_start = time.perf_counter()
            ray.get([a.set_instance.remote(V, Q) for a in gpu_actors])
            broadcast_sec = time.perf_counter() - t_broadcast_start
            best_score, best_k, best_z, best_l = process_rank_1_parallel_gpu(
                V[:, 0],
                Q,
                K=int(args.K),
                candidates_per_task=int(args.candidates_per_task),
                gpu_actors=gpu_actors,
            )
            
        else:
            best_score, best_k, best_z = process_rankr_recursive_pipelined_gpu(
                V,
                Q,
                K=int(args.K),
                candidates_per_task=int(args.candidates_per_task),
                cpu_combos_per_task=int(args.cpu_combos_per_task),
                target_gpu_batch=int(args.target_gpu_batch),
                max_in_flight_cpu=int(args.max_in_flight_cpu),
                max_pending_gpu=int(args.max_pending_gpu),
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
            "cpu_combos_per_task": int(args.cpu_combos_per_task),
            "target_gpu_batch": int(args.target_gpu_batch),
            "max_in_flight_cpu": int(args.max_in_flight_cpu),
            "max_pending_gpu": int(args.max_pending_gpu),
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
