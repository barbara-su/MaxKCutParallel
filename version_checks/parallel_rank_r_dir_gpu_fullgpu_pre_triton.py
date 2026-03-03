import argparse
import json
import logging
import math
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import ray

_TRITON_AVAILABLE = False

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


def _unrank_combination_lex(n: int, k: int, rank: int) -> np.ndarray:
    """
    Unrank a 0-based lexicographic combination, matching itertools.combinations(range(n), k).
    Returns shape (k,) int64.
    """
    if k < 0 or n < 0 or k > n:
        raise ValueError("Invalid n/k for combination unranking")
    total = math.comb(n, k)
    if rank < 0 or rank >= total:
        raise ValueError(f"rank out of bounds: rank={rank}, total={total}")

    out = np.empty(k, dtype=np.int64)
    x = 0
    r = int(rank)
    for i in range(k):
        while True:
            count = math.comb(n - x - 1, k - i - 1)
            if r < count:
                out[i] = x
                x += 1
                break
            r -= count
            x += 1
    return out


def _next_combination_inplace(c: np.ndarray, n: int, k: int) -> bool:
    """
    Advance c (shape (k,)) to next lexicographic combination in-place.
    Returns False if c was already the last one.
    """
    for i in range(k - 1, -1, -1):
        max_i = n - k + i
        if int(c[i]) < max_i:
            c[i] += 1
            for j in range(i + 1, k):
                c[j] = c[j - 1] + 1
            return True
    return False


def _build_combination_batch_from_rank(n: int, k: int, start_rank: int, batch_size: int) -> np.ndarray:
    """
    Build a contiguous lexicographic batch of combinations directly into an array.
    Output shape: (batch_size, k), dtype=int64.
    """
    if batch_size <= 0:
        return np.empty((0, k), dtype=np.int64)

    batch = np.empty((batch_size, k), dtype=np.int64)
    cur = _unrank_combination_lex(n, k, int(start_rank))
    batch[0] = cur
    for row in range(1, batch_size):
        ok = _next_combination_inplace(cur, n, k)
        if not ok:
            raise RuntimeError("Unexpected end while building combination batch")
        batch[row] = cur
    return batch


def _write_json_atomic(path: Path, payload: Dict[str, object]) -> None:
    tmp_path = Path(f"{path}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)


def _load_json_if_exists(path: Optional[Path]) -> Optional[Dict[str, object]]:
    if path is None or not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def _auto_gpu_inner_batch_size(n: int) -> int:
    # Rough peak working-set estimate is ~32 * n * B bytes
    # (Y complex + k int64 + Zcat float32). Bias toward throughput; callers
    # can still force a lower value via --gpu_inner_batch_size.
    target_workspace_bytes = 8 << 30
    est = target_workspace_bytes // (32 * max(1, int(n)))
    return max(4096, min(262144, int(est)))


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
        
        # Q is real.
        self.qdtype = getattr(torch, qdtype_name)

        kk = torch.arange(self.K, device=self.device, dtype=torch.float32)
        self.roots = torch.exp(2j * torch.pi * kk / self.K).to(self.cdtype)  # (K,)
        self.roots_conj = torch.conj(self.roots)
        ang = torch.tensor(np.pi / self.K, device=self.device, dtype=self.qdtype)
        self.sin_last = torch.sin(ang)
        self.cos_last = torch.cos(ang)

        self.V = None  # (n,r) complex
        self.Q = None  # (n,n) real (float32)
        self.V_tilde = None  # (K*n, 2r) real (float32)
        self.n = None
        self.r = None
        self._zcat_buf = None  # (n, >=2B) real workspace for one-GEMM scoring
        self._comb_lut = None  # (N+1, kmax+1) int64 on GPU, for lex unranking
        self.gpu_inner_batch_size = 4096

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
        self.gpu_inner_batch_size = _auto_gpu_inner_batch_size(int(self.n))

        # build Q as writable and contiguous
        Q_arr = np.asarray(Q_np)
        if (not Q_arr.flags.writeable) or (not Q_arr.flags.c_contiguous):
            Q_arr = np.array(Q_arr, copy=True, order="C")
        Q_t = torch.as_tensor(Q_arr)
        self.Q = Q_t.to(dtype=self.qdtype, device=self.device).contiguous()

        if V_tilde_np is None:
            self.V_tilde = None
            self._comb_lut = None
        else:
            VT_arr = np.asarray(V_tilde_np)
            if (not VT_arr.flags.writeable) or (not VT_arr.flags.c_contiguous):
                VT_arr = np.array(VT_arr, copy=True, order="C")
            VT_t = torch.as_tensor(VT_arr)
            self.V_tilde = VT_t.to(dtype=self.qdtype, device=self.device).contiguous()

            # Precompute binomial lookup on GPU: C(n,k) for n in [0..N], k in [0..2r-1].
            N = int(self.V_tilde.shape[0])
            kmax = max(1, 2 * int(self.r) - 1)
            lut_np = np.zeros((N + 1, kmax + 1), dtype=np.int64)
            lut_np[:, 0] = 1
            for nn in range(1, N + 1):
                upper = min(nn, kmax)
                for kk in range(1, upper + 1):
                    if kk == nn:
                        lut_np[nn, kk] = 1
                    else:
                        lut_np[nn, kk] = lut_np[nn - 1, kk - 1] + lut_np[nn - 1, kk]
            self._comb_lut = torch.as_tensor(lut_np, device=self.device, dtype=torch.int64)

    def _build_index_batch_gpu(self, start_rank: int, batch_size: int, comb_size: int):
        """
        Build lexicographic combinations on GPU by unranking a contiguous rank range.
        Output: I of shape (B, comb_size), int64, on GPU.
        """
        import torch

        if self.V_tilde is None or self._comb_lut is None:
            raise RuntimeError("V_tilde / comb LUT is not initialized in actor")

        N = int(self.V_tilde.shape[0])
        B = int(batch_size)
        k = int(comb_size)
        if B <= 0:
            return torch.empty((0, k), device=self.device, dtype=torch.int64)
        if k <= 0 or k > N:
            raise ValueError(f"Invalid comb_size={k} for N={N}")

        remaining = torch.arange(B, device=self.device, dtype=torch.int64) + int(start_rank)
        out = torch.empty((B, k), device=self.device, dtype=torch.int64)
        x = torch.zeros((B,), device=self.device, dtype=torch.int64)

        for pos in range(k):
            rem_pick = k - pos - 1
            max_x = N - (k - pos)
            while True:
                count = self._comb_lut[N - x - 1, rem_pick]
                go_next = (remaining >= count) & (x < max_x)
                if not bool(torch.any(go_next).item()):
                    break
                remaining = torch.where(go_next, remaining - count, remaining)
                x = torch.where(go_next, x + 1, x)
            out[:, pos] = x
            x = x + 1
        return out

    def _quantize_nearest_root(self, Y):
        import torch

        best_proj = (Y * self.roots_conj[0]).real
        k = torch.zeros_like(best_proj, dtype=torch.int64)
        for root_id in range(1, self.K):
            proj = (Y * self.roots_conj[root_id]).real
            better = proj > best_proj
            best_proj = torch.where(better, proj, best_proj)
            k = torch.where(
                better,
                torch.tensor(root_id, device=self.device, dtype=torch.int64),
                k,
            )
        return k

    def _ctilde_to_complex_single(self, c_tilde, r: int):
        re = c_tilde[0 : 2 * r : 2]
        im = c_tilde[1 : 2 * r : 2]
        return re.to(self.cdtype) + (1j * im.to(self.cdtype))

    def _find_intersection_fixed_angle_single(self, VI_minus, r: int):
        """
        Torch equivalent of utils.find_intersection_fixed_angle for one candidate.
        Returns c_tilde on success, else None.
        """
        import torch

        m = 2 * int(r) - 2
        A = VI_minus[:, :m]
        tail = VI_minus[:, m : m + 2]
        b = -(tail[:, 0] * self.sin_last + tail[:, 1] * self.cos_last)

        phi_reduced = None
        try:
            x, info = torch.linalg.solve_ex(A, b.unsqueeze(-1))
            if int(info.item()) == 0:
                phi_reduced = x.squeeze(-1)
        except RuntimeError:
            phi_reduced = None

        if phi_reduced is None:
            try:
                phi_reduced = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)
            except RuntimeError:
                return None

        c_tilde = torch.zeros(2 * int(r), device=self.device, dtype=self.qdtype)
        if m > 0:
            cosv = torch.cos(phi_reduced)
            sinv = torch.sin(phi_reduced)
            prefix = torch.empty(m + 1, device=self.device, dtype=self.qdtype)
            prefix[0] = 1.0
            prefix[1:] = torch.cumprod(cosv, dim=0)
            c_tilde[:m] = prefix[:m] * sinv
            prod_cos = prefix[-1]
        else:
            prod_cos = torch.tensor(1.0, device=self.device, dtype=self.qdtype)
        c_tilde[2 * int(r) - 2] = prod_cos * self.sin_last
        c_tilde[2 * int(r) - 1] = prod_cos * self.cos_last
        return c_tilde

    def _exact_refine_best_candidate(self, I_best, c_best, r: int):
        """
        Baseline-exact used-vertex refinement for a single best candidate.
        Returns (refined_score_float, refined_k_cpu_np, refined_z_cpu_np) or None on failure.
        """
        import torch

        VI = self.V_tilde[I_best]  # (2r-1, 2r)
        v_used = torch.unique(
            torch.div(I_best, int(self.K), rounding_mode="floor"),
            sorted=True,
        )

        rows_out: List[int] = []
        vals_out: List[int] = []
        comb_size = int(I_best.shape[0])

        for v_idx_t in v_used:
            v_idx = int(v_idx_t.item())
            assigned = False
            for rotation in range(int(self.K)):
                vtilde_idx = v_idx * int(self.K) + rotation
                pos_t = torch.nonzero(I_best == vtilde_idx, as_tuple=False)
                if pos_t.numel() == 0:
                    continue
                pos = int(pos_t[0].item())
                if pos == 0:
                    VI_minus = VI[1:, :]
                elif pos == comb_size - 1:
                    VI_minus = VI[:-1, :]
                else:
                    VI_minus = torch.cat((VI[:pos, :], VI[pos + 1 :, :]), dim=0)

                new_c_tilde = self._find_intersection_fixed_angle_single(VI_minus, int(r))
                if new_c_tilde is None:
                    continue
                new_c = self._ctilde_to_complex_single(new_c_tilde, int(r))
                v_c = torch.sum(self.V[v_idx, : int(r)] * new_c)
                root_id = int(torch.argmax((torch.conj(self.roots) * v_c).real).item())
                rows_out.append(v_idx)
                vals_out.append(root_id)
                assigned = True
                break

            if not assigned:
                v_c = torch.sum(self.V[v_idx, : int(r)] * c_best)
                root_id = int(torch.argmax((torch.conj(self.roots) * v_c).real).item())
                rows_out.append(v_idx)
                vals_out.append(root_id)

        # Re-quantize and apply exact overrides (same semantics as score_batch for B=1).
        y1 = torch.matmul(self.V[:, : int(r)], c_best)  # (n,)
        k1 = self._quantize_nearest_root(y1.unsqueeze(1)).squeeze(1)
        if len(rows_out) > 0:
            rows_t = torch.as_tensor(rows_out, device=self.device, dtype=torch.int64)
            vals_t = torch.as_tensor(vals_out, device=self.device, dtype=torch.int64)
            k1[rows_t] = vals_t

        z1 = self.roots[k1]  # (n,)
        zr = z1.real
        zi = z1.imag
        qzr = torch.matmul(self.Q, zr)
        qzi = torch.matmul(self.Q, zi)
        score = torch.sum(zr * qzr + zi * qzi)
        score_f = float(torch.round(score).item())
        return score_f, k1.to("cpu").numpy(), z1.to("cpu").numpy()

    def _score_index_batch_tensor(self, I, r: int):
        """
        Shared scorer for I already on GPU: I shape (B, 2r-1), int64.
        """
        import torch

        if I.ndim != 2:
            raise ValueError("I_batch must have shape (B, 2r-1)")

        B_total = int(I.shape[0])
        if B_total == 0:
            return float("-inf"), None, None, 0

        # Baseline-compatible used-vertex mapping:
        # v_tilde_idx = v_idx * K + rotation  =>  v_idx = v_tilde_idx // K
        # (matches src/parallel_rank_r_dir_gpu.py override semantics).
        v = torch.div(I, int(self.K), rounding_mode="floor")  # (B, comb_size)
        B = B_total

        # Gather VI and compute batched null vectors.
        VI = self.V_tilde[I]  # (B, 2r-1, 2r)
        c_tilde, valid_null = self._build_null_vectors_pivot(VI)
        phi, sign_c = self._determine_phi_sign_torch(c_tilde)
        feasible_phi = (-torch.pi / self.K < phi[:, 2 * r - 2]) & (phi[:, 2 * r - 2] <= torch.pi / self.K)

        c_tilde = c_tilde * sign_c.unsqueeze(1)
        C = self._ctilde_to_complex_torch(c_tilde, int(r))  # (B,r)

        # Quantize from Y = V @ C^T
        Y = torch.matmul(self.V[:, : int(r)], C.T)  # (n,B) complex
        k_assign = self._quantize_nearest_root(Y)

        # Baseline-compatible fallback override (no fixed-angle refit):
        # baseline uses v_used = np.unique(I // K), i.e. one override per unique vertex.
        # Keep the same semantics here by deduplicating vertices per candidate.
        comb_size = int(I.shape[1])
        v_sorted, _ = torch.sort(v, dim=1)  # (B, comb_size)
        mask = torch.ones((B, comb_size), device=self.device, dtype=torch.bool)
        if comb_size > 1:
            mask[:, 1:] = v_sorted[:, 1:] != v_sorted[:, :-1]

        rows = v_sorted.transpose(0, 1).contiguous().to(torch.int64)  # (comb_size, B)
        V_rows = self.V[rows, : int(r)]  # (comb_size, B, r)
        vc = torch.sum(V_rows * C.unsqueeze(0), dim=2)  # (comb_size, B) complex
        metric = (vc.unsqueeze(0) * torch.conj(self.roots).view(self.K, 1, 1)).real  # (K, comb_size, B)
        vals = torch.argmax(metric, dim=0).to(torch.int64)  # (comb_size, B)
        cols = (
            torch.arange(B, device=self.device, dtype=torch.int64)
            .view(1, B)
            .expand(comb_size, B)
        )
        mask_t = mask.transpose(0, 1).contiguous()  # (comb_size, B)
        k_assign[rows[mask_t], cols[mask_t]] = vals[mask_t]

        z = self.roots[k_assign]  # (n,B)

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
        best_k = k_assign[:, best_b].to("cpu").numpy()
        best_z = z[:, best_b].to("cpu").numpy()
        refined = self._exact_refine_best_candidate(I[best_b], C[best_b], int(r))
        if refined is not None:
            refined_score, refined_k, refined_z = refined
            if refined_score > best_score:
                best_score, best_k, best_z = refined_score, refined_k, refined_z
        return best_score, best_k, best_z, feasible_count

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
            return self._score_index_batch_tensor(I, int(r))

    def score_rank_batch(self, start_rank: int, batch_size: int, r: int, inner_batch_size: int = 0):
        """
        Full-GPU rank-batch path:
        build I on GPU via unranking, then score in internal chunks to keep
        memory bounded while allowing very large logical batch sizes.
        """
        import torch

        with torch.inference_mode():
            if self.V is None or self.Q is None or self.V_tilde is None:
                raise RuntimeError("Call set_instance(V, Q, V_tilde) before score_rank_batch(...)")

            comb_size = 2 * int(r) - 1
            logical_batch = int(batch_size)
            if logical_batch <= 0:
                return float("-inf"), None, None, 0, 0.0

            chunk_size = int(inner_batch_size) if int(inner_batch_size) > 0 else int(self.gpu_inner_batch_size)
            chunk_size = max(1, min(chunk_size, logical_batch))

            best_score = float("-inf")
            best_k = None
            best_z = None
            feasible_total = 0
            t_start = time.perf_counter()

            for offset in range(0, logical_batch, chunk_size):
                cur = min(chunk_size, logical_batch - offset)
                I = self._build_index_batch_gpu(
                    int(start_rank) + int(offset),
                    int(cur),
                    int(comb_size),
                )
                chunk_score, chunk_k, chunk_z, feasible_count = self._score_index_batch_tensor(I, int(r))
                feasible_total += int(feasible_count)
                if chunk_k is not None and chunk_score > best_score:
                    best_score = float(chunk_score)
                    best_k = chunk_k
                    best_z = chunk_z

            compute_sec = time.perf_counter() - t_start
            return best_score, best_k, best_z, feasible_total, float(compute_sec)

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

            # quantize by nearest root projection (avoids angle/trig over the full matrix)
            k = self._quantize_nearest_root(Y)

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

def process_rankr_single_fullgpu(
    V: np.ndarray,
    Q: np.ndarray,
    K: int,
    candidates_per_task: int,
    max_in_flight_cpu: int,
    gpu_inner_batch_size: int,
    gpu_actors: List["ray.actor.ActorHandle"],
):
    n, r = V.shape
    log.info(f"Rank r subroutine (full-GPU candidate generation): n={n}, r={r}, K={K}")
    log.info("Full-GPU path applies exact baseline used-vertex refinement fully on GPU.")

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
    num_gpu_actors = len(gpu_actors)
    if num_gpu_actors < 1:
        raise RuntimeError("gpu_actors list is empty")
    inner_batch = int(gpu_inner_batch_size) if int(gpu_inner_batch_size) > 0 else _auto_gpu_inner_batch_size(int(n))
    log.info(
        "Ray CPUs=%d, candidates_per_task=%d, total_cpu_tasks=%d, total_gpu_tasks=%d, gpu_inner_batch_size=%d",
        num_cpus,
        candidates_per_task,
        total_tasks,
        total_gpu_tasks,
        inner_batch,
    )

    start_time = time.time()
    if max_in_flight_cpu <= 0:
        max_in_flight = max(2 * num_gpu_actors, 1)
    else:
        max_in_flight = int(max_in_flight_cpu)
    in_flight = []
    in_flight_meta = {}
    submitted = 0
    completed = 0

    best_score = float("-inf")
    best_k = None
    best_z = None
    total_index_sec = 0.0
    total_gpu_compute_sec = 0.0
    total_gpu_queue_wall_sec = 0.0
    total_combos_seen = 0
    total_feasible = 0

    def submit_one(start_rank: int, batch_size: int, batch_id: int, index_sec: float):
        actor = gpu_actors[batch_id % num_gpu_actors]
        t_submit = time.perf_counter()
        fut = actor.score_rank_batch.remote(
            int(start_rank),
            int(batch_size),
            int(r),
            int(inner_batch),
        )
        in_flight_meta[fut] = (int(batch_id), float(index_sec), int(batch_size), float(t_submit))
        return fut

    def handle_done(done_ref):
        nonlocal completed, total_index_sec, total_gpu_compute_sec, total_gpu_queue_wall_sec
        nonlocal total_combos_seen, total_feasible, best_score, best_k, best_z

        batch_score, batch_k, batch_z, feasible_count, compute_sec = ray.get(done_ref)
        b_id, index_sec, combos_seen, t_submit = in_flight_meta.pop(done_ref)
        queue_wall_sec = time.perf_counter() - t_submit

        completed += 1
        total_index_sec += float(index_sec)
        total_gpu_compute_sec += float(compute_sec)
        total_gpu_queue_wall_sec += float(queue_wall_sec)
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
                total_gpu_compute_sec / completed,
                (total_feasible / total_combos_seen) if total_combos_seen > 0 else 0.0,
            )

    for batch_id in range(total_tasks):
        start_rank = batch_id * candidates_per_task
        combos_seen = min(candidates_per_task, num_combinations - start_rank)
        t_index_start = time.perf_counter()
        # Driver bookkeeping only; index generation is inside GPU actor.
        index_sec = time.perf_counter() - t_index_start

        in_flight.append(submit_one(start_rank, combos_seen, batch_id, index_sec))
        submitted += 1

        if len(in_flight) >= max_in_flight:
            done, in_flight = ray.wait(in_flight, num_returns=1)
            handle_done(done[0])

    while in_flight:
        done, in_flight = ray.wait(in_flight, num_returns=1)
        handle_done(done[0])

    elapsed = time.time() - start_time
    log.info(f"Full-GPU rank-r search complete in {elapsed:.4f}s; submitted={submitted}, completed={completed}")
    if completed > 0:
        log.info(
            "Rank-r timing summary: total_cpu_index=%.4fs, total_gpu_batch=%.4fs, avg_cpu_index=%.4fs/task, avg_gpu_batch=%.4fs/task, feasible_ratio=%.4f (%d/%d)",
            total_index_sec,
            total_gpu_compute_sec,
            total_index_sec / completed,
            total_gpu_compute_sec / completed,
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
    gpu_inner_batch_size: int,
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
        gpu_inner_batch_size=gpu_inner_batch_size,
        gpu_actors=gpu_actors,
    )

    log.info(f"Recursing to lower rank r={r-1}")
    lower_score, lower_k, lower_z = process_rankr_recursive_fullgpu(
        V[:, : r - 1],
        Q,
        K=K,
        candidates_per_task=candidates_per_task,
        max_in_flight_cpu=max_in_flight_cpu,
        gpu_inner_batch_size=gpu_inner_batch_size,
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
        help="Logical batch size. For rank>=2 this is the rank-range size; actors internally chunk it for memory. For rank=1 this is l-candidates per CPU task.",
    )
    ap.add_argument(
        "--max_in_flight_cpu",
        type=int,
        default=0,
        help="Max in-flight GPU requests for rank>=2. 0 uses auto=max(2*GPUs,1).",
    )
    ap.add_argument(
        "--gpu_inner_batch_size",
        type=int,
        default=0,
        help="Internal per-actor score chunk size. 0 uses an automatic memory-bounded value based on n.",
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
    log.info("Full-GPU mode: rank>=2 uses GPU unranking + GPU candidate generation + GPU scoring")

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
                gpu_inner_batch_size=int(args.gpu_inner_batch_size),
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
            "gpu_inner_batch_size": int(args.gpu_inner_batch_size),
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
