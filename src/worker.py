"""
Ray-free multi-GPU worker for rank-r Max-K-Cut.
Works on PyTorch 1.12+ (kp machines) and 2.4+ (anton machines).

Usage:
    python worker.py --q_path Q.npy --v_path V.npy --vtilde_path Vtilde.npy \
        --start_rank 0 --end_rank 1000000 --rank 2 --K 3 \
        --num_gpus 4 --out result.json

The coordinator splits the total combination space across machines,
each machine runs this worker which splits its range across local GPUs.
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

# ── PyTorch 1.12 compatibility ──────────────────────────────────────────────

_HAS_SOLVE_EX = hasattr(torch.linalg, "solve_ex")


def solve_ex_compat(A, b):
    """torch.linalg.solve_ex for PyTorch 1.12+.

    On PyTorch < 1.13, solve() raises on singular matrices. We use pinverse
    as a robust fallback that always succeeds (the residual check in
    _build_null_vectors_pivot will catch bad solutions).
    """
    if _HAS_SOLVE_EX:
        return torch.linalg.solve_ex(A, b)
    B = A.shape[0]
    info = torch.zeros(B, dtype=torch.int32, device=A.device)
    try:
        x = torch.linalg.solve(A, b)
        return x, info
    except Exception:
        # Fallback: pseudo-inverse solve, always succeeds
        x = torch.matmul(torch.linalg.pinv(A), b)
        return x, info


# ── Combination unranking (CPU, fed to GPU) ─────────────────────────────────

def unrank_combination(n, k, rank):
    """Unrank a 0-based lexicographic combination C(n,k). Returns (k,) int64."""
    out = np.empty(k, dtype=np.int64)
    x = 0
    for pos in range(k):
        remaining_pick = k - pos - 1
        while True:
            count = math.comb(n - x - 1, remaining_pick)
            if rank < count:
                break
            rank -= count
            x += 1
        out[pos] = x
        x += 1
    return out


def build_combination_batch(n, k, start_rank, batch_size):
    """Build a batch of combinations by unranking a contiguous range."""
    total = math.comb(n, k)
    actual = min(batch_size, total - start_rank)
    if actual <= 0:
        return np.empty((0, k), dtype=np.int64)
    c = unrank_combination(n, k, start_rank)
    out = np.empty((actual, k), dtype=np.int64)
    out[0] = c
    for i in range(1, actual):
        # Advance to next combination in-place
        j = k - 1
        while j >= 0:
            c[j] += 1
            if c[j] <= n - (k - j):
                for m in range(j + 1, k):
                    c[m] = c[m - 1] + 1
                break
            j -= 1
        out[i] = c
    return out


# ── Smart combination generation (valid-only, no filtering) ────────────────

def count_valid_combinations(n, K, r):
    """Count total valid combinations for rank r, K groups of size K, n vertices.

    For r=2, comb_size=3, choosing 3 indices from Kn rows where each group
    of K consecutive rows can contribute at most 2 indices.

    Type A: 3 indices from 3 different groups = C(n,3) * K^3
    Type B: 2 indices from 1 group + 1 from another = n * C(K,2) * (n-1) * K
    """
    if r != 2 or K != 3:
        raise NotImplementedError("Smart combos only implemented for r=2, K=3")
    type_a = math.comb(n, 3) * (K ** 3)  # 27 * C(n,3)
    type_b = n * math.comb(K, 2) * (n - 1) * K  # 9 * n * (n-1)
    return type_a + type_b


def _unrank_valid_combination(n, K, idx):
    """Unrank a single valid combination (r=2, K=3).

    Ordering: Type A first (sorted), then Type B (sorted).
    Returns a sorted tuple of 3 indices.
    """
    type_a_count = math.comb(n, 3) * (K ** 3)

    if idx < type_a_count:
        # Type A: 3 groups, 1 index from each
        per_group_combos = K ** 3  # 27
        group_combo_idx = idx // per_group_combos
        rotation_idx = idx % per_group_combos

        # Unrank 3 groups from n
        g = unrank_combination(n, 3, group_combo_idx)  # 3 group indices

        # Unrank 3 rotations from K^3
        r2 = rotation_idx % K
        rotation_idx //= K
        r1 = rotation_idx % K
        r0 = rotation_idx // K

        indices = sorted([int(g[0] * K + r0), int(g[1] * K + r1), int(g[2] * K + r2)])
        return indices
    else:
        # Type B: 2 from one group + 1 from another
        idx_b = idx - type_a_count
        # Layout: for each paired_group g0 (0..n-1):
        #   for each pair of rotations within g0: C(K,2) = 3 choices
        #     for each single_group g1 (0..n-2, skipping g0):
        #       for each rotation in g1: K = 3 choices
        per_paired_group = math.comb(K, 2) * (n - 1) * K  # 3 * (n-1) * 3
        g0 = idx_b // per_paired_group
        rem = idx_b % per_paired_group

        per_pair = (n - 1) * K
        pair_idx = rem // per_pair
        rem2 = rem % per_pair

        g1_raw = rem2 // K
        rot1 = rem2 % K

        # g1_raw is in [0, n-2], skip g0
        g1 = g1_raw if g1_raw < g0 else g1_raw + 1

        # Unrank pair of rotations from C(K,2)
        pair_combo = unrank_combination(K, 2, pair_idx)
        rot_a, rot_b = int(pair_combo[0]), int(pair_combo[1])

        indices = sorted([int(g0 * K + rot_a), int(g0 * K + rot_b), int(g1 * K + rot1)])
        return indices


def build_valid_combination_batch(n, K, r, start_idx, batch_size):
    """Generate batch_size valid combinations starting from start_idx.

    Returns (B, 2r-1) int64 array of valid index combinations.
    """
    total = count_valid_combinations(n, K, r)
    actual = min(batch_size, total - start_idx)
    if actual <= 0:
        return np.empty((0, 2 * r - 1), dtype=np.int64)

    comb_size = 2 * r - 1
    out = np.empty((actual, comb_size), dtype=np.int64)
    for i in range(actual):
        combo = _unrank_valid_combination(n, K, start_idx + i)
        out[i] = combo
    return out


# ── GPU Kernel (standalone, no Ray) ─────────────────────────────────────────

class GPUKernel:
    """All GPU compute for rank-r candidate scoring. No Ray dependency."""

    def __init__(self, device, K, precision=32):
        self.device = device
        self.K = K
        dtype_name = "float32" if precision in (16, 32) else "float64"
        self.qdtype = getattr(torch, dtype_name)
        self.cdtype = torch.complex64 if dtype_name == "float32" else torch.complex128

        kk = torch.arange(K, device=device, dtype=torch.float32)
        self.roots = torch.exp(2j * torch.pi * kk / K).to(self.cdtype)
        self.roots_conj = torch.conj(self.roots)
        ang = torch.tensor(np.pi / K, device=device, dtype=self.qdtype)
        self.sin_last = torch.sin(ang)
        self.cos_last = torch.cos(ang)

        self.V = None
        self.Q = None
        self.V_tilde = None
        self.n = None
        self.r = None
        self._zcat_buf = None

    def set_instance(self, V_np, Q_np, V_tilde_np):
        """Upload V, Q, V_tilde to GPU."""
        self.V = torch.as_tensor(np.ascontiguousarray(V_np)).to(
            dtype=self.cdtype, device=self.device
        ).contiguous()
        self.n, self.r = self.V.shape

        self.Q = torch.as_tensor(np.ascontiguousarray(Q_np)).to(
            dtype=self.qdtype, device=self.device
        ).contiguous()

        self.V_tilde = torch.as_tensor(np.ascontiguousarray(V_tilde_np)).to(
            dtype=self.qdtype, device=self.device
        ).contiguous()

    def _build_null_vectors_pivot(self, VI):
        """Batched null-space via pivoted solve. VI: (B, m, d), m=d-1."""
        B, m, d = VI.shape
        inf_val = torch.full((B,), float("inf"), device=self.device, dtype=self.qdtype)
        best_res = inf_val.clone()
        best_c = torch.zeros((B, d), device=self.device, dtype=self.qdtype)
        any_success = torch.zeros((B,), device=self.device, dtype=torch.bool)
        all_cols = torch.arange(d, device=self.device, dtype=torch.int64)

        for pivot in range(d):
            cols = all_cols[all_cols != pivot]
            A = VI[:, :, cols]
            b = -VI[:, :, pivot:pivot + 1]
            x, info = solve_ex_compat(A, b)
            x = x.squeeze(-1)

            c = torch.zeros((B, d), device=self.device, dtype=self.qdtype)
            c[:, pivot] = 1.0
            c[:, cols] = x

            res = torch.linalg.norm(
                torch.matmul(VI, c.unsqueeze(-1)).squeeze(-1), dim=1
            )
            ok = info == 0
            any_success = any_success | ok
            res = torch.where(ok, res, inf_val)
            better = res < best_res
            best_res = torch.where(better, res, best_res)
            best_c = torch.where(better.unsqueeze(1), c, best_c)

        eps = 1e-8
        nrm = torch.linalg.norm(best_c, dim=1, keepdim=True)
        good_nrm = nrm.squeeze(1) > eps
        best_c = torch.where(
            good_nrm.unsqueeze(1), best_c / torch.clamp(nrm, min=eps), best_c
        )
        valid = any_success & good_nrm & torch.isfinite(best_res)
        return best_c, valid

    def _determine_phi_sign(self, c_tilde):
        """Batched phi/sign computation."""
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

    def _ctilde_to_complex(self, c_tilde, r):
        """Convert real c_tilde to complex c."""
        re = c_tilde[:, 0:2 * r:2]
        im = c_tilde[:, 1:2 * r:2]
        return re.to(self.cdtype) + 1j * im.to(self.cdtype)

    def _quantize_k3_ids(self, X):
        """Fast K=3 quantization."""
        x_r, x_i = X.real, X.imag
        p0 = x_r
        p1 = (-0.5 * x_r) + (0.8660254037844386 * x_i)
        p2 = (-0.5 * x_r) - (0.8660254037844386 * x_i)
        k = torch.zeros_like(x_r, dtype=torch.int64)
        better1 = p1 > p0
        best = torch.where(better1, p1, p0)
        k = torch.where(better1, torch.ones_like(k), k)
        better2 = p2 > best
        k = torch.where(better2, torch.full_like(k, 2), k)
        return k

    def _quantize_nearest_root(self, Y):
        """Quantize Y to nearest K-th root of unity."""
        if self.K == 3:
            return self._quantize_k3_ids(Y)
        best_proj = (Y * self.roots_conj[0]).real
        k = torch.zeros_like(best_proj, dtype=torch.int64)
        for root_id in range(1, self.K):
            proj = (Y * self.roots_conj[root_id]).real
            better = proj > best_proj
            best_proj = torch.where(better, proj, best_proj)
            k = torch.where(better, torch.tensor(root_id, device=self.device, dtype=torch.int64), k)
        return k

    def _apply_fallback_override(self, I_row, c, r, k_assign):
        """Override used vertices with exact projection."""
        v_used = torch.unique(torch.div(I_row, self.K, rounding_mode="floor"), sorted=True)
        for v_idx_t in v_used:
            v_idx = int(v_idx_t.item())
            v_c = torch.sum(self.V[v_idx, :r] * c)
            root_id = int(torch.argmax((self.roots_conj * v_c).real).item())
            k_assign[v_idx] = root_id
        return k_assign

    def score_batch(self, I_np, r):
        """
        Score a batch of candidate index sets.
        I_np: (B, 2r-1) int64 numpy array
        Returns: (best_score, best_k, best_z, feasible_count)
        """
        with torch.inference_mode():
            I = torch.as_tensor(I_np, device=self.device, dtype=torch.int64)
            B = I.shape[0]
            if B == 0:
                return float("-inf"), None, None, 0

            # Gather and compute null vectors
            VI = self.V_tilde[I]  # (B, 2r-1, 2r)
            c_tilde, valid_null = self._build_null_vectors_pivot(VI)
            phi, sign_c = self._determine_phi_sign(c_tilde)
            feasible_phi = (
                (-torch.pi / self.K < phi[:, 2 * r - 2])
                & (phi[:, 2 * r - 2] <= torch.pi / self.K)
            )

            c_tilde = c_tilde * sign_c.unsqueeze(1)
            C = self._ctilde_to_complex(c_tilde, r)  # (B, r)

            # Project and quantize
            Y = torch.matmul(self.V[:, :r], C.T)  # (n, B)
            k_assign = self._quantize_nearest_root(Y)

            # Fallback override for used vertices
            v = torch.div(I, self.K, rounding_mode="floor")
            v_sorted, _ = torch.sort(v, dim=1)
            comb_size = I.shape[1]
            mask = torch.ones((B, comb_size), device=self.device, dtype=torch.bool)
            if comb_size > 1:
                mask[:, 1:] = v_sorted[:, 1:] != v_sorted[:, :-1]

            rows = v_sorted.T.contiguous().to(torch.int64)
            V_rows = self.V[rows, :r]
            vc = torch.sum(V_rows * C.unsqueeze(0), dim=2)
            if self.K == 3:
                vals = self._quantize_k3_ids(vc)
            else:
                metric = (vc.unsqueeze(0) * self.roots_conj.view(self.K, 1, 1)).real
                vals = torch.argmax(metric, dim=0).to(torch.int64)
            mask_t = mask.T.contiguous()
            cols = torch.arange(B, device=self.device).view(1, B).expand(comb_size, B)
            k_assign[rows[mask_t], cols[mask_t]] = vals[mask_t]

            # Score with one GEMM
            z = self.roots[k_assign]
            zr, zi = z.real, z.imag
            if self._zcat_buf is None or self._zcat_buf.shape[1] < 2 * B:
                self._zcat_buf = torch.empty((self.n, 2 * B), device=self.device, dtype=self.qdtype)
            Zcat = self._zcat_buf[:, :2 * B]
            Zcat[:, :B] = zr
            Zcat[:, B:] = zi
            QZcat = torch.matmul(self.Q, Zcat)
            scores = torch.sum(zr * QZcat[:, :B] + zi * QZcat[:, B:], dim=0)

            valid = valid_null & feasible_phi & torch.isfinite(scores)
            feasible_count = int(valid.sum().item())
            if feasible_count == 0:
                return float("-inf"), None, None, 0

            neg_inf = torch.full_like(scores, float("-inf"))
            scores = torch.where(valid, scores, neg_inf)
            best_b = torch.argmax(scores)
            best_score = float(torch.round(scores[best_b]).item())
            best_k = k_assign[:, best_b].cpu().numpy()
            best_z = z[:, best_b].cpu().numpy()
            return best_score, best_k, best_z, feasible_count


# ── Per-GPU worker process ──────────────────────────────────────────────────

def gpu_worker(
    gpu_id, V_np, Q_np, V_tilde_np, start_rank, end_rank, r, K,
    chunk_size, result_dict, precision=32, smart_combos=False, n_vertices=0
):
    """Run on a single GPU. Writes best result to shared dict."""
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    kernel = GPUKernel(device, K, precision)
    kernel.set_instance(V_np, Q_np, V_tilde_np)

    N = V_tilde_np.shape[0]  # K * n
    comb_size = 2 * r - 1

    best_score = float("-inf")
    best_k = None
    best_z = None
    total_feasible = 0
    total_processed = 0

    offset = start_rank
    batch_id = 0
    t0 = time.time()

    while offset < end_rank:
        cur = min(chunk_size, end_rank - offset)
        if smart_combos:
            I_batch = build_valid_combination_batch(n_vertices, K, r, offset, cur)
        else:
            I_batch = build_combination_batch(N, comb_size, offset, cur)
        if I_batch.shape[0] == 0:
            break

        score, k, z, feas = kernel.score_batch(I_batch, r)
        total_feasible += feas
        total_processed += I_batch.shape[0]

        if k is not None and score > best_score:
            best_score = score
            best_k = k.copy()
            best_z = z.copy()

        offset += cur
        batch_id += 1

        if batch_id % 50 == 0:
            elapsed = time.time() - t0
            rate = total_processed / elapsed if elapsed > 0 else 0
            pct = (offset - start_rank) / max(1, end_rank - start_rank) * 100
            print(
                f"  GPU {gpu_id}: {pct:.1f}% done, "
                f"score={best_score:.0f}, "
                f"rate={rate:.0f} cand/s, "
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
        f"processed={total_processed}, feasible={total_feasible}",
        flush=True,
    )


# ── Main driver ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ray-free multi-GPU rank-r Max-K-Cut worker")
    parser.add_argument("--q_path", type=str, required=True)
    parser.add_argument("--v_path", type=str, required=True)
    parser.add_argument("--vtilde_path", type=str, default="", help="Path to precomputed V_tilde.npy. If empty, compute from V.")
    parser.add_argument("--start_rank", type=int, default=0)
    parser.add_argument("--end_rank", type=int, default=-1, help="-1 means all")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--num_gpus", type=int, default=-1, help="-1 means all visible")
    parser.add_argument("--chunk_size", type=int, default=50000)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--out", type=str, default="result.json")
    parser.add_argument("--smart_combos", action="store_true",
                        help="Generate only valid combinations (skip infeasible). Currently r=2, K=3 only.")
    args = parser.parse_args()

    # Load data
    print(f"Loading Q from {args.q_path}")
    Q = np.load(args.q_path).astype(np.float64 if args.precision == 64 else np.float32)
    print(f"Loading V from {args.v_path}")
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
        print(f"Loading V_tilde from {args.vtilde_path}")
        V_tilde = np.load(args.vtilde_path).astype(np.float64 if args.precision == 64 else np.float32)
    else:
        print("Computing V_tilde...")
        from utils import compute_vtilde
        V_tilde = compute_vtilde(V).astype(np.float64 if args.precision == 64 else np.float32)

    N = K * n
    comb_size = 2 * r - 1

    smart = args.smart_combos
    if smart:
        total_comb = count_valid_combinations(n, K, r)
        print(f"Smart combos mode: {total_comb:,} valid combinations (vs {math.comb(N, comb_size):,} total)")
    else:
        total_comb = math.comb(N, comb_size)

    start = args.start_rank
    end = total_comb if args.end_rank < 0 else min(args.end_rank, total_comb)

    num_gpus = args.num_gpus
    if num_gpus <= 0:
        num_gpus = torch.cuda.device_count()
    num_gpus = min(num_gpus, torch.cuda.device_count())

    work = end - start
    print(f"Instance: n={n}, r={r}, K={K}")
    print(f"Total combinations: C({N},{comb_size}) = {total_comb:,}")
    print(f"This worker: [{start:,}, {end:,}) = {work:,} combinations")
    print(f"GPUs: {num_gpus}")
    print(f"Chunk size: {args.chunk_size}")

    # Split range across GPUs
    per_gpu = work // num_gpus
    remainder = work % num_gpus

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_dict = manager.dict()

    processes = []
    gpu_start = start
    t_total = time.time()

    for g in range(num_gpus):
        gpu_work = per_gpu + (1 if g < remainder else 0)
        gpu_end = gpu_start + gpu_work
        p = mp.Process(
            target=gpu_worker,
            args=(g, V, Q, V_tilde, gpu_start, gpu_end, r, K, args.chunk_size, result_dict, args.precision, smart, n),
        )
        p.start()
        processes.append(p)
        print(f"  Launched GPU {g}: [{gpu_start:,}, {gpu_end:,}) = {gpu_work:,} combinations")
        gpu_start = gpu_end

    for p in processes:
        p.join()

    total_elapsed = time.time() - t_total

    # Merge results
    best_score = float("-inf")
    best_k = None
    best_z = None
    total_feasible = 0
    total_processed = 0

    for g in range(num_gpus):
        res = result_dict[g]
        total_feasible += res["feasible"]
        total_processed += res["processed"]
        if res["best_k"] is not None and res["best_score"] > best_score:
            best_score = res["best_score"]
            best_k = res["best_k"]
            best_z = res["best_z"]

    print(f"\nFinal: score={best_score:.0f}, elapsed={total_elapsed:.1f}s")
    print(f"  Processed: {total_processed:,}, Feasible: {total_feasible:,} ({total_feasible/max(1,total_processed)*100:.1f}%)")
    print(f"  Throughput: {total_processed/total_elapsed:,.0f} candidates/sec ({num_gpus} GPUs)")

    # Save result
    output = {
        "best_score": float(best_score),
        "time_seconds": float(total_elapsed),
        "start_rank": start,
        "end_rank": end,
        "combinations_processed": total_processed,
        "feasible_count": total_feasible,
        "feasible_ratio": total_feasible / max(1, total_processed),
        "num_gpus": num_gpus,
        "n": n,
        "rank": r,
        "K": K,
        "best_k": best_k.tolist() if best_k is not None else None,
        "best_z_real": np.real(best_z).tolist() if best_z is not None else None,
        "best_z_imag": np.imag(best_z).tolist() if best_z is not None else None,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
