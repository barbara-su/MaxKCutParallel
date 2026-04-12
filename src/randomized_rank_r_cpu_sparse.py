"""
CPU sparse randomized rank-r solver for Max-K-Cut.

Designed for large n (n >= 50K) where the dense GPU Q matrix doesn't fit in
VRAM. Uses scipy.sparse CSR for the Laplacian and batched numpy operations
for null-vector computation, phi check, projection, quantization, and scoring.

Mirrors the GPU kernel logic in worker.py so results are directly comparable.

Key insight: at fixed sample budget, quality is constant in n (see B2_progress).
This means CPU sparse beats GPU dense above n ~5K because:
  - GPU dense Q is n^2 (40 GB at n=100K, OOMs)
  - CPU sparse L is 5n nnz (8 MB at n=100K)
  - Per-candidate scoring: O(n) sparse matvec vs O(n^2) dense GEMM

Usage:
    python randomized_rank_r_cpu_sparse.py \\
        --q_path L.npz --v_path V.npy \\
        --rank 2 --K 3 --max_samples 1000000 --seed 42 --out result.json

Input formats:
- q_path: .npy (dense Laplacian) or .npz (scipy.sparse)
- v_path: .npy with shape (n, r) complex (top-r eigenvectors scaled by sqrt(eigval))
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy import sparse

from utils import compute_vtilde


# Python 3.7 polyfill for math.comb (added in 3.8)
if not hasattr(math, "comb"):
    def _comb(n, k):
        if k < 0 or k > n:
            return 0
        k = min(k, n - k)
        num = 1
        for i in range(k):
            num = num * (n - i) // (i + 1)
        return num
    math.comb = _comb


# ── I/O helpers ─────────────────────────────────────────────────────────────

def load_laplacian(path):
    """Load L as scipy.sparse.csr_matrix. Accepts .npy (dense) or .npz (sparse)."""
    if path.endswith(".npz"):
        L = sparse.load_npz(path).tocsr().astype(np.float64)
    else:
        Q = np.load(path)
        L = sparse.csr_matrix(Q.astype(np.float64))
    return L


def extract_edge_list(L, tol=1e-10):
    """Extract edge list from a Laplacian L = D - A.

    Returns a dict:
        {
            "rows": int32 (E,),
            "cols": int32 (E,),
            "weights": float32 (E,) or None if unweighted,
            "unweighted": bool,
        }
    Each undirected edge appears once (row < col). Valid only if L is a
    symmetric Laplacian (diagonal = row sums, off-diagonals <= 0).

    Returns None if L does not look like a Laplacian.
    """
    if not sparse.isspmatrix_csr(L):
        L = L.tocsr()

    # Check Laplacian structure: row sums should be ~zero
    row_sums = np.asarray(L.sum(axis=1)).ravel()
    if np.max(np.abs(row_sums)) > tol * max(1.0, np.max(np.abs(L.diagonal()))):
        return None

    coo = sparse.triu(L, k=1).tocoo()
    rows = coo.row.astype(np.int32)
    cols = coo.col.astype(np.int32)
    weights = -coo.data.astype(np.float64)

    if np.any(weights < -tol):
        return None

    keep = weights > tol
    rows = rows[keep]
    cols = cols[keep]
    weights = weights[keep]

    unweighted = bool(np.allclose(weights, 1.0, atol=1e-8))

    return {
        "rows": rows,
        "cols": cols,
        "weights": None if unweighted else weights.astype(np.float32),
        "unweighted": unweighted,
    }


def score_laplacian(k_assign_i8, edge_list, B, cand_chunk=0):
    """Score via Laplacian cut formula: score[b] = 3 * weighted_cut(k[:,b]).

    Processes candidates in chunks of size cand_chunk to bound peak memory.
    Inner-loop peak memory: ~2 * E * cand_chunk bytes for k_r/k_c int8.
    Default: auto-pick cand_chunk to cap inner buffers at ~200 MB.

    For unweighted graphs, uses bool sum; for weighted, uses float32 BLAS.
    """
    erows = edge_list["rows"]
    ecols = edge_list["cols"]
    eweights = edge_list["weights"]
    unweighted = edge_list["unweighted"]
    E = erows.shape[0]

    if cand_chunk <= 0:
        # Cap 2 * E * cand_chunk at ~200 MB
        cand_chunk = max(16, min(512, 100_000_000 // max(E, 1)))

    scores = np.empty(B, dtype=np.float64)
    for cstart in range(0, B, cand_chunk):
        cend = min(cstart + cand_chunk, B)
        cb = cend - cstart
        ka = k_assign_i8[:, cstart:cend]  # (n, cb) int8
        k_r = ka[erows, :]  # (E, cb) int8
        k_c = ka[ecols, :]
        # cut[e, b] = 1 if k_r[e, b] != k_c[e, b] else 0
        if unweighted:
            # sum(cut) along edges → int → cast to float
            scores[cstart:cend] = (k_r != k_c).sum(axis=0, dtype=np.int64)
        else:
            cut = (k_r != k_c).astype(np.float32)
            # scores[b] = sum_e weights[e] * cut[e, b]
            scores[cstart:cend] = eweights @ cut
        del k_r, k_c
    return 3.0 * scores


# ── Random index generation ─────────────────────────────────────────────────

def generate_random_indices(Kn, comb_size, batch_size, rng):
    """Generate batch_size random sorted (comb_size,)-tuples from [0, Kn).

    Uses rejection sampling for duplicates within a row. For Kn >> comb_size
    (typical), duplicate rate is < 0.1%.
    """
    I = rng.integers(0, Kn, size=(batch_size, comb_size), dtype=np.int64)
    I.sort(axis=1)
    if comb_size > 1:
        dups = (I[:, 1:] == I[:, :-1]).any(axis=1)
        retries = 0
        while dups.any() and retries < 5:
            n_bad = int(dups.sum())
            new_I = rng.integers(0, Kn, size=(n_bad, comb_size), dtype=np.int64)
            new_I.sort(axis=1)
            I[dups] = new_I
            dups = (I[:, 1:] == I[:, :-1]).any(axis=1)
            retries += 1
        if dups.any():
            I = I[~dups]
    return I


# ── Batched scoring kernels ─────────────────────────────────────────────────

def batched_null_vectors(VI):
    """Batched null vectors via SVD.

    VI: (B, m, d) where m = d - 1 (so the matrix has a 1-D null space).
    Returns:
        c_tilde: (B, d) — unit-norm null vector for each batch element
        valid: (B,) bool — True if the null vector is well-defined
    """
    # np.linalg.svd handles batched input out of the box.
    # For (B, m, d) with m < d, S has shape (B, m), Vt has shape (B, d, d).
    _, S, Vt = np.linalg.svd(VI, full_matrices=True)
    c_tilde = Vt[:, -1, :]  # (B, d) — last right singular vector

    eps = 1e-10
    # Valid if VI has full row rank (smallest sv > eps relative to largest)
    valid = (S[:, -1] > eps * np.maximum(S[:, 0], eps))
    nrm = np.linalg.norm(c_tilde, axis=1)
    valid = valid & (nrm > eps) & np.all(np.isfinite(c_tilde), axis=1)
    return c_tilde, valid


def batched_phi_sign(c_tilde, eps=1e-10):
    """Batched spherical-coordinate angles and sign correction.

    Mirrors GPUKernel._determine_phi_sign exactly.

    c_tilde: (B, d), unit-norm.
    Returns:
        phi: (B, d-1)
        sign_c: (B,)
    """
    B, d = c_tilde.shape
    phi = np.zeros((B, d - 1), dtype=c_tilde.dtype)

    # phi[0] = arcsin(c_tilde[0])
    phi[:, 0] = np.arcsin(np.clip(c_tilde[:, 0], -1.0, 1.0))

    # Iteratively compute prod_cos and the next phi.
    # Note: GPU version recomputes prod_cos from scratch each iter; we keep
    # the running product for O(d) instead of O(d^2). Mathematically identical.
    prod_cos = np.cos(phi[:, 0])  # (B,)
    for j in range(1, d - 1):
        safe = np.abs(prod_cos) > eps
        arg = np.zeros(B, dtype=c_tilde.dtype)
        np.divide(c_tilde[:, j], prod_cos, out=arg, where=safe)
        arg = np.clip(arg, -1.0, 1.0)
        phi[:, j] = np.where(safe, np.arcsin(arg), 0.0)
        prod_cos = prod_cos * np.cos(phi[:, j])

    # Sign correction at j = d-2
    j = d - 2
    sign_c = np.ones(B, dtype=c_tilde.dtype)
    base = (phi[:, j] != 0.0) & (c_tilde[:, j] != 0.0)
    cos_j = np.cos(phi[:, j])
    cos_ok = np.abs(cos_j) >= eps
    m = base & cos_ok
    val = np.tan(phi[:, j]) * c_tilde[:, j] * c_tilde[:, j + 1]
    sign_c = np.where(m, np.sign(val), sign_c)
    return phi, sign_c


def quantize_k3(yr, yi):
    """Quantize complex array (real/imag split) to nearest cube root of unity.

    Returns int64 array in {0, 1, 2}, same shape as yr.
    """
    p0 = yr
    p1 = -0.5 * yr + 0.8660254037844386 * yi
    p2 = -0.5 * yr - 0.8660254037844386 * yi
    k = np.zeros_like(yr, dtype=np.int64)
    best = p0.copy()
    better1 = p1 > best
    k[better1] = 1
    np.copyto(best, p1, where=better1)
    better2 = p2 > best
    k[better2] = 2
    return k


def quantize_general(Y, roots_conj):
    """General-K quantization. Y: complex, roots_conj: (K,) complex."""
    K = roots_conj.shape[0]
    proj = np.real(Y[None, ...] * roots_conj.reshape((K,) + (1,) * Y.ndim))
    return np.argmax(proj, axis=0).astype(np.int64)


def score_batch_cpu_sparse(L, V, V_tilde, I_batch, r, K, roots, roots_conj,
                            edge_list=None):
    """Score a batch of candidate index tuples.

    L: (n, n) scipy.sparse.csr_matrix (real)
    V: (n, r) complex (top-r scaled eigenvectors)
    V_tilde: (Kn, 2r) real
    I_batch: (B, 2r-1) int64
    edge_list: optional (rows, cols, weights) for Laplacian fast path.
               If provided AND K==3, scores via the cut-edge formula
               (score = 3 * weighted_cut), which is ~10-50x faster than
               sparse matvec at large n.

    Returns: (best_score, best_k, best_z, feasible_count)
    """
    B = I_batch.shape[0]
    if B == 0:
        return -np.inf, None, None, 0

    # Gather VI: (B, 2r-1, 2r)
    VI = V_tilde[I_batch]

    # Null vectors and phi/sign
    c_tilde, valid_null = batched_null_vectors(VI)
    phi, sign_c = batched_phi_sign(c_tilde)
    feasible_phi = (
        (-np.pi / K < phi[:, 2 * r - 2])
        & (phi[:, 2 * r - 2] <= np.pi / K)
    )

    c_tilde = c_tilde * sign_c[:, None]

    # Convert to complex C: (B, r)
    C = (c_tilde[:, 0:2 * r:2] + 1j * c_tilde[:, 1:2 * r:2]).astype(np.complex128)

    # Project Y = V @ C.T → (n, B) complex
    Y = V @ C.T

    # Quantize to nearest root of unity
    yr = Y.real
    yi = Y.imag
    if K == 3:
        k_assign = quantize_k3(yr, yi)
    else:
        k_assign = quantize_general(Y, roots_conj)

    # Fallback override: for each candidate, override its used vertices'
    # assignments with the exact projection. Reuse Y directly.
    v_used = I_batch // K  # (B, 2r-1)
    cols_b = np.broadcast_to(
        np.arange(B, dtype=np.int64)[:, None], v_used.shape
    )
    if K == 3:
        Y_used_r = yr[v_used, cols_b]
        Y_used_i = yi[v_used, cols_b]
        new_k = quantize_k3(Y_used_r, Y_used_i)
    else:
        Y_used = Y[v_used, cols_b]
        new_k = quantize_general(Y_used, roots_conj)

    k_assign[v_used, cols_b] = new_k  # (duplicates write same value)

    # Cast to int8 (values are in {0, 1, 2} for K=3) for memory efficiency
    k_assign_i8 = k_assign.astype(np.int8)
    # Free the large int64 buffer and Y projection
    del k_assign, Y, yr, yi

    # Score
    if edge_list is not None and K == 3:
        # Laplacian fast path: z^* L z = 3 * weighted_cut_size
        scores = score_laplacian(k_assign_i8, edge_list, B)
    else:
        # General sparse matvec path (slower but works for any Q)
        z = roots[k_assign_i8.astype(np.int64)]
        zr_d = np.ascontiguousarray(z.real, dtype=np.float64)
        zi_d = np.ascontiguousarray(z.imag, dtype=np.float64)
        Lzr = L @ zr_d
        Lzi = L @ zi_d
        scores = (zr_d * Lzr).sum(axis=0) + (zi_d * Lzi).sum(axis=0)

    valid = valid_null & feasible_phi & np.isfinite(scores)
    feasible_count = int(valid.sum())
    if feasible_count == 0:
        return -np.inf, None, None, 0

    scores_masked = np.where(valid, scores, -np.inf)
    best_b = int(np.argmax(scores_masked))
    best_score = float(np.round(scores_masked[best_b]))
    best_k = k_assign_i8[:, best_b].astype(np.int64)
    best_z = roots[best_k]
    return best_score, best_k, best_z, feasible_count


# ── Per-process driver ─────────────────────────────────────────────────────

def run_single_process(L, V, V_tilde, max_samples, r, K, batch_size, seed,
                       roots, roots_conj, edge_list=None,
                       verbose=True, log_prefix=""):
    """Run randomized sampling in a single process."""
    Kn = K * V.shape[0]
    comb_size = 2 * r - 1
    rng = np.random.default_rng(seed)

    best_score = -np.inf
    best_k = None
    best_z = None
    total_feasible = 0
    total_processed = 0
    t0 = time.time()
    batch_id = 0

    while total_processed < max_samples:
        cur = min(batch_size, max_samples - total_processed)
        I_batch = generate_random_indices(Kn, comb_size, cur, rng)
        if I_batch.shape[0] == 0:
            total_processed += cur
            batch_id += 1
            continue

        score, k, z, feas = score_batch_cpu_sparse(
            L, V, V_tilde, I_batch, r, K, roots, roots_conj,
            edge_list=edge_list,
        )
        total_feasible += feas
        total_processed += I_batch.shape[0]

        if k is not None and score > best_score:
            best_score = score
            best_k = k
            best_z = z

        batch_id += 1

        if verbose and batch_id % 10 == 0:
            elapsed = time.time() - t0
            rate = total_processed / elapsed if elapsed > 0 else 0
            pct = total_processed / max_samples * 100
            print(
                f"  {log_prefix}{pct:.1f}% ({total_processed:,}/{max_samples:,}), "
                f"score={best_score:.0f}, rate={rate:,.0f}/s, "
                f"feasible={total_feasible}/{total_processed}",
                flush=True,
            )

    elapsed = time.time() - t0
    return {
        "best_score": best_score,
        "best_k": best_k,
        "best_z": best_z,
        "feasible": total_feasible,
        "processed": total_processed,
        "elapsed": elapsed,
    }


def _worker_entry(args):
    """Pool worker entry point. Each worker reloads L from disk to avoid
    pickling a large sparse matrix through the queue."""
    (worker_id, L_path, V, V_tilde, max_samples, r, K, batch_size, seed,
     use_laplacian_fast) = args
    # Cap BLAS threads per worker to avoid oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    L = load_laplacian(L_path)
    edge_list = None
    if use_laplacian_fast:
        edge_list = extract_edge_list(L)
    roots = np.exp(2j * np.pi * np.arange(K) / K)
    roots_conj = np.conj(roots)
    sub_seed = seed * 1000 + worker_id
    res = run_single_process(
        L, V, V_tilde, max_samples, r, K, batch_size, sub_seed,
        roots, roots_conj, edge_list=edge_list,
        verbose=(worker_id == 0),
        log_prefix=f"[w{worker_id}] ",
    )
    return res


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CPU sparse randomized rank-r Max-K-Cut solver"
    )
    parser.add_argument("--q_path", type=str, required=True,
                        help="Laplacian: .npy (dense) or .npz (scipy.sparse)")
    parser.add_argument("--v_path", type=str, required=True)
    parser.add_argument("--vtilde_path", type=str, default="")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--max_samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Parallel worker processes")
    parser.add_argument("--batch_size", type=int, default=0,
                        help="0 = auto-scale by n (caps Y projection at ~1GB)")
    parser.add_argument("--no_laplacian_fast", action="store_true",
                        help="Disable the Laplacian fast-path scoring "
                             "(falls back to sparse matvec). Default: "
                             "auto-detect Laplacian structure.")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    print(f"Loading L from {args.q_path}")
    L = load_laplacian(args.q_path)
    n = L.shape[0]
    print(f"  n={n}, nnz={L.nnz}, density={L.nnz/(n*n):.6f}")

    use_laplacian_fast = not args.no_laplacian_fast
    edge_list = None
    if use_laplacian_fast:
        edge_list = extract_edge_list(L)
        if edge_list is None:
            print("  Not a Laplacian → falling back to sparse matvec scoring")
            use_laplacian_fast = False
        else:
            print(f"  Laplacian detected: {len(edge_list['rows'])} edges "
                  f"({'unweighted' if edge_list['unweighted'] else 'weighted'}, "
                  f"fast-path scoring enabled)")

    # Auto-scale batch size to bound Y = V @ C.T memory at ~1 GB
    # Y is (n, B) complex128 = 16 * n * B bytes; 1 GB → B ≈ 6.25e7 / n
    if args.batch_size <= 0:
        auto_B = max(50, min(2000, int(64_000_000 / max(n, 1))))
        args.batch_size = auto_B
        print(f"  Auto batch_size = {auto_B} (for ~1 GB Y buffer at n={n})")

    print(f"Loading V from {args.v_path}")
    V = np.load(args.v_path)
    if V.ndim == 1:
        V = V.reshape(-1, 1)
    V = V[:, :args.rank].astype(np.complex128)

    if args.vtilde_path and os.path.exists(args.vtilde_path):
        V_tilde = np.load(args.vtilde_path).astype(np.float64)
    else:
        V_tilde = compute_vtilde(V).astype(np.float64)

    r = args.rank
    K = args.K
    Kn = K * n
    comb_size = 2 * r - 1
    total_comb = math.comb(Kn, comb_size)

    print(f"Instance: n={n}, r={r}, K={K}")
    print(f"Total candidates: C({Kn},{comb_size}) = {total_comb:,}")
    print(f"Sampling: {args.max_samples:,} ({args.max_samples/total_comb*100:.6f}%)")
    print(f"Workers: {args.num_workers}, batch_size: {args.batch_size}")

    roots = np.exp(2j * np.pi * np.arange(K) / K)
    roots_conj = np.conj(roots)

    t_start = time.time()

    if args.num_workers <= 1:
        result = run_single_process(
            L, V, V_tilde, args.max_samples, r, K, args.batch_size,
            args.seed, roots, roots_conj, edge_list=edge_list
        )
        best_score = result["best_score"]
        best_k = result["best_k"]
        best_z = result["best_z"]
        total_feasible = result["feasible"]
        total_processed = result["processed"]
    else:
        per_worker = args.max_samples // args.num_workers
        rem = args.max_samples % args.num_workers
        worker_args = []
        for w in range(args.num_workers):
            ws = per_worker + (1 if w < rem else 0)
            worker_args.append(
                (w, args.q_path, V, V_tilde, ws, r, K,
                 args.batch_size, args.seed, use_laplacian_fast)
            )
        with mp.Pool(args.num_workers) as pool:
            results = pool.map(_worker_entry, worker_args)

        best_score = -np.inf
        best_k = None
        best_z = None
        total_feasible = 0
        total_processed = 0
        for res in results:
            total_feasible += res["feasible"]
            total_processed += res["processed"]
            if res["best_k"] is not None and res["best_score"] > best_score:
                best_score = res["best_score"]
                best_k = res["best_k"]
                best_z = res["best_z"]

    total_elapsed = time.time() - t_start
    rate = total_processed / total_elapsed if total_elapsed > 0 else 0

    print(f"\nResult: score={best_score:.0f}")
    print(f"  Samples: {total_processed:,} ({total_processed/total_comb*100:.6f}%)")
    print(f"  Feasible: {total_feasible:,} "
          f"({total_feasible/max(total_processed,1)*100:.1f}%)")
    print(f"  Time: {total_elapsed:.1f}s ({rate:,.0f} cand/s)")

    if args.out:
        out = {
            "n": n,
            "r": r,
            "K": K,
            "best_score": float(best_score) if best_score > -np.inf else None,
            "max_samples": args.max_samples,
            "total_processed": total_processed,
            "total_candidates": total_comb,
            "sample_fraction": total_processed / total_comb,
            "feasible_count": total_feasible,
            "elapsed": round(total_elapsed, 2),
            "rate": round(rate, 1),
            "num_workers": args.num_workers,
            "seed": args.seed,
            "q_path": args.q_path,
            "solver": "cpu_sparse",
        }
        if best_k is not None:
            out["best_k"] = best_k.tolist()
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
