"""
GPU-accelerated rank-r discrete optimization solver using CuPy.

This implementation mirrors the combinatorial enumeration idea:
- For each combination of (2r-1) rows of V_tilde, compute a null-space
  direction via SVD (small matrix).
- Quantize to the nearest K-th root of unity.
- Evaluate the objective ||V^H x||^2 on GPU.
- Track the best candidate.

Notes:
- This follows the structure of the provided Ray implementation but runs on a
  single GPU. It includes the "used-vertex" refinement step (fixed-angle
  intersection) for better fidelity with the Ray code.
- Intended for moderate problem sizes; extremely large combinatorial counts
  are infeasible to enumerate directly on a single device.
"""

import itertools
from math import comb
from typing import Tuple, Optional, Iterable, List

import cupy as cp
import numpy as np


def _omega(K: int, dtype=cp.complex128):
    return cp.exp(2j * cp.pi * cp.arange(K, dtype=dtype) / K)


def determine_phi_sign_c_cp(c_tilde: cp.ndarray):
    """
    CuPy version of determine_phi_sign_c.
    c_tilde: shape (2r,)
    Returns: phi (shape 2r-1), sign_c (scalar float in {-1,1})
    """
    D = c_tilde.shape[0]
    phi = cp.zeros(D - 1, dtype=cp.float64)

    for phi_ind in range(D - 1):
        if phi_ind > 0:
            cos_values = cp.cos(phi[:phi_ind])
            if cp.any(cp.abs(cos_values) < 1e-10):
                phi[phi_ind] = 0.0
                continue
            prod_cos = cp.prod(cos_values)
        else:
            prod_cos = 1.0

        if abs(prod_cos) > 1e-10:
            arg = c_tilde[phi_ind] / prod_cos
        else:
            arg = 0.0

        arg = cp.clip(arg, -1.0, 1.0)
        phi[phi_ind] = cp.arcsin(arg)

    if phi[D - 2] == 0 or c_tilde[D - 2] == 0:
        sign_c = 1.0
    else:
        if abs(cp.cos(phi[D - 2])) < 1e-10:
            sign_c = 1.0
        else:
            sign_c = cp.sign(cp.tan(phi[D - 2]) * c_tilde[D - 2] * c_tilde[D - 1])

    return phi, float(sign_c.get())


def convert_ctilde_to_complex_cp(c_tilde: cp.ndarray, r: int) -> cp.ndarray:
    c = cp.zeros(r, dtype=cp.complex128)
    for j in range(r):
        if 2 * j + 1 < c_tilde.shape[0]:
            c[j] = c_tilde[2 * j] + 1j * c_tilde[2 * j + 1]
    return c


def find_nullspace_vec(VI: cp.ndarray) -> cp.ndarray:
    """
    Returns a normalized null-space vector of VI (small matrix).
    Uses SVD: the right singular vector corresponding to the smallest singular value.
    """
    # VI shape: (2r-1, 2r)
    _, _, vh = cp.linalg.svd(VI, full_matrices=False)
    c_tilde = vh[-1]
    norm = cp.linalg.norm(c_tilde)
    if norm == 0:
        raise cp.linalg.LinAlgError("Nullspace vector has zero norm")
    return c_tilde / norm


def find_intersection_fixed_angle_cp(VI_minus: cp.ndarray, r: int, K: int) -> cp.ndarray:
    """
    Solve for c_tilde with last angle fixed at pi/K.
    VI_minus: shape (2r-2, 2r)
    """
    A = VI_minus[:, : 2 * r - 2]
    b = -VI_minus[:, 2 * r - 2 :] @ cp.array([cp.sin(cp.pi / K), cp.cos(cp.pi / K)], dtype=VI_minus.dtype)

    # Least squares solve
    phi_reduced, _, _, _ = cp.linalg.lstsq(A, b, rcond=None)
    # construct c_tilde
    c_tilde = cp.zeros(2 * r, dtype=VI_minus.dtype)
    c_tilde[0] = cp.sin(phi_reduced[0])

    prod_cos = cp.cos(phi_reduced[0])
    for i in range(1, phi_reduced.shape[0]):
        c_tilde[i] = prod_cos * cp.sin(phi_reduced[i])
        prod_cos *= cp.cos(phi_reduced[i])

    c_tilde[2 * r - 2] = prod_cos * cp.sin(cp.pi / K)
    c_tilde[2 * r - 1] = prod_cos * cp.cos(cp.pi / K)
    return c_tilde


def _quantize_vector_to_omega(v: cp.ndarray, omega: cp.ndarray) -> cp.ndarray:
    """
    For each element of v, pick omega_k maximizing Re(conj(omega_k)*v_i).
    v: shape (n,)
    omega: shape (K,)
    """
    metrics = (omega.conj()[None, :] * v[:, None]).real  # (n, K)
    idx = cp.argmax(metrics, axis=1)
    return omega[idx]


def _batched_combinations(m: int, comb_size: int, batch_size: int) -> Iterable[List[tuple]]:
    iterator = itertools.combinations(range(m), comb_size)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def gpu_rank_r_solver(
    V_tilde: np.ndarray,
    V: np.ndarray,
    K: int,
    batch_size: int = 10000,
) -> Tuple[np.ndarray, float]:
    """
    GPU-accelerated rank-r discrete optimization solver.

    Args:
        V_tilde: Normalized factor matrix (m x 2r) real-valued (from compute_vtilde or row-normalized)
        V: Original factor matrix (n x r) complex
        K: Number of discrete phases (roots of unity)
        batch_size: Number of combinations to process per batch on GPU

    Returns:
        x_opt: Optimal solution vector (n,), complex with elements in Omega
        obj_opt: Optimal objective value (real)
    """
    # Move to GPU
    V_tilde_gpu = cp.asarray(V_tilde)
    V_gpu = cp.asarray(V)
    n, r = V.shape
    m = V_tilde.shape[0]
    comb_size = 2 * r - 1
    if comb_size > m:
        raise ValueError("Combination size exceeds available rows in V_tilde")

    total_combos = comb(m, comb_size)
    omega = _omega(K, dtype=cp.complex128)

    best_score = -cp.inf
    best_candidate = None

    # Precompute mapping for used vertices assuming ordering v_tilde_idx = v_idx * K + rot
    def v_idx_from_row(idx: int) -> int:
        return idx // K

    for comb_batch in _batched_combinations(m, comb_size, batch_size):
        # process each combination sequentially (small comb_size), GPU heavy ops inside
        for combo in comb_batch:
            try:
                I = cp.array(combo, dtype=cp.int32)
                VI = V_tilde_gpu[I]  # (comb_size, 2r)

                c_tilde = find_nullspace_vec(VI)
                phi, sign_c = determine_phi_sign_c_cp(c_tilde)

                # decision region check
                if not (-cp.pi / K < phi[2 * r - 2] <= cp.pi / K):
                    continue

                c_tilde = c_tilde * sign_c
                c = convert_ctilde_to_complex_cp(c_tilde, r)

                # vertices used in selected hyperplanes
                v_used = set(int(v_idx_from_row(int(ii))) for ii in combo)

                # base assignment for all vertices
                v_c_all = V_gpu @ c  # (n,)
                candidate = _quantize_vector_to_omega(v_c_all, omega)

                # refine vertices in hyperplanes
                for v_idx in v_used:
                    # find which rows in combo belong to this vertex
                    rows_for_v = [i for i in combo if v_idx_from_row(i) == v_idx]
                    assigned = False
                    for vtilde_idx in rows_for_v:
                        pos = combo.index(vtilde_idx)
                        VI_minus = cp.delete(VI, pos, axis=0)
                        try:
                            new_c_tilde = find_intersection_fixed_angle_cp(VI_minus, r, K)
                            new_c = convert_ctilde_to_complex_cp(new_c_tilde, r)
                            v_c = V_gpu[v_idx] @ new_c
                            cand_val = _quantize_vector_to_omega(cp.asarray([v_c]), omega)[0]
                            candidate[v_idx] = cand_val
                            assigned = True
                            break
                        except cp.linalg.LinAlgError:
                            continue
                    if (not assigned):
                        # fallback already assigned by base
                        pass

                # objective: ||V^H x||^2 = sum |(x^H V)_j|^2
                t = candidate.conj() @ V_gpu  # (r,)
                score = cp.sum(cp.abs(t) ** 2).real

                if score > best_score:
                    best_score = score
                    best_candidate = candidate.copy()

            except cp.linalg.LinAlgError:
                continue

    if best_candidate is None:
        raise RuntimeError("No feasible candidate found")

    x_opt = cp.asnumpy(best_candidate)
    obj_opt = float(best_score.get())
    return x_opt, obj_opt


# Convenience CLI (optional)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPU rank-r solver demo")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--r", type=int, default=2)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4096)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    V = (rng.standard_normal((args.n, args.r)) + 1j * rng.standard_normal((args.n, args.r))) / np.sqrt(2)
    row_norms = np.linalg.norm(V, axis=1, keepdims=True)
    V_tilde = V / row_norms  # simple normalization

    x_opt, obj_opt = gpu_rank_r_solver(V_tilde, V, args.K, batch_size=args.batch_size)
    print("Objective:", obj_opt)