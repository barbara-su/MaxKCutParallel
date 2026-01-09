#!/usr/bin/env python3
"""
Generate a random Gaussian-PSD matrix Q ("random_gaus" mode) and its rank-1
low-rank factor V using your existing low_rank_matrix() utility.

This mirrors your ER script: it saves
  - Q_{n}.npy
  - V_{n}.npy   (shape (n, r), r default 1)

Usage:
  python gen_random_gaus_QV.py --n 1000 --sigma 1.0 --seed 42 --rank 1 --out_dir graphs/graphs_random_gaus
"""

import argparse
import os
import numpy as np
from utils import low_rank_matrix  # assumes this exists and matches your project


def generate_random_gaus_Q(n: int, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Re = rng.normal(0.0, sigma, size=(n, n))
    Im = rng.normal(0.0, sigma, size=(n, n))
    A = Re + 1j * Im
    Q = A @ A.conj().T  # Hermitian PSD
    return Q


def main():
    parser = argparse.ArgumentParser(
        description="Generate random Gaussian-PSD Q and low-rank V via eigen-decomposition."
    )
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="graphs/graphs_random_gaus")
    parser.add_argument("--dtype", type=str, default="complex64", choices=["complex64", "complex128"])
    args = parser.parse_args()

    n = args.n
    sigma = args.sigma
    seed = args.seed
    r = args.rank
    out_dir = args.out_dir

    print(f"Generating random_gaus Q with n={n}, sigma={sigma}, seed={seed}, rank={r}")
    os.makedirs(out_dir, exist_ok=True)

    # Generate Q
    Q = generate_random_gaus_Q(n, sigma, seed)

    # Choose dtype
    if args.dtype == "complex64":
        Q = Q.astype(np.complex64, copy=False)
        work_dtype = np.complex128  # for stable eigh
    else:
        Q = Q.astype(np.complex128, copy=False)
        work_dtype = np.complex128

    # Eigen decomposition (use complex128 for numerical stability)
    eigvals, eigvecs = np.linalg.eigh(Q.astype(work_dtype, copy=False))

    # Low-rank factor (your project’s convention)
    _, V = low_rank_matrix(Q, eigvals, eigvecs, r=r)

    # Cast V consistently
    if args.dtype == "complex64":
        V = np.asarray(V, dtype=np.complex64)
    else:
        V = np.asarray(V, dtype=np.complex128)

    print(f"Q shape: {Q.shape}, dtype: {Q.dtype}")
    print(f"V shape: {V.shape}, dtype: {V.dtype}")

    # Save in the same naming convention you use elsewhere
    q_path = os.path.join(out_dir, f"Q_{n}.npy")
    v_path = os.path.join(out_dir, f"V_{n}.npy")
    np.save(q_path, Q)
    np.save(v_path, V)

    print(f"Saved Q to {q_path}")
    print(f"Saved V (rank {r}) to {v_path}")


if __name__ == "__main__":
    main()
