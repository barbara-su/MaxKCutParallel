import argparse
import os
import numpy as np
import logging
import time
from scipy.linalg import eigh as sp_eigh


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def low_rank_V_from_top_eigs(eigvals, eigvecs, r: int, return_timing: bool = False):
    """
    Build V (n×r) from the provided top-r eigenpairs.

    Returns:
      - V                          if return_timing is False
      - (V, construct_V_time_sec)  if return_timing is True
    """
    t0 = time.perf_counter()

    if r <= 0:
        raise ValueError("r must be positive")
    if eigvals.shape[0] != r:
        raise ValueError(f"Expected exactly r={r} eigenvalues, got {eigvals.shape[0]}")
    if eigvecs.shape[1] != r:
        raise ValueError(f"Expected eigvecs with r={r} columns, got {eigvecs.shape[1]}")

    if np.iscomplexobj(eigvals):
        imag_max = np.max(np.abs(eigvals.imag))
        log.info(f"[low_rank_V] eigvals complex: max|imag|={imag_max}")
        if not np.allclose(eigvals.imag, 0):
            raise ValueError("Eigenvalues should be real for a Hermitian matrix")
        ldas = eigvals.real
    else:
        ldas = eigvals

    ldas_min = float(np.min(ldas))
    ldas_max = float(np.max(ldas))

    if ldas_min < -1e-10:
        raise ValueError("Found significant negative eigenvalues in supposedly PSD matrix")
    if ldas_min < 0:
        ldas = np.maximum(ldas, 0)

    # Column-wise scaling by sqrt(eigs)
    V = eigvecs * np.sqrt(ldas)

    t_construct = time.perf_counter() - t0

    # Keep logs light and stable
    log.info(f"[low_rank_V] top-{r} eigs: max={ldas_max:.6g}, min={ldas_min:.6g} | construct_V={t_construct:.4f}s")

    if return_timing:
        return V, t_construct
    return V


def top_r_eigh(Q, r: int, return_timing: bool = False):
    """
    Compute only the largest-r eigenpairs via SciPy subset_by_index.

    Returns:
      - (evals, evecs)                         if return_timing is False
      - (evals, evecs, compute_eigs_time_sec)  if return_timing is True
    """
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError(f"Q must be square. Got {Q.shape}")

    n = Q.shape[0]
    if r <= 0 or r > n:
        raise ValueError(f"r must satisfy 1 <= r <= n. Got r={r}, n={n}")

    t0 = time.perf_counter()

    lo = n - r
    hi = n - 1
    evals, evecs = sp_eigh(Q, subset_by_index=(lo, hi))

    # Convert to descending order
    evals = evals[::-1]
    evecs = evecs[:, ::-1]

    t_eigs = time.perf_counter() - t0
    log.info(f"[top_r_eigh] n={n} r={r} | compute_eigs={t_eigs:.4f}s")

    if return_timing:
        return evals, evecs, t_eigs
    return evals, evecs


def gen_V_given_Q(Q, r: int, return_timing: bool = False):
    """
    Generate V from Q using top-r eigenpairs.

    Returns:
      - V if return_timing is False
      - (V, timing_dict) if return_timing is True

    timing_dict keys:
      - compute_eigenvectors_time_s
      - construct_V_time_s
    """
    evals, evecs, t_eigs = top_r_eigh(Q, r, return_timing=True)
    V, t_construct = low_rank_V_from_top_eigs(evals, evecs, r=r, return_timing=True)

    if return_timing:
        return V, {
            "compute_eigenvectors_time_s": t_eigs,
            "construct_V_time_s": t_construct,
        }
    return V


def gen_Q_hat_given_V(V: np.ndarray) -> np.ndarray:
    if V.ndim != 2:
        raise ValueError(f"V must be 2D (n,r). Got shape {V.shape}")

    if np.iscomplexobj(V):
        Q_hat = V @ V.conj().T
        if np.allclose(np.imag(Q_hat), 0, atol=1e-10):
            Q_hat = np.real(Q_hat)
    else:
        Q_hat = V @ V.T
    return Q_hat


def main():
    parser = argparse.ArgumentParser(
        description="Generate low-rank eigenvector matrix V from a saved Q (.npy)."
    )
    parser.add_argument("--q_path", type=str, required=True)
    parser.add_argument("--v_path", type=str, required=True)
    parser.add_argument("--rank", type=int, default=1)
    args = parser.parse_args()

    if not os.path.exists(args.q_path):
        raise FileNotFoundError(f"Q file not found: {args.q_path}")

    log.info(f"Loading Q from {args.q_path}")
    Q = np.load(args.q_path)

    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError(f"Q must be a square 2D array, got shape {Q.shape}")

    V = gen_V_given_Q(Q, args.rank, return_timing=False)

    os.makedirs(os.path.dirname(os.path.abspath(args.v_path)), exist_ok=True)
    np.save(args.v_path, V)
    log.info(f"Saved V (rank {args.rank}) to {args.v_path}")


if __name__ == "__main__":
    main()
