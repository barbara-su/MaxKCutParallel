import argparse
import os
import numpy as np
import logging
from utils import *
import time
from scipy.linalg import eigh as sp_eigh

# Initialize logging (match your other scripts)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def low_rank_V_from_top_eigs(eigvals, eigvecs, r: int):
    """
    Build V (n×r) from the returned eigenpairs that are ALREADY the top-r ones.

    Assumes eigvals/eigvecs come from a Hermitian eigendecomposition.
    """
    t0 = time.time()

    if r <= 0:
        raise ValueError("r must be positive")
    if eigvals.shape[0] != r:
        raise ValueError(f"Expected exactly r={r} eigenvalues, got {eigvals.shape[0]}")
    if eigvecs.shape[1] != r:
        raise ValueError(f"Expected eigvecs with r={r} columns, got {eigvecs.shape[1]}")

    # eigh returns eigenvalues as real for Hermitian, but keep the guard
    if np.iscomplexobj(eigvals):
        imag_max = np.max(np.abs(eigvals.imag))
        log.info(f"[low_rank_V] eigvals complex: max|imag|={imag_max}")
        if not np.allclose(eigvals.imag, 0):
            raise ValueError("Eigenvalues should be real for a Hermitian matrix")
        ldas = eigvals.real
    else:
        ldas = eigvals

    ldas_min = np.min(ldas)
    ldas_max = np.max(ldas)
    log.info(f"[low_rank_V] top-{r} eigenvalues: max={ldas_max}, min={ldas_min}")

    if ldas_min < -1e-10:
        raise ValueError("Found significant negative eigenvalues in supposedly PSD matrix")
    if ldas_min < 0:
        log.info(f"[low_rank_V] clipping small negative eigenvalues: min={ldas_min}")
        ldas = np.maximum(ldas, 0)

    V = eigvecs * np.sqrt(ldas)  # broadcast scale columns by sqrt(eigs)

    log.info(f"[low_rank_V] V shape={V.shape}, dtype={V.dtype}")
    log.info(f"[low_rank_V] done in {time.time() - t0:.4f}s")
    return V


def top_r_eigh(Q, r: int):
    """
    Compute ONLY the largest-r eigenpairs.

    Uses SciPy (subset_by_index) when available.
    Falls back to NumPy full eigh if SciPy is not installed.
    """
    n = Q.shape[0]
    t0 = time.time()
    
    lo = n - r
    hi = n - 1
    log.info(f"[top_r_eigh] using scipy.linalg.eigh subset_by_index=[{lo}, {hi}]")
    # SciPy returns eigenvalues in ascending order for the selected range.
    evals, evecs = sp_eigh(Q, subset_by_index=(lo, hi))
    log.info(f"[top_r_eigh] scipy eigh done in {time.time() - t0:.4f}s")

    # Reverse to descending so column 0 is the largest eigenvalue
    evals = evals[::-1]
    evecs = evecs[:, ::-1]
    return evals, evecs

def main():
    parser = argparse.ArgumentParser(
        description="Generate low-rank eigenvector matrix V from a saved Q (.npy)."
    )
    parser.add_argument(
        "--q_path",
        type=str,
        required=True,
        help="Path to input Q_{n}.npy",
    )
    parser.add_argument(
        "--v_path",
        type=str,
        required=True,
        help="Path to output V_{n}.npy (will be created/overwritten).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="Rank r for the low-rank eigenvector matrix V.",
    )

    args = parser.parse_args()
    q_path = args.q_path
    v_path = args.v_path
    r = args.rank

    if not os.path.exists(q_path):
        raise FileNotFoundError(f"Q file not found: {q_path}")

    log.info(f"Loading Q from {q_path}")
    Q = np.load(q_path)

    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError(f"Q must be a square 2D array, got shape {Q.shape}")

    n = Q.shape[0]
    log.info(f"Q shape: {Q.shape}")
    log.info(f"Computing V with rank r = {r}")

    # Compute ONLY top-r eigenpairs (SciPy subset if available)
    eigvals_top, eigvecs_top = top_r_eigh(Q, r)

    # Build V from the top-r eigenpairs
    V = low_rank_V_from_top_eigs(eigvals_top, eigvecs_top, r=r)

    log.info(f"V shape: {V.shape}")

    os.makedirs(os.path.dirname(os.path.abspath(v_path)), exist_ok=True)
    np.save(v_path, V)
    log.info(f"Saved V (rank {r}) for n={n} to {v_path}")


if __name__ == "__main__":
    main()
