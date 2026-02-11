"""
Wrapper around the existing Ray-based rank-r solver (parallel_rank_r.py).
It initializes Ray locally, invokes the recursive solver, and shuts down Ray.
"""

import numpy as np
import ray

# Import the existing solver module from the repository
import parallel_rank_r as ray_rankr_mod


def ray_rank_r_solver(
    V_tilde: np.ndarray,
    V: np.ndarray,
    K: int,
    num_cpus: int = None,
    candidates_per_task: int = 1000,
):
    """
    Wrapper for the existing Ray-based distributed solver.

    Args:
        V_tilde: unused by the Ray solver (kept for API symmetry)
        V: (n, r) complex factor matrix
        K: number of phases
        num_cpus: optional cap on Ray CPUs
        candidates_per_task: batch size for Ray tasks

    Returns:
        x_opt: complex vector length n (elements in Omega)
        obj_opt: float objective value
    """
    # Build Q
    Q = V @ V.conj().T

    # Init Ray
    ray.init(num_cpus=num_cpus, ignore_reinit_error=True)

    # Call the recursive solver (this internally uses Ray tasks)
    best_score, best_k, best_z = ray_rankr_mod.process_rankr_recursive(
        V, Q, K=K, candidates_per_task=candidates_per_task
    )

    ray.shutdown()

    return best_z, float(best_score)
