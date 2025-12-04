import numpy as np
import ray
from utils import *
import cvxpy as cvx
import time
import logging
import warnings
import argparse
import json
from datetime import datetime
import os
import math

# ignore the ray warning
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API"
)

# initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
def process_rank1_block(l_start, l_end, k0, sorted_idx, Q, roots, K):
    """
    Evaluate all prefixes l in [l_start, l_end) inside a single Ray task.
    This reduces the number of tasks and keeps only one copy of k and z per task.
    """
    n = k0.shape[0]
    k = k0.copy()

    # Apply the first l_start flips
    if l_start > 0:
        idx_prefix = sorted_idx[:l_start]
        k[idx_prefix] = (k[idx_prefix] + 1) % K

    best_score = -np.inf
    best_l = l_start

    for l in range(l_start, l_end):
        if l > l_start:
            idx = sorted_idx[l - 1]
            k[idx] = (k[idx] + 1) % K

        z = roots[k]  # complex64 vector
        # Q is float32, z is complex64, result is complex64
        score = float(np.real(z.conj() @ (Q @ z)))
        if score > best_score:
            best_score = score
            best_l = l

    return best_score, best_l


def process_rank1_parallel(V, Q_ref, K, max_parallel_tasks=120):
    """
    Parallel rank 1 algorithm that uses block tasks for memory efficiency.
    V is the top eigenvector (1d complex array).
    Q_ref is a Ray object reference for the Laplacian matrix Q.
    """
    n = V.shape[0]
    log.info(f"Rank 1 subroutine: received eigenvector of length {n}")

    # Use lower precision for large runs
    V = V.astype(np.complex64, copy=False)
    real_q1 = np.real(V).astype(np.float32, copy=False).flatten()
    im_q1 = np.imag(V).astype(np.float32, copy=False).flatten()

    thetas = np.arctan2(im_q1, real_q1)
    thetas = np.where(thetas < 0, thetas + 2 * np.pi, thetas)
    b = K * thetas / (2 * np.pi)
    b_floor = np.floor(b).astype(np.int32)
    k0 = b_floor % K

    log.info("Initial assignment k0 computed")

    phi_hat = 0.5 - b + b_floor
    phis = 2 * np.pi * phi_hat / K
    sorted_idx = np.argsort(phis).astype(np.int64)

    # Put small arrays into the Ray object store
    k0_ref = ray.put(k0)
    sorted_idx_ref = ray.put(sorted_idx)

    roots = np.exp(2 * np.pi * 1j * np.arange(K) / K).astype(np.complex64)
    roots_ref = ray.put(roots)

    num_l = n + 1  # l ranges from 0 to n inclusive
    num_blocks = min(num_l, max_parallel_tasks)
    block_size = math.ceil(num_l / num_blocks)

    log.info(f"Using {num_blocks} Ray tasks with block_size {block_size}")

    futures = []
    for block_idx in range(num_blocks):
        l_start = block_idx * block_size
        l_end = min(num_l, (block_idx + 1) * block_size)
        if l_start >= l_end:
            continue
        futures.append(
            process_rank1_block.remote(
                l_start,
                l_end,
                k0_ref,
                sorted_idx_ref,
                Q_ref,
                roots_ref,
                K,
            )
        )

    results = ray.get(futures)
    best_score = -np.inf
    best_l = 0
    for score, l in results:
        if score > best_score:
            best_score = score
            best_l = l

    log.info(f"Best prefix l = {best_l}")

    # Reconstruct the best assignment on the driver
    k = k0.copy()
    if best_l > 0:
        k[sorted_idx[:best_l]] = (k[sorted_idx[:best_l]] + 1) % K
    z = roots[k]

    return best_score, k, z


def compute_recovery(z_alg, Q, opt_value):
    alg_value = np.real(z_alg.conj() @ Q @ z_alg)
    return alg_value / opt_value


def solve_sdp_optimal(Q):
    n = Q.shape[0]
    log.info(f"Solving SDP relaxation for n={n}")
    X = cvx.Variable((n, n), PSD=True)
    obj = cvx.Maximize(cvx.sum(cvx.multiply(Q, X)))
    constraints = [cvx.diag(X) == 1]
    prob = cvx.Problem(obj, constraints)
    prob.solve(solver=cvx.SCS, verbose=False)
    log.info("SDP optimal value computed")
    return prob.value


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel MAX k CUT experiment")
    parser.add_argument("--n", type=int, default=10000, help="Problem size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--compute_recovery", action="store_true", help="Compute recovery ratio (for small n)")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to store outputs")
    parser.add_argument("--graph_dir", type=str, default=None,
                        help="Directory containing Q_{n}.npy and V_{n}.npy")
    parser.add_argument("--max_parallel_tasks", type=int, default=120,
                        help="Maximum number of Ray tasks in the rank 1 search")
    parser.add_argument("--save_assignment", action="store_true",
                        help="Save best_k and best_z to .npy files")
    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    log.info("Starting MAX 3 CUT experiment")
    ray.init(address="auto", ignore_reinit_error=True)
    log.info("Ray initialized")

    log.info("Loading Q and V...")

    Q_for_sdp = None

    if args.graph_dir is not None:
        q_path = os.path.join(args.graph_dir, f"Q_{args.n}.npy")
        v_path = os.path.join(args.graph_dir, f"V_{args.n}.npy")

        log.info(f"Loading Q from {q_path}")
        log.info(f"Loading V from {v_path}")

        Q = np.load(q_path)
        V_full = np.load(v_path)
    else:
        Q = generate_Q(0.5, args.n, "erdos_renyi", seed=args.seed)
        log.info("Random graph Laplacian generated")
        eigvals, eigvecs = np.linalg.eigh(Q.astype(np.float64))
        _, V_full = low_rank_matrix(Q, eigvals, eigvecs, r=1)

    # For the algorithm, use float32 for Q and complex64 for V
    Q = Q.astype(np.float32, copy=False)
    V_full = V_full.astype(np.complex64, copy=False)

    # Keep a high precision copy of Q only if we truly need the SDP and n is small
    if args.compute_recovery and args.n <= 10:
        Q_for_sdp = Q.astype(np.float64, copy=False)

    # Put Q into Ray object store once and free the driver copy
    Q_ref = ray.put(Q)
    del Q

    log.info("Eigen decomposition complete and top eigenvector extracted")

    log.info("Executing parallel rank 1 algorithm")
    start = time.time()
    best_score, best_k, best_z = process_rank1_parallel(
        V_full[:, 0],
        Q_ref,
        K=3,
        max_parallel_tasks=args.max_parallel_tasks,
    )
    elapsed = time.time() - start

    # Log only summaries, not full vectors
    log.info(f"Rank 1 result: score = {best_score}")
    log.info(f"Execution time: {elapsed:.4f} seconds")
    log.info(f"k counts: {np.bincount(best_k, minlength=3)}")
    log.info(f"First 10 entries of k: {best_k[:10]}")

    output = {
        "n": args.n,
        "seed": args.seed,
        "best_score": float(best_score),
        "time_seconds": float(elapsed),
    }

    # Optionally compute recovery ratio, only if Q_for_sdp is available
    if args.compute_recovery and Q_for_sdp is not None:
        opt_value = solve_sdp_optimal(Q_for_sdp)
        recovery = compute_recovery(best_z.astype(np.complex128), Q_for_sdp, opt_value)
        log.info(f"Recovery ratio: {recovery}")
        output["recovery_ratio"] = float(recovery)

    # Optionally store the full assignment and z as .npy files
    # if args.save_assignment:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    k_filename = f"{timestamp}_k_n{args.n}.npy"
    z_filename = f"{timestamp}_z_n{args.n}.npy"

    k_path = os.path.join(args.results_dir, k_filename)
    z_path = os.path.join(args.results_dir, z_filename)

    np.save(k_path, best_k)
    np.save(z_path, best_z)

    output["best_k_path"] = k_filename
    output["best_z_path"] = z_filename

    # Store scalar stats in JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"{timestamp}_result_n{args.n}.json"
    json_path = os.path.join(args.results_dir, json_filename)
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved results summary to {json_path}")


if __name__ == "__main__":
    main()
