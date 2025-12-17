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
from itertools import product

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


@ray.remote
def process_rank_1_candidate(l, k0, sorted_idx, Q, roots, K):
    k = k0.copy()
    k[sorted_idx[:l]] = (k[sorted_idx[:l]] + 1) % K
    z = roots[k]
    score = np.real(z.conj() @ Q @ z)
    return score

def process_rank_1_parallel(V, Q, K):
    n = V.shape[0]
    log.info(f"Rank 1 subroutine: received eigenvector of length {n}")
    real_q1 = np.real(V).flatten()
    im_q1 = np.imag(V).flatten()
    
    thetas = np.arctan2(im_q1, real_q1)
    thetas = np.where(thetas < 0, thetas + 2*np.pi, thetas)
    b = K * thetas / (2 * np.pi)
    b_floor = np.floor(b).astype(int)
    k0 = b_floor % K

    log.info("Initial assignment k0 computed")

    phi_hat = 0.5 - b + b_floor
    phis = 2 * np.pi * phi_hat / K
    sorted_idx = np.argsort(phis)

    # Use available CPU to determine batch size.
    resources = ray.available_resources()
    if "CPU" in resources:
        batch_size = int(resources["CPU"])
    else:
        batch_size = 1

    batch_size = max(1, batch_size)
    log.info(f"Batch size set to number of available workers: {batch_size}")
    
    # publish the objects through ray's object store
    k0_ref = ray.put(k0)
    Q_ref = ray.put(Q)
    sorted_idx_ref = ray.put(sorted_idx)

    # precompute phase table and publish
    roots = np.exp(2 * np.pi * 1j * np.arange(K) / K)
    roots_ref = ray.put(roots)
    
    best_score = -np.inf
    best_l = 0

    for batch_start in range(0, n + 1, batch_size):
        batch_end = min(batch_start + batch_size, n + 1)
        futures = [
            process_rank_1_candidate.remote(
                l, 
                k0_ref, 
                sorted_idx_ref, 
                Q_ref, 
                roots_ref, 
                K
            )
            for l in range(batch_start, batch_end)
        ]
        batch_scores = ray.get(futures)
        
        # update max
        local_max_idx = int(np.argmax(batch_scores))
        local_max = batch_scores[local_max_idx]
        if local_max > best_score:
            best_score = local_max
            best_l = batch_start + local_max_idx
        log.info(f"Completed batch {batch_start} to {batch_end - 1}")
    log.info(f"Best prefix l = {best_l}")

    # reconstruct the assignment
    best_k = k0.copy()
    best_k[sorted_idx[:best_l]] = (best_k[sorted_idx[:best_l]] + 1) % K

    # compute final z vector
    best_z = roots[best_k]

    return best_score, best_k, best_z

def compute_recovery(z_alg, Q, opt_value):
    alg_value = np.real(z_alg.conj() @ Q @ z_alg)
    return alg_value / opt_value

def parse_args():
    parser = argparse.ArgumentParser(description="Parallel MAX k CUT experiment")
    parser.add_argument("--n", type=int, default=10000, help="Problem size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Compute correctness with opt_K_cut")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to store outputs")
    parser.add_argument("--graph_dir", type=str, default=None,
                        help="Directory containing Q_{n}.npy and V_{n}.npy")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    log.info("Starting MAX 3 CUT experiment")
    ray.init(address="auto", ignore_reinit_error=True)
    log.info("Ray initialized")
    resources = ray.available_resources()
    num_workers = int(resources.get("CPU", 1))
    log.info(f"Detected {num_workers} Ray workers (CPU slots)")

    # load Q, V is not in debug mode
    if not args.debug:
        log.info("Loading Q and V...")
        
        if args.graph_dir is not None:
            q_path = os.path.join(args.graph_dir, f"Q_{args.n}.npy")
            v_path = os.path.join(args.graph_dir, f"V_{args.n}.npy")

            log.info(f"Loading Q from {q_path}")
            log.info(f"Loading V from {v_path}")

            Q = np.load(q_path)
            V = np.load(v_path)
        else:
            Q = generate_Q(0.5, args.n, 'erdos_renyi', seed=args.seed)
            log.info("Random graph Laplacian generated")
            eigvals, eigvecs = np.linalg.eigh(Q)
            _, V = low_rank_matrix(Q, eigvals, eigvecs, r=1)
            log.info("Eigen decomposition complete and top eigenvector extracted")
    else:
        log.info("Generating rank 1 Q, V for debug...")
        Q, V = generate_debug_QV(args.n, 1, seed=args.seed)
    
    log.info("Executing parallel rank 1 algorithm")
    start = time.time()
    best_score, best_k, best_z = process_rank_1_parallel(V[:, 0], Q, K=3)
    elapsed = time.time() - start

    log.info(f"Rank 1 result: score = {best_score}")
    log.info(f"Execution time: {elapsed:.4f} seconds")
    log.info(f"Final partition assignment k:\n{best_k}")
    log.info(f"Computed complex spin vector z:\n{best_z}")
    
    output = {
        "n": args.n,
        "seed": args.seed,
        "best_score": float(best_score),
        "time_seconds": float(elapsed),
        "best_k": best_k.tolist(),
        "best_z_real": np.real(best_z).tolist(),
        "best_z_imag": np.imag(best_z).tolist(),
        "num_workers": num_workers,
    }
    
    if args.debug:
        log.info(f"Computing optimal K-cut...")
        best_score, _ = opt_K_cut(Q)
        log.info(f"Correct score: {best_score}")
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_result_n{args.n}.json"
    path = os.path.join(args.results_dir, filename)
    
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved results to {path}")


if __name__ == "__main__":
    main()
