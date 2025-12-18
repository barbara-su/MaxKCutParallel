import numpy as np
import ray
from utils import *
import time
import logging
import warnings
import argparse
import json
from datetime import datetime
import os
import itertools

from scipy.sparse import csc_matrix, issparse

# Ignore Ray warning
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API"
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)


@ray.remote
def process_rank_1_batch(l_values, k0, sorted_idx, Q, roots, K, batch_id):
    """
    One Ray task.
    Evaluates a batch of prefix lengths l_values and returns
    the best (score, l) in this batch.
    """
    best_score = float("-inf")
    best_l = None

    k = k0.copy()

    for l in l_values:
        if l > 0:
            idx = sorted_idx[:l]
            k[idx] = (k[idx] + 1) % K

        z = roots[k]

        # Sparse quadratic form: z* Q z
        Qz = Q @ z
        score = np.vdot(z, Qz).real

        if score > best_score:
            best_score = score
            best_l = l

        # undo mutation
        if l > 0:
            k[idx] = (k[idx] - 1) % K

    return best_score, best_l, batch_id


def process_rank_1_parallel(V, Q, K=3, candidates_per_task=10):
    """
    Parallel rank-1 sweep over prefix lengths l = 0..n.
    """
    n = V.shape[0]
    log.info(f"Rank 1 subroutine: received eigenvector of length {n}")

    if candidates_per_task <= 0:
        raise ValueError("--candidates_per_task must be positive")

    # Initial assignment
    real_q1 = np.real(V).flatten()
    im_q1 = np.imag(V).flatten()

    thetas = np.arctan2(im_q1, real_q1)
    thetas = np.where(thetas < 0, thetas + 2 * np.pi, thetas)

    b = K * thetas / (2 * np.pi)
    b_floor = np.floor(b).astype(int)
    k0 = b_floor % K

    phi_hat = 0.5 - b + b_floor
    phis = 2 * np.pi * phi_hat / K
    sorted_idx = np.argsort(phis)

    num_candidates = n + 1
    total_tasks = (num_candidates + candidates_per_task - 1) // candidates_per_task

    resources = ray.available_resources()
    num_cpus = max(1, int(resources.get("CPU", 1)))

    log.info(f"Total candidates: {num_candidates}")
    log.info(f"Detected CPUs: {num_cpus}")
    log.info(f"Candidates per task: {candidates_per_task}")
    log.info(f"Total Ray tasks: {total_tasks}")

    # Put shared objects in Ray object store
    k0_ref = ray.put(k0)
    Q_ref = ray.put(Q)
    sorted_idx_ref = ray.put(sorted_idx)

    roots = np.exp(2 * np.pi * 1j * np.arange(K) / K)
    roots_ref = ray.put(roots)

    def batched_l_values():
        it = iter(range(num_candidates))
        while True:
            batch = list(itertools.islice(it, candidates_per_task))
            if not batch:
                break
            yield batch

    start_time = time.time()

    futures = [
        process_rank_1_batch.remote(
            batch, k0_ref, sorted_idx_ref, Q_ref, roots_ref, K, batch_id
        )
        for batch_id, batch in enumerate(batched_l_values())
    ]

    log.info(f"Submitted {len(futures)} tasks")

    best_score = float("-inf")
    best_l = 0

    for fut in futures:
        batch_score, batch_best_l, b_id = ray.get(fut)
        log.info(f"Completed task {b_id}")
        if batch_best_l is not None and batch_score > best_score:
            best_score = batch_score
            best_l = batch_best_l
            log.info(f"New best: score={best_score}, l={best_l}")

    elapsed = time.time() - start_time
    log.info(f"Rank 1 search completed in {elapsed:.4f} seconds")

    # Reconstruct best solution
    best_k = k0.copy()
    if best_l > 0:
        best_k[sorted_idx[:best_l]] = (best_k[sorted_idx[:best_l]] + 1) % K

    best_z = roots[best_k]

    return best_score, best_k, best_z, best_l


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel MAX-k-CUT rank-1")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", type=int, default=64, choices=[16, 32, 64])
    parser.add_argument("--candidates_per_task", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--graph_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    float_dtype, complex_dtype = set_numpy_precision(args.precision)
    np.random.seed(args.seed)

    ray.init(address="auto", ignore_reinit_error=True)

    if not args.debug:
        if args.graph_dir is not None:
            Q = np.load(os.path.join(args.graph_dir, f"Q_{args.n}.npy"))
            V = np.load(os.path.join(args.graph_dir, f"V_{args.n}.npy"))
        else:
            Q = generate_Q(0.5, args.n, "erdos_renyi", seed=args.seed)
            eigvals, eigvecs = np.linalg.eigh(Q.astype(np.float64))
            _, V = low_rank_matrix(Q, eigvals, eigvecs, r=1)
    else:
        Q, V = generate_debug_QV(n=args.n, rank=1, seed=args.seed)

    # Convert Q to sparse CSC once
    if not issparse(Q):
        Q = csc_matrix(Q)
    Q = Q.astype(float_dtype)
    Q.sort_indices()
    Q.eliminate_zeros()

    V = np.asarray(V, dtype=complex_dtype)

    best_score, best_k, best_z, best_l = process_rank_1_parallel(
        V[:, 0], Q, K=3, candidates_per_task=args.candidates_per_task
    )

    output = {
        "n": args.n,
        "seed": args.seed,
        "rank": 1,
        "precision": args.precision,
        "candidates_per_task": args.candidates_per_task,
        "best_l": best_l,
        "best_score": best_score,
    }

    os.makedirs(args.results_dir, exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
    with open(os.path.join(args.results_dir, filename), "w") as f:
        json.dump(output, f, indent=2)

    log.info("Done")


if __name__ == "__main__":
    main()
