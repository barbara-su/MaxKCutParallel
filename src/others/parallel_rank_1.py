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

# Ignore the Ray warning
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API"
)

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)


@ray.remote
def process_rank_1_batch(l_values, k0, sorted_idx, Q, roots, K, batch_id):
    """
    One Ray task: evaluate a batch of candidate prefix lengths (l_values).
    Returns the best (score, l) within this task.
    """
    best_score = float("-inf")
    best_l = None

    for l in l_values:
        k = k0.copy()
        if l > 0:
            idx = sorted_idx[:l]
            k[idx] = (k[idx] + 1) % K
        z = roots[k]
        score = np.einsum('i,ij,j->', z.conj(), Q, z).real
        if score > best_score:
            best_score = score
            best_l = int(l)

    return best_score, best_l, batch_id


def process_rank_1_parallel(V, Q, K=3, candidates_per_task=10):
    n = V.shape[0]
    log.info(f"Rank 1 subroutine: received eigenvector of length {n}")

    if candidates_per_task <= 0:
        raise ValueError("--candidates_per_task must be a positive integer")

    real_q1 = np.real(V).flatten()
    im_q1 = np.imag(V).flatten()

    thetas = np.arctan2(im_q1, real_q1)
    thetas = np.where(thetas < 0, thetas + 2 * np.pi, thetas)

    b = K * thetas / (2 * np.pi)
    b_floor = np.floor(b).astype(int)
    k0 = b_floor % K
    log.info("Initial assignment k0 computed")

    phi_hat = 0.5 - b + b_floor
    phis = 2 * np.pi * phi_hat / K
    sorted_idx = np.argsort(phis)

    num_candidates = n + 1
    total_tasks = (num_candidates + candidates_per_task - 1) // candidates_per_task

    resources = ray.available_resources()
    num_cpus = max(1, int(resources.get("CPU", 1)))

    log.info(f"Total candidates (prefix lengths): {num_candidates}")
    log.info(f"Detected CPUs: {num_cpus}")
    log.info(f"candidates_per_task: {candidates_per_task}")
    log.info(f"Total Ray tasks to submit (ceil(n/candidates_per_task)): {total_tasks}")

    k0_ref = ray.put(k0)
    Q_ref = ray.put(Q)
    sorted_idx_ref = ray.put(sorted_idx)

    roots = np.exp(2 * np.pi * 1j * np.arange(K) / K)
    roots_ref = ray.put(roots)

    def batched_l_values():
        iterator = iter(range(num_candidates))
        while True:
            batch = list(itertools.islice(iterator, candidates_per_task))
            if not batch:
                break
            yield batch

    start_time = time.time()

    # --- NEW: cap in-flight tasks ---
    max_in_flight = max(2 * num_cpus, 1)
    in_flight = []
    submitted = 0
    completed = 0

    best_score = float("-inf")
    best_l = 0

    batch_id = 0
    for batch in batched_l_values():
        # submit
        in_flight.append(
            process_rank_1_batch.remote(
                batch, k0_ref, sorted_idx_ref, Q_ref, roots_ref, K, batch_id
            )
        )
        submitted += 1
        batch_id += 1

        # drain when at cap
        if len(in_flight) >= max_in_flight:
            done, in_flight = ray.wait(in_flight, num_returns=1)
            batch_score, batch_best_l, b_id = ray.get(done[0])
            completed += 1

            log.info(f"Completed task {b_id}")
            if batch_best_l is not None and batch_score > best_score:
                best_score = batch_score
                best_l = batch_best_l
                log.info(f"New best from task {b_id}: score={best_score} (l={best_l})")

    # drain remaining
    while in_flight:
        done, in_flight = ray.wait(in_flight, num_returns=1)
        batch_score, batch_best_l, b_id = ray.get(done[0])
        completed += 1

        log.info(f"Completed task {b_id}")
        if batch_best_l is not None and batch_score > best_score:
            best_score = batch_score
            best_l = batch_best_l
            log.info(f"New best from task {b_id}: score={best_score} (l={best_l})")

    elapsed = time.time() - start_time
    log.info(f"Rank 1 search complete in {elapsed:.4f} seconds")
    log.info(f"Submitted={submitted}, completed={completed}, max_in_flight={max_in_flight}")
    log.info(f"Best prefix l = {best_l}")

    best_k = k0.copy()
    if best_l > 0:
        best_k[sorted_idx[:best_l]] = (best_k[sorted_idx[:best_l]] + 1) % K
    best_z = roots[best_k]

    return best_score, best_k, best_z, best_l


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel MAX k CUT experiment (rank 1)")
    parser.add_argument("--n", type=int, default=10000, help="Problem size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--precision", type=int, default=64, choices=[16, 32, 64],
                        help="Numeric precision: 16, 32, or 64 (default: 64)")
    parser.add_argument("--candidates_per_task", type=int, default=10,
                        help="How many candidates (l values) each Ray task evaluates serially (default: 10).")
    parser.add_argument("--debug", action="store_true", help="Compute correctness with opt_K_cut")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to store outputs")
    parser.add_argument("--graph_dir", type=str, default=None,
                        help="Directory containing Q_{n}.npy and V_{n}.npy")
    return parser.parse_args()


def main():
    args = parse_args()

    float_dtype, complex_dtype = set_numpy_precision(args.precision)
    log.info(f"Using precision={args.precision} -> float={float_dtype.__name__}, complex={complex_dtype.__name__}")

    np.random.seed(args.seed)

    log.info("Starting MAX 3 CUT experiment (rank 1)")
    ray.init(address="auto", ignore_reinit_error=True)
    log.info("Ray initialized")

    resources = ray.available_resources()
    num_workers = int(resources.get("CPU", 1))
    log.info(f"Detected {num_workers} Ray workers (CPU slots)")

    # Load Q and V
    if not args.debug:
        log.info("Loading Q and V...")
        if args.graph_dir is not None:
            q_path = os.path.join(args.graph_dir, f"Q_{args.n}.npy")
            v_path = os.path.join(args.graph_dir, f"V_{args.n}.npy")

            log.info(f"Loading Q from {q_path}")
            log.info(f"Loading V from {v_path}")

            Q = np.load(q_path).astype(float_dtype, copy=False)
            V = np.asarray(np.load(v_path), dtype=complex_dtype)
        else:
            Q = np.asarray(generate_Q(0.5, args.n, "erdos_renyi", seed=args.seed), dtype=float_dtype)
            log.info("Random graph Laplacian generated")
            eigvals, eigvecs = np.linalg.eigh(Q.astype(np.float64, copy=False))
            _, V = low_rank_matrix(Q, eigvals, eigvecs, r=1)
            V = np.asarray(V, dtype=complex_dtype)
            log.info("Eigen decomposition complete and top eigenvector extracted")
    else:
        log.info("Generating debug low rank Q, V (rank 1)")
        Q, V = generate_debug_QV(n=args.n, rank=1, seed=args.seed)
        Q = np.asarray(Q, dtype=float_dtype)
        V = np.asarray(V, dtype=complex_dtype)

    log.info("Executing parallel rank 1 algorithm")
    start = time.time()

    best_score, best_k, best_z, best_l = process_rank_1_parallel(
        V[:, 0], Q, K=3, candidates_per_task=args.candidates_per_task
    )

    elapsed = time.time() - start

    log.info(f"Rank 1 result: score = {best_score}")
    log.info(f"Execution time: {elapsed:.4f} seconds")
    log.info(f"candidates_per_task={args.candidates_per_task}")
    log.info(f"best_l={best_l}")

    output = {
        "n": args.n,
        "seed": args.seed,
        "rank": 1,
        "precision": args.precision,
        "candidates_per_task": args.candidates_per_task,
        "best_l": best_l,
        "best_score": best_score,
        "time_seconds": elapsed,
        "best_k": best_k.tolist(),
        "best_z_real": np.real(best_z).tolist(),
        "best_z_imag": np.imag(best_z).tolist(),
        "num_workers": num_workers,
    }

    if args.debug:
        log.info("Computing optimal K-cut...")
        opt_score, _ = opt_K_cut(Q.astype(np.float64, copy=False))
        log.info(f"Correct score: {opt_score}")

    os.makedirs(args.results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_result_n{args.n}_r1_p{args.precision}_cpt{args.candidates_per_task}.json"
    path = os.path.join(args.results_dir, filename)

    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved results to {path}")


if __name__ == "__main__":
    main()
