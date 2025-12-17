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
    Process a batch of prefix lengths l for the rank-1 algorithm.

    For each l in l_values:
      - flip the first l indices (in sorted order) by +1 mod K
      - evaluate z^* Q z
    Return the best score and best l in this batch.
    """
    best_score = float("-inf")
    best_l = None

    for l in l_values:
        k = k0.copy()
        if l > 0:
            idx = sorted_idx[:l]
            k[idx] = (k[idx] + 1) % K
        z = roots[k]
        score = float(np.real(z.conj() @ Q @ z))
        if score > best_score:
            best_score = score
            best_l = int(l)

    return best_score, best_l, batch_id


def process_rank_1_parallel(V, Q, K=3):
    """
    Parallel rank-1 max-k-cut sweep over prefix lengths l = 0..n.

    Parallel pattern matches your rank-r logic:
      - stream batches
      - submit all Ray tasks
      - collect and track the global best
    """
    n = V.shape[0]
    log.info(f"Rank 1 subroutine: received eigenvector of length {n}")

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
    resources = ray.available_resources()
    num_cpus = int(resources.get("CPU", 1))
    num_cpus = max(1, num_cpus)

    batch_size = max(1, num_candidates // (num_cpus * 10)) if num_candidates > 0 else 1
    log.info(f"Total candidates (prefix lengths): {num_candidates}")
    log.info(f"Using {num_cpus} CPUs, batch size {batch_size}")

    # Publish objects in Ray object store
    k0_ref = ray.put(k0)
    Q_ref = ray.put(Q)
    sorted_idx_ref = ray.put(sorted_idx)

    roots = np.exp(2 * np.pi * 1j * np.arange(K) / K)
    roots_ref = ray.put(roots)

    # Stream batches without materializing all l values
    def batched_l_values():
        iterator = iter(range(num_candidates))
        while True:
            batch = list(itertools.islice(iterator, batch_size))
            if not batch:
                break
            yield batch

    # Launch Ray tasks
    futures = []
    batch_id = 0
    start_time = time.time()

    for batch in batched_l_values():
        fut = process_rank_1_batch.remote(
            batch,
            k0_ref,
            sorted_idx_ref,
            Q_ref,
            roots_ref,
            K,
            batch_id,
        )
        futures.append(fut)
        batch_id += 1

    log.info(f"Submitted {len(futures)} batches to Ray")

    # Collect results
    best_score = float("-inf")
    best_l = 0

    for fut in futures:
        batch_score, batch_best_l, b_id = ray.get(fut)
        log.info(f"Completed batch {b_id}")
        if batch_best_l is None:
            continue
        if batch_score > best_score:
            best_score = batch_score
            best_l = batch_best_l
            log.info(f"New best score from batch {b_id}: {best_score} (l = {best_l})")

    elapsed = time.time() - start_time
    log.info(f"Rank 1 search complete in {elapsed:.4f} seconds")
    log.info(f"Best prefix l = {best_l}")

    # Reconstruct best assignment
    best_k = k0.copy()
    if best_l > 0:
        best_k[sorted_idx[:best_l]] = (best_k[sorted_idx[:best_l]] + 1) % K
    best_z = roots[best_k]

    return float(best_score), best_k, best_z


def compute_recovery(z_alg, Q, opt_value):
    alg_value = float(np.real(z_alg.conj() @ Q @ z_alg))
    return alg_value / opt_value


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel MAX k CUT experiment (rank 1)")
    parser.add_argument("--n", type=int, default=10000, help="Problem size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--precision", type=int, default=64, choices=[16, 32, 64],
                        help="Numeric precision: 16, 32, or 64 (default: 64)")
    parser.add_argument("--debug", action="store_true", help="Compute correctness with opt_K_cut")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to store outputs")
    parser.add_argument(
        "--graph_dir",
        type=str,
        default=None,
        help="Directory containing Q_{n}.npy and V_{n}.npy",
    )
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

    # Load Q and V if not in debug mode
    if not args.debug:
        log.info("Loading Q and V...")

        if args.graph_dir is not None:
            q_path = os.path.join(args.graph_dir, f"Q_{args.n}.npy")
            v_path = os.path.join(args.graph_dir, f"V_{args.n}.npy")

            log.info(f"Loading Q from {q_path}")
            log.info(f"Loading V from {v_path}")

            Q = np.load(q_path).astype(float_dtype, copy=False)
            V = np.load(v_path)
            V = np.asarray(V, dtype=complex_dtype)
        else:
            Q = generate_Q(0.5, args.n, "erdos_renyi", seed=args.seed)
            Q = np.asarray(Q, dtype=float_dtype)
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
    best_score, best_k, best_z = process_rank_1_parallel(V[:, 0], Q, K=3)
    elapsed = time.time() - start

    log.info(f"Rank 1 result: score = {best_score}")
    log.info(f"Execution time: {elapsed:.4f} seconds")
    log.info(f"Final partition assignment k:\n{best_k}")
    log.info(f"Computed complex spin vector z:\n{best_z}")

    output = {
        "n": args.n,
        "seed": args.seed,
        "rank": 1,
        "precision": args.precision,
        "best_score": float(best_score),
        "time_seconds": float(elapsed),
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
    filename = f"{timestamp}_result_n{args.n}_r1_p{args.precision}.json"
    path = os.path.join(args.results_dir, filename)

    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved results to {path}")


if __name__ == "__main__":
    main()
