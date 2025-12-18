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

# ignore ray warning
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


@ray.remote
def process_rank_1_batch(
    l_values,
    k0,
    sorted_idx,
    Q_csc,
    Q_diag,
    roots,
    omega,
    omega_minus_1_abs2,
    K,
    batch_id,
):
    """
    Incremental evaluation over a contiguous batch of prefix lengths l_values.
    Dtype-consistent: z uses roots.dtype, Qz uses promoted dtype from sparse @.
    """
    if not l_values:
        return float("-inf"), None, batch_id

    l0 = int(l_values[0])

    k = k0.copy()
    if l0 > 0:
        idx0 = sorted_idx[:l0]
        k[idx0] = (k[idx0] + 1) % K

    z = roots[k].copy()

    Qz = Q_csc @ z
    score = np.vdot(z, Qz).real

    best_score = score
    best_l = l0

    for l in l_values[1:]:
        i = int(sorted_idx[l - 1])

        z_i_old = z[i]
        delta = (omega - 1) * z_i_old
        Qz_i_old = Qz[i]

        score += 2 * np.real(np.conj(delta) * Qz_i_old) + omega_minus_1_abs2 * Q_diag[i]

        z[i] = z_i_old + delta

        col = Q_csc.getcol(i)
        if col.nnz:
            Qz[col.indices] += delta * col.data

        if score > best_score:
            best_score = score
            best_l = l

    return float(best_score), int(best_l), batch_id


def process_rank_1_parallel(V, Q, K=3, candidates_per_task=10):
    """
    Rank-1 parallel sweep over l = 0..n with:
      - sparse CSC Q
      - incremental updates within each batch
      - capped in-flight Ray tasks
    Returns (best_score, best_k, best_z, best_l)
    """
    n = V.shape[0]
    log.info(f"Rank 1 subroutine: received eigenvector of length {n}")

    if candidates_per_task <= 0:
        raise ValueError("--candidates_per_task must be a positive integer")

    # ensure sparse CSC Q
    if not issparse(Q):
        Q_csc = csc_matrix(Q)
    else:
        Q_csc = Q.tocsc()
    Q_csc.sort_indices()
    Q_csc.eliminate_zeros()
    Q_diag = np.asarray(Q_csc.diagonal(), dtype=Q_csc.dtype)

    # initial assignment
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
    num_cpus = max(1, int(resources.get("CPU", 1)))
    max_in_flight = max(2 * num_cpus, 1)

    log.info(f"Total candidates (prefix lengths): {num_candidates}")
    log.info(f"Detected CPUs: {num_cpus}")
    log.info(f"candidates_per_task: {candidates_per_task}")
    log.info(f"max_in_flight cap: {max_in_flight}")

    # roots and omega use V dtype
    roots = np.exp(2j * np.pi * np.arange(K) / K).astype(V.dtype, copy=False)
    omega = roots[1]
    omega_minus_1_abs2 = np.abs(omega - 1) ** 2  # real scalar (float dtype)

    # publish objects
    k0_ref = ray.put(k0)
    sorted_idx_ref = ray.put(sorted_idx)
    Q_ref = ray.put(Q_csc)
    Q_diag_ref = ray.put(Q_diag)
    roots_ref = ray.put(roots)
    omega_ref = ray.put(omega)
    omega_minus_1_abs2_ref = ray.put(float(omega_minus_1_abs2))

    def batched_l_values():
        it = iter(range(num_candidates))
        while True:
            batch = list(itertools.islice(it, candidates_per_task))
            if not batch:
                break
            yield batch

    start_time = time.time()

    in_flight = []
    submitted = 0
    completed = 0

    best_score = float("-inf")
    best_l = 0

    def submit_one(batch, batch_id):
        return process_rank_1_batch.remote(
            batch,
            k0_ref,
            sorted_idx_ref,
            Q_ref,
            Q_diag_ref,
            roots_ref,
            omega_ref,
            omega_minus_1_abs2_ref,
            K,
            batch_id,
        )

    batch_id = 0
    for batch in batched_l_values():
        in_flight.append(submit_one(batch, batch_id))
        batch_id += 1
        submitted += 1

        if len(in_flight) >= max_in_flight:
            done, in_flight = ray.wait(in_flight, num_returns=1)
            batch_score, batch_best_l, b_id = ray.get(done[0])
            completed += 1

            if batch_best_l is not None and batch_score > best_score:
                best_score = float(batch_score)
                best_l = int(batch_best_l)
                log.info(f"New best from batch {b_id}: score={best_score}, l={best_l}")

            if completed % 1000 == 0:
                log.info(
                    f"Progress: submitted={submitted}, completed={completed}, in_flight={len(in_flight)}"
                )

    while in_flight:
        done, in_flight = ray.wait(in_flight, num_returns=1)
        batch_score, batch_best_l, b_id = ray.get(done[0])
        completed += 1

        if batch_best_l is not None and batch_score > best_score:
            best_score = float(batch_score)
            best_l = int(batch_best_l)
            log.info(f"New best from batch {b_id}: score={best_score}, l={best_l}")

        if completed % 1000 == 0:
            log.info(
                f"Progress: submitted={submitted}, completed={completed}, in_flight={len(in_flight)}"
            )

    elapsed = time.time() - start_time
    log.info(f"Rank 1 search complete in {elapsed:.4f} seconds")
    log.info(f"Total submitted batches: {submitted}, completed: {completed}")
    log.info(f"Best prefix l = {best_l}")

    # reconstruct best assignment
    best_k = k0.copy()
    if best_l > 0:
        best_k[sorted_idx[:best_l]] = (best_k[sorted_idx[:best_l]] + 1) % K
    best_z = roots[best_k]

    return best_score, best_k, best_z, best_l


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel MAX k CUT experiment (rank 1)")
    parser.add_argument("--n", type=int, default=10000, help="Problem size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--precision",
        type=int,
        default=64,
        choices=[16, 32, 64],
        help="Numeric precision: 16, 32, or 64 (default: 64)",
    )
    parser.add_argument(
        "--candidates_per_task",
        type=int,
        default=10,
        help="How many candidates (l values) each Ray task evaluates serially (default: 10).",
    )
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
    log.info(
        f"Using precision={args.precision} -> float={float_dtype.__name__}, complex={complex_dtype.__name__}"
    )

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

            Q = np.load(q_path)
            V = np.load(v_path)
        else:
            Q = generate_Q(0.5, args.n, "erdos_renyi", seed=args.seed)
            log.info("Random graph Laplacian generated")
            eigvals, eigvecs = np.linalg.eigh(np.asarray(Q, dtype=np.float64))
            _, V = low_rank_matrix(Q, eigvals, eigvecs, r=1)
            log.info("Eigen decomposition complete and top eigenvector extracted")
    else:
        log.info("Generating debug low rank Q, V (rank 1)")
        Q, V = generate_debug_QV(n=args.n, rank=1, seed=args.seed)

    # enforce dtypes at boundary
    Q = np.asarray(Q, dtype=float_dtype)
    V = np.asarray(V, dtype=complex_dtype)

    log.info("Executing parallel rank 1 algorithm (incremental sparse updates + capped in-flight)")
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
    filename = f"{timestamp}_result_n{args.n}_r1_p{args.precision}_cpt{args.candidates_per_task}.json"
    path = os.path.join(args.results_dir, filename)

    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved results to {path}")


if __name__ == "__main__":
    main()
