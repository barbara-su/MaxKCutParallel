import numpy as np
import ray
from utils import *
import cvxpy as cvx
import time
import logging
import warnings
import argparse
import json
import os
from datetime import datetime

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
def process_rank1_chunk(l_start, l_end, k0, sorted_idx, Q, roots, K):
    """
    Ray worker that searches over l in [l_start, l_end)
    and returns the best score and best l in that interval.
    k0, sorted_idx, Q, roots are already concrete arrays here.
    """
    worker_log = logging.getLogger(__name__)
    num_candidates = l_end - l_start
    worker_log.info(
        f"[Worker] Starting chunk [{l_start}, {l_end}) with {num_candidates} candidates"
    )

    best_score = -np.inf
    best_l = l_start

    sorted_idx_local = sorted_idx
    k0_local = k0

    for idx, l in enumerate(range(l_start, l_end)):
        k = k0_local.copy()
        k[sorted_idx_local[:l]] = (k[sorted_idx_local[:l]] + 1) % K

        z = roots[k]
        score = np.real(z.conj() @ Q @ z)

        if score > best_score:
            best_score = score
            best_l = l

        # progress log every 1000 candidates within this chunk
        if idx > 0 and idx % 1000 == 0:
            worker_log.info(
                f"[Worker] Chunk [{l_start}, {l_end}) processed {idx} / {num_candidates}"
            )

    worker_log.info(
        f"[Worker] Finished chunk [{l_start}, {l_end}); best_l = {best_l}, best_score = {best_score}"
    )
    return float(best_score), int(best_l)


def process_rank1_parallel(V, Q, K):
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

    # precompute K-th roots of unity
    roots = np.exp(2 * np.pi * 1j * np.arange(K) / K)

    # evaluate the starting configuration at l = 0
    z0 = roots[k0]
    best_score = np.real(z0.conj() @ Q @ z0)
    best_l = 0

    # put large arrays into Ray object store once
    Q_ref = ray.put(Q)
    k0_ref = ray.put(k0)
    sorted_idx_ref = ray.put(sorted_idx)
    roots_ref = ray.put(roots)

    # decide how many chunks we want
    resources = ray.available_resources()
    if "CPU" in resources:
        cpus = int(resources["CPU"])
    else:
        cpus = 1

    total_l = n + 1

    num_chunks = max(1, cpus)
    chunk_size = max(1, (total_l + num_chunks - 1) // num_chunks)

    log.info(f"Using {num_chunks} chunks with chunk_size {chunk_size}")

    futures = []
    for l_start in range(0, total_l, chunk_size):
        l_end = min(l_start + chunk_size, total_l)
        futures.append(
            process_rank1_chunk.remote(
                l_start, l_end,
                k0_ref, sorted_idx_ref, Q_ref, roots_ref, K
            )
        )

    results = ray.get(futures)

    for chunk_score, chunk_l in results:
        if chunk_score > best_score:
            best_score = chunk_score
            best_l = chunk_l

    log.info(f"Best prefix l = {best_l}")

    # reconstruct the assignment for best_l
    best_k = k0.copy()
    best_k[sorted_idx[:best_l]] = (best_k[sorted_idx[:best_l]] + 1) % K

    best_z = roots[best_k]

    return best_score, best_k, best_z


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
    parser.add_argument("--compute_recovery", action="store_true", help="Compute recovery ratio")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to store outputs")
    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)
    log.info("Starting MAX 3 CUT experiment")
    ray.init(ignore_reinit_error=True)
    log.info("Ray initialized")

    Q = generate_Q(0.5, args.n, 'erdos_renyi', seed=args.seed)
    log.info("Random graph Laplacian generated")

    eigvals, eigvecs = np.linalg.eigh(Q)
    _, V = low_rank_matrix(Q, eigvals, eigvecs, r=1)
    log.info("Eigen decomposition complete and top eigenvector extracted")

    log.info("Executing chunked parallel rank 1 algorithm")
    start = time.time()
    best_score, best_k, best_z = process_rank1_parallel(V[:, 0], Q, K=3)
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
        "best_z_imag": np.imag(best_z).tolist()
    }

    if args.compute_recovery:
        opt_value = solve_sdp_optimal(Q)
        recovery = compute_recovery(best_z, Q, opt_value)
        log.info(f"Recovery ratio: {recovery}")
        output["recovery_ratio"] = float(recovery)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_result_n{args.n}.json"
    os.makedirs(args.results_dir, exist_ok=True)
    path = os.path.join(args.results_dir, filename)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved results to {path}")


if __name__ == "__main__":
    main()
