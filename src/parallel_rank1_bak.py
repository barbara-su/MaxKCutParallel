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
def process_rank1_candidate(l, k0, sorted_idx, Q, K, phase_table):
    k = k0.copy()
    k[sorted_idx[:l]] = (k[sorted_idx[:l]] + 1) % K
    z = phase_table[k]
    score = np.real(z.conj() @ Q @ z)
    return score

def process_rank1_parallel(V, Q, K, phase_table):
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

    best_score = -np.inf
    best_l = 0

    for batch_start in range(0, n + 1, batch_size):
        batch_end = min(batch_start + batch_size, n + 1)
        futures = [
            process_rank1_candidate.remote(
                l, k0, sorted_idx, Q, K, phase_table
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
    best_z = np.exp(2 * np.pi * 1j * best_k / K)

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
    Q_ref = ray.put(Q) # put Q into ray object store to speed up
    log.info("Random graph Laplacian generated")

    eigvals, eigvecs = np.linalg.eigh(Q)
    _, V = low_rank_matrix(Q, eigvals, eigvecs, r=1)
    log.info("Eigen decomposition complete and top eigenvector extracted")

    log.info("Executing parallel rank 1 algorithm")
    start = time.time()
    
    # precompute K exponential values
    K = 3
    phase_table = np.exp(2 * np.pi * 1j * np.arange(K) / K)
    phase_table_ref = ray.put(phase_table)
    log.info("Computed phase table")
    
    best_score, best_k, best_z = process_rank1_parallel(
        V[:, 0], Q_ref, K, phase_table_ref
    )
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

    # this only works for n <= 10
    if args.compute_recovery:
        opt_value = solve_sdp_optimal(Q)
        recovery = compute_recovery(best_z, Q, opt_value)
        log.info(f"Recovery ratio: {recovery}")
        output["recovery_ratio"] = float(recovery)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_result_n{args.n}.json"
    path = os.path.join(args.results_dir, filename)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved results to {path}")


if __name__ == "__main__":
    main()
