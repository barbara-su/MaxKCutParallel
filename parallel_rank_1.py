# import numpy as np
# import ray
# from utils import *
# import cvxpy as cvx
# import time
# import os

# import warnings
# warnings.filterwarnings(
#     "ignore",
#     message="pkg_resources is deprecated as an API"
# )


# @ray.remote
# def process_rank1_candidate(l, k0, sorted_idx, Q, K):
#     """
#     rank 1 subroutine
    
#     :param l: Description
#     :param k0: Description
#     :param sorted_idx: Description
#     :param Q: Description
#     :param K: Description
#     """
#     k = k0.copy()
    
#     # Apply l prefix increments: ks[j] = ks[j] + 1 mod K
#     k[sorted_idx[:l]] = (k[sorted_idx[:l]] + 1) % K

#     # Convert to complex spins
#     z = np.exp(2 * np.pi * 1j * k / K)
#     score = np.real(z.conj() @ Q @ z)

#     return score, k

# def process_rank1_parallel(V, Q, K):
#     """
#     Main alg for ray cluster processing rank 1
    
#     :param V: The single eigenvector of Q
#     :param Q: The Laplacian matrix
#     :param K: number of cuts to produce
#     """
#     n = V.shape[0] # number of nodes

#     # extract real and imaginary parts of the eigenvector
#     real_q1 = np.real(V).flatten()
#     im_q1 = np.imag(V).flatten()

#     # compute initial assignment k0
#     thetas = np.arctan2(im_q1, real_q1)
#     thetas = np.where(thetas < 0, thetas + 2*np.pi, thetas)
#     b = K * thetas / (2 * np.pi)
#     b_floor = np.floor(b).astype(int)
#     k0 = b_floor % K 
    
#     # compute boundary points, and argsort
#     phi_hat = 0.5 - b + b_floor
#     phis = 2 * np.pi * phi_hat / K
#     sorted_idx = np.argsort(phis)

#     futures = [
#         process_rank1_candidate.remote(l, k0, sorted_idx, Q, K)
#         for l in range(n + 1)
#     ]
    
#     results = ray.get(futures)
#     best_score, best_k = max(results, key=lambda x: x[0])
    
#     return best_score, best_k


# def compute_recovery(z_alg, Q, opt_value):
#     """
#     algorithm value / optimal value
#     """
#     alg_value = np.real(z_alg.conj() @ Q @ z_alg)
#     return alg_value / opt_value


# def solve_sdp_optimal(Q):
#     """
#     Solve the SDP relaxation for MAX-K-CUT (rank-1 case uses MAX-3-CUT).
#     Returns the SDP optimal objective value.

#     For small sizes (n <= 12), CVXPY is fine.
#     """
#     n = Q.shape[0]

#     # SDP variable
#     X = cvx.Variable((n, n), PSD=True)

#     # MAX-K-CUT SDP relaxation objective
#     obj = cvx.Maximize(cvx.sum(cvx.multiply(Q, X)))

#     # Constraints
#     constraints = [
#         cvx.diag(X) == 1      # unit diagonal
#     ]
#     prob = cvx.Problem(obj, constraints)
#     prob.solve(solver=cvx.SCS, verbose=False)

#     return prob.value


# # example unit test
# def main():
#     print("\nInitializing Ray...")
#     np.random.seed(42)
#     ray.init(ignore_reinit_error=True)
    
#     Q = generate_Q(0.5, 5000, 'erdos_renyi')
    
#     eigvals, eigvecs = np.linalg.eigh(Q)
#     _, V = low_rank_matrix(Q, eigvals, eigvecs, r=1)

#     print("\nRunning parallel Rank-1 MAX-3-CUT")
#     start = time.time()
#     best_score, best_k = process_rank1_parallel(V[:, 0], Q, K=3)
    
#     end = time.time()
#     elapsed = end - start
    
#     print(f"\nRank-1 algorithm result: Score={best_score}")
#     print(f"Execution time: {elapsed:.4f} seconds")
    
#     print("\nFinal partition assignment k:")
#     print(best_k)
    
#     # opt_value = solve_sdp_optimal(Q)
#     # recovery = compute_recovery(best_z, Q, opt_value)
#     # print("\nRecovery percentage:", recovery * 100, "%")
#     # print("\np = 50 test completed.\n")


# # Run if executed directly
# if __name__ == "__main__":
#     main()

import numpy as np
import ray
from utils import *
import cvxpy as cvx
import time
import os
import logging
import warnings

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API"
)

# -----------------------------------------------------------
# Logging setup
# -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)


@ray.remote
def process_rank1_candidate(l, k0, sorted_idx, Q, K):
    k = k0.copy()
    k[sorted_idx[:l]] = (k[sorted_idx[:l]] + 1) % K
    z = np.exp(2 * np.pi * 1j * k / K)
    score = np.real(z.conj() @ Q @ z)
    return score, k


def process_rank1_parallel(V, Q, K):
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

    log.info("Dispatching Ray tasks for all prefix increments")

    futures = [
        process_rank1_candidate.remote(l, k0, sorted_idx, Q, K)
        for l in range(n + 1)
    ]

    results = ray.get(futures)
    best_score, best_k = max(results, key=lambda x: x[0])

    log.info("Ray tasks completed and best score recovered")

    return best_score, best_k


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


def main():
    log.info("Starting MAX 3 CUT experiment")

    np.random.seed(42)
    ray.init(ignore_reinit_error=True)
    log.info("Ray initialized")

    Q = generate_Q(0.5, 4000, 'erdos_renyi')
    log.info("Random graph Laplacian generated")

    eigvals, eigvecs = np.linalg.eigh(Q)
    _, V = low_rank_matrix(Q, eigvals, eigvecs, r=1)
    log.info("Eigen decomposition complete; extracted top eigenvector")

    log.info("Executing parallel rank 1 algorithm")
    start = time.time()
    best_score, best_k = process_rank1_parallel(V[:, 0], Q, K=3)
    elapsed = time.time() - start

    log.info(f"Rank 1 result: score={best_score}")
    log.info(f"Execution time: {elapsed:.4f} seconds")

    print("\nFinal partition assignment k:")
    print(best_k)


if __name__ == "__main__":
    main()
