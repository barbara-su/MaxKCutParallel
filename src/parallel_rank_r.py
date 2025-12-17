import numpy as np
import ray
from utils import *
import time
import logging
import warnings
import argparse
import json
from datetime import datetime
import itertools
import os
from parallel_rank_1 import process_rank_1_parallel
import cvxpy as cvx

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
def process_combination_batch(V_tilde,
                              V,
                              K,
                              r,
                              Q,
                              combinations_batch,
                              batch_id,
                              row_mapping,
                              inverse_mapping,
                              s):
    """
    Process a batch of combinations for the rank-r algorithm.
    """
    n = V.shape[0]
    best_score = float("-inf")
    best_candidate = None

    for combo in combinations_batch:
        try:
            # selected hyperplanes
            I = np.array(combo, dtype=int)
            VI = V_tilde[I]  # shape (2r - 1, 2r)

            # find intersection of hyperplanes
            c_tilde = find_intersection(VI)
            phi, sign_c = determine_phi_sign_c(c_tilde)

            # check if the last angle is within the decision region
            if not (-np.pi / K < phi[2 * r - 2] <= np.pi / K):
                continue

            # candidate spin vector
            candidate = np.zeros(n, dtype=complex)

            # adjust sign of c_tilde
            c_tilde = c_tilde * sign_c

            # convert to complex form
            c = convert_ctilde_to_complex(c_tilde, r)

            # get the original V indices that correspond to the selected V_tilde rows
            v_indices_used = set()
            for idx in I:
                v_row, _ = row_mapping[idx]
                v_indices_used.add(v_row)

            # vertices not in selected hyperplanes
            for k in range(n):
                if k in v_indices_used:
                    continue
                v_c = V[k] @ c
                metric = np.real(np.conj(s) * v_c)
                s_idx = int(np.argmax(metric))
                candidate[k] = s[s_idx]

            # vertices in selected hyperplanes
            for v_idx in v_indices_used:
                # all V_tilde rows incident to this vertex
                vtilde_rows_for_v = [idx for idx in inverse_mapping[v_idx] if idx in I]

                assigned = False
                for vtilde_idx in vtilde_rows_for_v:
                    pos = int(np.where(I == vtilde_idx)[0][0])
                    VI_minus = np.delete(VI, pos, axis=0)
                    try:
                        new_c_tilde = find_intersection_fixed_angle(VI_minus, r, K)
                        new_c = convert_ctilde_to_complex(new_c_tilde, r)
                        v_c = V[v_idx] @ new_c
                        metric = np.real(np.conj(s) * v_c)
                        s_idx = int(np.argmax(metric))
                        candidate[v_idx] = s[s_idx]
                        assigned = True
                        break
                    except ValueError:
                        continue

                # fallback
                if not assigned or np.abs(candidate[v_idx]) < 1e-10:
                    v_c = V[v_idx] @ c
                    metric = np.real(np.conj(s) * v_c)
                    s_idx = int(np.argmax(metric))
                    candidate[v_idx] = s[s_idx]

            # evaluate objective
            score = np.real(candidate.conj() @ Q @ candidate)

            if score > best_score:
                best_score = score
                best_candidate = candidate.copy()

        except (ValueError, np.linalg.LinAlgError):
            continue

    return best_score, best_candidate, batch_id

def process_rankr_single(V, Q, K=3):
    """
    Single rank-r enumeration using Algorithm 2 with Ray.
    Does not recurse on lower ranks.
    """
    n, r = V.shape
    log.info(f"Rank r subroutine (single rank): n = {n}, r = {r}, K = {K}")

    # compute v_tilde
    log.info("Computing V_tilde")
    V_tilde = compute_vtilde(V)

    # row mappings between V_tilde rows and vertices
    log.info("Computing row mappings for V_tilde")
    row_mapping, inverse_mapping = get_row_mapping(n, K)

    # Kth roots of unity
    s = np.exp(1j * 2 * np.pi * np.arange(K) / K)

    # number of V_tilde rows and combination size
    num_vtilde_rows = K * n
    comb_size = 2 * r - 1

    if comb_size > num_vtilde_rows:
        raise ValueError("Combination size 2r - 1 exceeds K * n")

    # theoretical number of combinations
    num_combinations = math.comb(num_vtilde_rows, comb_size)
    log.info(
        f"Total number of (2r - 1)-tuples: C({num_vtilde_rows}, {comb_size}) = {num_combinations}"
    )

    # decide batch size
    resources = ray.available_resources()
    if "CPU" in resources:
        num_cpus = int(resources["CPU"])
    else:
        num_cpus = 1
    num_cpus = max(1, num_cpus)

    # 10 batches per cpu
    if num_combinations > 0:
        batch_size = max(1, num_combinations // (num_cpus * 10))
    else:
        batch_size = 1

    log.info(f"Using {num_cpus} CPUs, batch size {batch_size}")

    # put objects into Ray object store
    V_tilde_ref = ray.put(V_tilde)
    V_ref = ray.put(V)
    Q_ref = ray.put(Q)
    row_mapping_ref = ray.put(row_mapping)
    inverse_mapping_ref = ray.put(inverse_mapping)
    s_ref = ray.put(s)

    # helper to stream batches without storing all combinations
    def batched_combinations():
        iterator = itertools.combinations(range(num_vtilde_rows), comb_size)
        while True:
            batch = list(itertools.islice(iterator, batch_size))
            if not batch:
                break
            yield batch

    # launch Ray tasks
    futures = []
    batch_id = 0
    start_time = time.time()
    for batch in batched_combinations():
        fut = process_combination_batch.remote(
            V_tilde_ref,
            V_ref,
            K,
            r,
            Q_ref,
            batch,
            batch_id,
            row_mapping_ref,
            inverse_mapping_ref,
            s_ref,
        )
        futures.append(fut)
        batch_id += 1

    log.info(f"Submitted {len(futures)} batches to Ray")

    # collect results
    best_score = float("-inf")
    best_candidate = None

    for fut in futures:
        batch_score, batch_candidate, b_id = ray.get(fut)
        log.info(f"Completed batch {b_id}")
        if batch_candidate is None:
            continue
        if batch_score > best_score:
            best_score = batch_score
            best_candidate = batch_candidate
            log.info(f"New best score from batch {b_id}: {best_score}")

    elapsed = time.time() - start_time
    log.info(f"Rank r single-rank search complete in {elapsed:.4f} seconds")

    if best_candidate is None:
        raise RuntimeError("Rank r algorithm did not find any feasible candidate")

    best_k = complex_to_partition(best_candidate, K)
    best_z = best_candidate
    return float(best_score), best_k, best_z

def process_rankr_recursive(V, Q, K=3):
    """
    Recursive rank-r max-k-cut
    """
    n, r = V.shape
    log.info(f"Recursive rank solver at rank r = {r}")

    # base case: r = 1, use the rank 1 Ray routine
    if r == 1:
        log.info("Base case r = 1, calling process_rank_1_parallel")
        best_score, best_k, best_z = process_rank_1_parallel(V[:, 0], Q, K)
        return best_score, best_k, best_z

    # compute best candidate at current rank r
    curr_score, curr_k, curr_z = process_rankr_single(V, Q, K)

    best_score = curr_score
    best_k = curr_k
    best_z = curr_z

    # recursively consider lower rank r - 1
    if r > 1:
        log.info(f"Recursing to lower rank r = {r-1}")
        lower_score, lower_k, lower_z = process_rankr_recursive(V[:, :r-1], Q, K)

        if lower_score > best_score:
            log.info(f"Lower rank {r-1} improved score from {best_score} to {lower_score}")
            best_score = lower_score
            best_k = lower_k
            best_z = lower_z

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
    parser.add_argument("--rank", type=int, default=1, help="Low rank parameter r")
    parser.add_argument("--compute_recovery", action="store_true", help="Compute recovery ratio")
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

    log.info("Loading Q and V or computing them")
    
    if args.graph_dir is not None:
        q_path = os.path.join(args.graph_dir, f"Q_{args.n}.npy")
        v_path = os.path.join(args.graph_dir, f"V_{args.n}.npy")

        log.info(f"Loading Q from {q_path}")
        log.info(f"Loading V from {v_path}")

        Q = np.load(q_path)
        V_full = np.load(v_path)

        # if stored V has more columns than needed, truncate
        if V_full.shape[1] < args.rank:
            raise ValueError(f"Loaded V has only {V_full.shape[1]} columns, "
                             f"but rank {args.rank} was requested")
        V = V_full[:, :args.rank]

    else:
        Q = generate_Q(0.5, args.n, 'erdos_renyi', seed=args.seed)
        log.info("Random graph Laplacian generated")

        eigvals, eigvecs = np.linalg.eigh(Q)
        # low_rank_matrix should return V with shape (n, args.rank)
        _, V = low_rank_matrix(Q, eigvals, eigvecs, r=args.rank)

    log.info(f"Eigen decomposition complete and low rank factor V has shape {V.shape}")

    start = time.time()

    if args.rank == 1:
        log.info("Executing parallel rank 1 algorithm")
        best_score, best_k, best_z = process_rank_1_parallel(V[:, 0], Q, K=3)
    else:
        log.info(f"Executing recursive rank {args.rank} algorithm (r = 1..{args.rank})")
        best_score, best_k, best_z = process_rankr_recursive(V, Q, K=3)
    
    elapsed = time.time() - start

    log.info(f"Rank {args.rank} result: score = {best_score}")
    log.info(f"Execution time: {elapsed:.4f} seconds")
    log.info(f"Final partition assignment k:\n{best_k}")
    log.info(f"Computed complex spin vector z:\n{best_z}")
    
    output = {
        "n": args.n,
        "seed": args.seed,
        "rank": args.rank,
        "best_score": float(best_score),
        "time_seconds": float(elapsed),
        "best_k": best_k.tolist(),
        "best_z_real": np.real(best_z).tolist(),
        "best_z_imag": np.imag(best_z).tolist(),
        "num_workers": num_workers,
    }
    
    if args.compute_recovery:
        opt_value = solve_sdp_optimal(Q)
        recovery = compute_recovery(best_z, Q, opt_value)
        log.info(f"Score: {opt_value}")
        log.info(f"Recovery ratio: {recovery}")
        output["recovery_ratio"] = float(recovery)

    os.makedirs(args.results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_result_n{args.n}_r{args.rank}.json"
    path = os.path.join(args.results_dir, filename)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved results to {path}")

if __name__ == "__main__":
    main()