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
import math
from src.parallel_rank_1 import process_rank_1_parallel
import cvxpy as cvx  # kept for parity

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
            candidate = np.zeros(n, dtype=V.dtype)

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
                if (not assigned) or (np.abs(candidate[v_idx]) < 1e-10):
                    v_c = V[v_idx] @ c
                    metric = np.real(np.conj(s) * v_c)
                    s_idx = int(np.argmax(metric))
                    candidate[v_idx] = s[s_idx]

            # evaluate objective
            score = float(np.real(candidate.conj() @ Q @ candidate))

            if score > best_score:
                best_score = score
                best_candidate = candidate.copy()

        except (ValueError, np.linalg.LinAlgError):
            continue

    return best_score, best_candidate, batch_id


def process_rankr_single(V, Q, K=3, candidates_per_task=1000):
    """
    Single rank-r enumeration using Algorithm 2 with Ray.
    Does not recurse on lower ranks.
    """
    n, r = V.shape
    log.info(f"Rank r subroutine (single rank): n = {n}, r = {r}, K = {K}")

    if candidates_per_task <= 0:
        raise ValueError("--candidates_per_task must be a positive integer")

    # compute v_tilde
    log.info("Computing V_tilde")
    V_tilde = compute_vtilde(V)

    # row mappings between V_tilde rows and vertices
    log.info("Computing row mappings for V_tilde")
    row_mapping, inverse_mapping = get_row_mapping(n, K)

    # Kth roots of unity
    s = np.exp(1j * 2 * np.pi * np.arange(K) / K).astype(V.dtype, copy=False)

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
    num_cpus = int(resources.get("CPU", 1))
    num_cpus = max(1, num_cpus)

    # NOTE: now candidates_per_task is the chunk size (combos per Ray task), not derived from num_cpus.
    batch_size = int(candidates_per_task)
    total_tasks = (num_combinations + batch_size - 1) // batch_size if num_combinations > 0 else 0

    log.info(f"Using {num_cpus} CPUs, candidates_per_task (batch_size) {batch_size}")
    log.info(f"Total Ray tasks (ceil(num_combinations/batch_size)): {total_tasks}")

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


def process_rankr_recursive(V, Q, K=3, candidates_per_task=1000):
    """
    Recursive rank-r max-k-cut
    """
    n, r = V.shape
    log.info(f"Recursive rank solver at rank r = {r}")

    # base case: r = 1, use the rank 1 Ray routine
    if r == 1:
        log.info("Base case r = 1, calling process_rank_1_parallel")
        best_score, best_k, best_z = process_rank_1_parallel(
            V[:, 0], Q, K, candidates_per_task=candidates_per_task
        )
        return best_score, best_k, best_z

    # compute best candidate at current rank r
    curr_score, curr_k, curr_z, _, _, _ = process_rankr_single(
        V, Q, K, candidates_per_task=candidates_per_task
    )

    best_score = curr_score
    best_k = curr_k
    best_z = curr_z

    # recursively consider lower rank r - 1
    if r > 1:
        log.info(f"Recursing to lower rank r = {r-1}")
        lower_score, lower_k, lower_z = process_rankr_recursive(
            V[:, :r-1], Q, K, candidates_per_task=candidates_per_task
        )

        if lower_score > best_score:
            log.info(f"Lower rank {r-1} improved score from {best_score} to {lower_score}")
            best_score = lower_score
            best_k = lower_k
            best_z = lower_z

    return best_score, best_k, best_z


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel MAX k CUT experiment")
    parser.add_argument("--n", type=int, default=10000, help="Problem size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--rank", type=int, default=1, help="Low rank parameter r")
    parser.add_argument("--precision", type=int, default=64, choices=[16, 32, 64],
                        help="Numeric precision: 16, 32, or 64 (default: 64)")
    parser.add_argument("--candidates_per_task", type=int, default=1000,
                        help="Max enumeration candidates per Ray task. Rank-1: prefix lengths l. Rank-r: combinations.")
    parser.add_argument("--debug", action="store_true", help="Compute recovery ratio")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to store outputs")
    parser.add_argument("--graph_dir", type=str, default=None,
                        help="Directory containing Q_{n}.npy and V_{n}.npy")
    return parser.parse_args()


def main():
    args = parse_args()

    float_dtype, complex_dtype = set_numpy_precision(args.precision)
    log.info(f"Using precision={args.precision} -> float={float_dtype.__name__}, complex={complex_dtype.__name__}")

    np.random.seed(args.seed)
    log.info("Starting MAX 3 CUT experiment")
    ray.init(address="auto", ignore_reinit_error=True)
    log.info("Ray initialized")

    resources = ray.available_resources()
    num_workers = int(resources.get("CPU", 1))
    log.info(f"Detected {num_workers} Ray workers (CPU slots)")

    if not args.debug:
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

            Q = np.asarray(Q, dtype=float_dtype)
            V = np.asarray(V, dtype=complex_dtype)

        else:
            Q = generate_Q(0.5, args.n, 'erdos_renyi', seed=args.seed)
            Q = np.asarray(Q, dtype=float_dtype)
            log.info("Random graph Laplacian generated")

            eigvals, eigvecs = np.linalg.eigh(Q.astype(np.float64, copy=False))
            # low_rank_matrix should return V with shape (n, args.rank)
            _, V = low_rank_matrix(Q, eigvals, eigvecs, r=args.rank)
            V = np.asarray(V, dtype=complex_dtype)

    else:
        log.info("Generating debug low rank Q, V")
        Q, V = generate_debug_QV(n=args.n, rank=args.rank, seed=args.seed)
        Q = np.asarray(Q, dtype=float_dtype)
        V = np.asarray(V, dtype=complex_dtype)

    log.info(f"Eigen decomposition complete and low rank factor V has shape {V.shape}")

    start = time.time()

    if args.rank == 1:
        log.info("Executing parallel rank 1 algorithm")
        best_score, best_k, best_z = process_rank_1_parallel(
            V[:, 0], Q, K=3, candidates_per_task=args.candidates_per_task
        )
        num_candidates = args.n + 1
        total_tasks = (num_candidates + args.candidates_per_task - 1) // args.candidates_per_task
    else:
        log.info(f"Executing recursive rank {args.rank} algorithm (r = 1..{args.rank})")

        # log/store top-rank combination count
        top_n = V.shape[0]
        top_r = V.shape[1]
        num_vtilde_rows = 3 * top_n
        comb_size = 2 * top_r - 1
        num_candidates = int(math.comb(num_vtilde_rows, comb_size))
        total_tasks = (num_candidates + args.candidates_per_task - 1) // args.candidates_per_task

        best_score, best_k, best_z = process_rankr_recursive(
            V, Q, K=3, candidates_per_task=args.candidates_per_task
        )
        extra = {
            "num_candidates": int(num_candidates),
            "total_tasks": int(total_tasks),
        }

    elapsed = time.time() - start

    log.info(f"Rank {args.rank} result: score = {best_score}")
    log.info(f"Execution time: {elapsed:.4f} seconds")
    log.info(f"Final partition assignment k:\n{best_k}")
    log.info(f"Computed complex spin vector z:\n{best_z}")

    output = {
        "n": args.n,
        "seed": args.seed,
        "rank": args.rank,
        "precision": args.precision,
        "candidates_per_task": int(args.candidates_per_task),
        "best_score": float(best_score),
        "time_seconds": float(elapsed),
        "best_k": best_k.tolist(),
        "best_z_real": np.real(best_z).tolist(),
        "best_z_imag": np.imag(best_z).tolist(),
        "num_workers": num_workers,
    }

    if args.debug:  # will only work for n = 10
        opt_score, _ = opt_K_cut(Q.astype(np.float64, copy=False))
        log.info(f"Correct score: {opt_score}")

    os.makedirs(args.results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_result_n{args.n}_r{args.rank}_p{args.precision}_cpt{args.candidates_per_task}.json"
    path = os.path.join(args.results_dir, filename)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved results to {path}")


if __name__ == "__main__":
    main()
