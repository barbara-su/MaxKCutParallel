import argparse
import os
import numpy as np
import networkx as nx
from utils import *
from gen_v import gen_V_given_Q


def main():
    parser = argparse.ArgumentParser(
        description="Generate stochastic block model graph and low-rank eigenvector matrix."
    )
    parser.add_argument(
        "--n", type=int, default=1000,
        help="Number of nodes in the graph."
    )
    parser.add_argument(
        "--blocks", type=float, default=2,
        help="number of blocks."
    )
    parser.add_argument(
        "--block_sizes", nargs="*", type=int, default=[500, 500],
        help="sizes of each blocks."
    )
    parser.add_argument(
        "--prob_within", nargs="*", type=float, default=[0.5],
        help="Probability of an edge between two nodes in same block."
    )
    parser.add_argument(
        "--prob_between", nargs="*", type=float, default=[0.01],
        help="Probability of an edge between two nodes in different blocks."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed."
    )
    parser.add_argument(
        "--rank", type=int, default=1,
        help="Rank r for the low-rank eigenvector matrix V."
    )
    parser.add_argument(
        "--out_dir", type=str, default="graphs",
        help="Directory to store Q_n.npy and V_n.npy."
    )
    args = parser.parse_args()

    n = args.n
    num_blocks = args.blocks

    block_sizes = args.block_sizes
    if len(block_sizes) != 1 and sum(block_sizes) != n:
        parser.error(
            f"--block_sizes must sum to {args.n}"
        )
    if len(block_sizes) == 1 and n % num_blocks != 0:
        parser.error(
            "if --block_sizes is 1, n must be divisible by num_blocks "
        )
    if len(block_sizes) != 1 and len(block_sizes) != num_blocks:
        parser.error(
            "--block_sizes must have length 1 or match --blocks "
            f"(got {len(args.block_sizes)}, expected 1 or {args.blocks})"
        )

    if len(args.prob_within) != 1 and len(args.prob_within) != num_blocks:
        parser.error(
            "--prob_within must have length 1 or match --num_blocks "
            f"(got {len(args.prob_within)}, expected 1 or {args.blocks})"
        )
    if len(args.prob_within) == 1:
        prob_within = args.prob_within[0]
    else:
        prob_within = args.prob_within

    if len(args.prob_between) != 1 and len(args.prob_between) != num_blocks:
        parser.error(
            "--prob_between must have length 1 or match --num_blocks "
            f"(got {len(args.prob_between)}, expected 1 or {args.blocks})"
        )
    if len(args.prob_between) == 1:
        prob_between = args.prob_between[0]
    else:
        prob_between = args.prob_between

    seed = args.seed
    r = args.rank
    out_dir = args.out_dir

    print(f"Generating SBM graph with n = {n}, num_blocks = {num_blocks}, block_sizes = {block_sizes}, prob_within = {prob_within}, prob_between = {prob_between}, seed = {seed}, rank = {r}")
    os.makedirs(out_dir, exist_ok=True)

    probs = np.zeros((num_blocks, num_blocks))
    if isinstance(prob_between, float):
        probs[:] = prob_between
        np.fill_diagonal(probs, 0)
    else:
        raise NotImplementedError("not implemented: prob_between as an array")

    if isinstance(prob_within, float):
        np.fill_diagonal(probs, prob_within)
    else:
        raise NotImplementedError("not implemented: prob_within as an array")

    G = nx.stochastic_block_model(
            block_sizes, probs, nodelist=None, seed=seed, directed=False, selfloops=False, sparse=True
        )

    # generate graph
    Q = np.array(nx.laplacian_matrix(G).todense())

    V = gen_V_given_Q(Q, r)

    print(f"Q shape: {Q.shape}")
    print(f"V shape: {V.shape}")

    # save
    q_path = os.path.join(out_dir, f"Q_{n}.npy")
    v_path = os.path.join(out_dir, f"V_{n}.npy")

    np.save(q_path, Q)
    np.save(v_path, V)

    print(f"Saved Q to {q_path}")
    print(f"Saved V (rank {r}) to {v_path}")

if __name__ == "__main__":
    main()
