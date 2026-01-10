import argparse
import os
import numpy as np
import networkx as nx
from gen_v import gen_V_given_Q


def main():
    parser = argparse.ArgumentParser(
        description="Generate multiple SBM graphs with different seeds and low-rank eigenvector matrices."
    )
    parser.add_argument(
        "--n", type=int, default=1000,
        help="Number of nodes in the graph."
    )
    parser.add_argument(
        "--blocks", type=int, default=2,
        help="Number of blocks."
    )
    parser.add_argument(
        "--block_sizes", nargs="*", type=int, default=[500, 500],
        help="Sizes of each block."
    )
    parser.add_argument(
        "--prob_within", type=float, default=0.5,
        help="Within-block edge probability."
    )
    parser.add_argument(
        "--prob_between", type=float, default=0.01,
        help="Between-block edge probability."
    )
    parser.add_argument(
        "--rank", type=int, default=1,
        help="Rank r for the low-rank eigenvector matrix V."
    )
    parser.add_argument(
        "--num_seeds", type=int, default=20,
        help="Number of seeds to generate, starting from 0."
    )
    parser.add_argument(
        "--out_dir", type=str, default="graphs_sbm",
        help="Directory to store Q and V matrices."
    )
    args = parser.parse_args()

    n = args.n
    num_blocks = args.blocks
    block_sizes = args.block_sizes
    p_in = args.prob_within
    p_out = args.prob_between
    r = args.rank
    num_seeds = args.num_seeds
    out_dir = args.out_dir

    if len(block_sizes) != 1 and sum(block_sizes) != n:
        raise ValueError(f"--block_sizes must sum to {n}")
    if len(block_sizes) == 1:
        if n % num_blocks != 0:
            raise ValueError("If --block_sizes has length 1, n must be divisible by --blocks")
        block_sizes = [n // num_blocks] * num_blocks
    if len(block_sizes) != num_blocks:
        raise ValueError("--block_sizes must have length 1 or match --blocks")

    os.makedirs(out_dir, exist_ok=True)

    probs = np.full((num_blocks, num_blocks), p_out)
    np.fill_diagonal(probs, p_in)

    for seed in range(num_seeds):
        print(
            f"Generating SBM graph with n={n}, blocks={num_blocks}, "
            f"block_sizes={block_sizes}, p_in={p_in}, p_out={p_out}, "
            f"seed={seed}, rank={r}"
        )

        G = nx.stochastic_block_model(
            block_sizes,
            probs,
            seed=seed,
            directed=False,
            selfloops=False,
            sparse=True,
        )

        Q = np.array(nx.laplacian_matrix(G).todense())
        V = gen_V_given_Q(Q, r)

        q_path = os.path.join(out_dir, f"Q_{n}_seed_{seed}.npy")
        v_path = os.path.join(out_dir, f"V_{n}_seed_{seed}.npy")

        np.save(q_path, Q)
        np.save(v_path, V)

        print(f"Saved Q to {q_path}")
        print(f"Saved V to {v_path}")


if __name__ == "__main__":
    main()
