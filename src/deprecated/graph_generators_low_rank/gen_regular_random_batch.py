import argparse
import os
import numpy as np
import networkx as nx
from gen_v import *

def main():
    parser = argparse.ArgumentParser(
        description="Generate multiple random regular graphs and low-rank eigenvector matrices."
    )
    parser.add_argument(
        "--n", type=int, required=True,
        help="Number of nodes in the graph."
    )
    parser.add_argument(
        "--d", type=int, required=True,
        help="Degree of each node."
    )
    parser.add_argument(
        "--rank", type=int, default=1,
        help="Rank r for the low-rank eigenvector matrix V."
    )
    parser.add_argument(
        "--out_dir", type=str, default="graphs",
        help="Directory to store Q and V matrices."
    )
    parser.add_argument(
        "--num_seeds", type=int, default=20,
        help="Number of seeds to generate, starting from 0."
    )
    args = parser.parse_args()

    n = args.n
    d = args.d
    r = args.rank
    out_dir = args.out_dir
    num_seeds = args.num_seeds

    os.makedirs(out_dir, exist_ok=True)

    for seed in range(num_seeds):
        print(f"Generating graph with n={n}, d={d}, seed={seed}, rank={r}")

        G = nx.random_regular_graph(d=d, n=n, seed=seed)
        Q = np.array(nx.laplacian_matrix(G).todense())
        V = gen_V_given_Q(Q, r)
        Q_hat = gen_Q_hat_given_V(V)

        q_path = os.path.join(out_dir, f"Q_{n}_seed_{seed}.npy")
        v_path = os.path.join(out_dir, f"V_{n}_seed_{seed}.npy")

        np.save(q_path, Q_hat)
        np.save(v_path, V)

        print(f"Saved Q to {q_path}")
        print(f"Saved V to {v_path}")

if __name__ == "__main__":
    main()
