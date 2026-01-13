import argparse
import os
import numpy as np
import networkx as nx
from utils import *
from gen_v import *

def main():
    parser = argparse.ArgumentParser(
        description="Generate G(n, p) graph and low-rank eigenvector matrix."
    )
    parser.add_argument(
        "--n", type=int, default=10000,
        help="Number of nodes in the graph."
    )
    parser.add_argument(
        "--p", type=float, default=0.5,
        help="Edge probability for G(n, p)."
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
    p = args.p
    seed = args.seed
    r = args.rank
    out_dir = args.out_dir

    print(f"Generating graph with n = {n}, p = {p}, seed = {seed}, rank = {r}")
    os.makedirs(out_dir, exist_ok=True)

    # generate graph
    G = nx.fast_gnp_random_graph(n=n, p=p, seed=seed)
    Q = np.array(nx.laplacian_matrix(G).todense())

    V = gen_V_given_Q(Q, r)
    Q_hat = gen_Q_hat_given_V(V)

    print(f"Q shape: {Q.shape}")
    print(f"V shape: {V.shape}")

    # save
    q_path = os.path.join(out_dir, f"Q_{n}.npy")
    v_path = os.path.join(out_dir, f"V_{n}.npy")

    np.save(q_path, Q_hat)
    np.save(v_path, V)

    print(f"Saved Q to {q_path}")
    print(f"Saved V (rank {r}) to {v_path}")

if __name__ == "__main__":
    main()
