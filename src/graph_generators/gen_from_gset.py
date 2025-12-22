import argparse
import os
import numpy as np
import networkx as nx
from utils import *

def main():
    parser = argparse.ArgumentParser(
        description="Generate random regular graph and low-rank eigenvector matrix."
    )
    parser.add_argument(
        "--gset", type=int, default=10000,
        help="Number of nodes in the graph."
    )
    parser.add_argument(
        "--random_weights", type=bool, default=0,
        help="0 for boolean weights, 1 for random weights"
    )
    parser.add_argument(
        "--random_low", type=float, default=0,
        help="lower bound of random weights"
    )
    parser.add_argument(
        "--random_high", type=float, default=1,
        help="upper bound of random weights"
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
        "--in_dir", type=str, default="../graphs/gset",
        help="Directory containing gset graphs."
    )
    parser.add_argument(
        "--out_dir", type=str, default="graphs",
        help="Directory to store Q_n.npy and V_n.npy."
    )
    args = parser.parse_args()

    gset = args.gset
    seed = args.seed
    r = args.rank
    random_weights = args.random_weights
    random_high = args.random_high
    random_low = args.random_high
    in_dir = args.in_dir
    out_dir = args.out_dir


    G = nx.Graph()
    with open(f"{in_dir}/G{args.gset}.txt") as f:
        num_nodes = int(next(f).split()[0])
        n = num_nodes
        G.add_nodes_from(range(num_nodes))
        for line in f:
            i, j, w = map(int, line.split()[:3])
            if random_weights is False:
                G.add_edge(i - 1, j - 1, weight=float(w))
            elif random_weights is True:
                G.add_edge(i - 1, j - 1, weight=float(w) *((random_high - random_low) * np.random.rand() + random_low))

    # generate graph
    print(f"Using G{gset} from GSet, n = {n}, seed = {seed}, rank = {r}")
    os.makedirs(out_dir, exist_ok=True)
    
    Q = np.array(nx.laplacian_matrix(G).todense())

    # eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(Q)
    _, V = low_rank_matrix(Q, eigvals, eigvecs, r=r)

    print(f"Q shape: {Q.shape}")
    print(f"V shape: {V.shape}")

    # save
    q_path = os.path.join(out_dir, f"Q_gset{gset}.npy")
    v_path = os.path.join(out_dir, f"V_gset{gset}.npy")

    np.save(q_path, Q)
    np.save(v_path, V)

    print(f"Saved Q to {q_path}")
    print(f"Saved V (rank {r}) to {v_path}")

if __name__ == "__main__":
    main()
