import argparse
import numpy as np
import networkx as nx
from utils import *

def main():
    parser = argparse.ArgumentParser(description="Generate G(n, p) graph and rank-1 eigenvector matrix.")
    parser.add_argument("--n", type=int, default=10000, help="Number of nodes in the graph.")
    parser.add_argument("--p", type=float, default=0.5, help="Edge probability for G(n, p).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    n = args.n
    p = args.p
    seed = args.seed

    print(f"Generating graph with n = {n}, p = {p}, seed = {seed}")

    # Generate graph
    G = nx.fast_gnp_random_graph(n=n, p=p, seed=seed)
    Q = np.array(nx.laplacian_matrix(G).todense())

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(Q)
    _, V = low_rank_matrix(Q, eigvals, eigvecs, r=1)

    # Save
    np.save(f"graphs/Q_{n}.npy", Q)
    np.save(f"graphs/V_{n}.npy", V)

    print("Saved Q and V.")

if __name__ == "__main__":
    main()
