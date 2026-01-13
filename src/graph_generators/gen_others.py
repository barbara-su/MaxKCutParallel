import argparse
import os
import numpy as np
import networkx as nx
from gen_v import gen_V_given_Q

def main():
    parser = argparse.ArgumentParser(
        description="Generate multiple random regular graphs and low-rank eigenvector matrices."
    )
    parser.add_argument(
        "--n", type=int, default=1000,
        help="Number of nodes in the graph."
    )
    parser.add_argument(
        "--m", type=int, default=100,
        help="Degree of each node."
    )
    parser.add_argument(
        "--p", type=float, default=0.5,
        help="probability of connection"
    )
    parser.add_argument(
        "--q", type=int, default=0,
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
    m = args.m
    p = args.p
    q = args.q
    r = args.rank
    out_dir = args.out_dir
    num_seeds = args.num_seeds

    os.makedirs(out_dir, exist_ok=True)

    for seed in range(num_seeds):
        print(f"Generating blob graph with n={n}, p={p}")

        size_of_blob = int(n/3)
        G1 = nx.erdos_renyi_graph(size_of_blob, p)
        G2 = nx.erdos_renyi_graph(size_of_blob, p)
        G3 = nx.erdos_renyi_graph(size_of_blob, p)
        
        # Relabel nodes to combine them
        G = nx.disjoint_union(G1, G2)
        G = nx.disjoint_union(G, G3)
        
        # 2. Add the "Bridges" (Weak Links)
        # Connect last node of G1 to first of G2
        # Connect last node of G2 to first of G3
        # Connect last node of G3 to first of G1
        
        # Nodes are 0 to 3*n-1
        bridge_edges = [
            (size_of_blob-1, size_of_blob),           # Blob 1 -> Blob 2
            (2*size_of_blob-1, 2*size_of_blob),       # Blob 2 -> Blob 3
            (3*size_of_blob-1, 0)          # Blob 3 -> Blob 1
        ]
        
        G.add_edges_from(bridge_edges)

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
