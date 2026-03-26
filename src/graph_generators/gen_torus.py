"""
Generate toroidal grid graph instances for Max-K-Cut experiments.
Creates a 2D torus (periodic grid) with optional edge weight perturbations.
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def generate_torus_instance(p, q, rank=2, epsilon=0.01, seed=42):
    """Generate a p×q toroidal grid graph.

    Args:
        p, q: grid dimensions (n = p*q nodes)
        rank: number of eigenvectors to extract
        epsilon: random weight perturbation magnitude (0 = unweighted)
        seed: random seed for perturbations
    Returns:
        Q (n×n), V (n×rank) complex
    """
    import networkx as nx

    n = p * q
    G = nx.grid_2d_graph(p, q, periodic=True)

    # Add small random weight perturbations to break spectral degeneracy
    rng = np.random.RandomState(seed)
    for u, v in G.edges():
        G[u][v]['weight'] = 1.0 + epsilon * rng.randn()

    # Compute weighted Laplacian
    L = nx.laplacian_matrix(G, weight='weight').toarray().astype(np.float64)

    # Top-r eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(L)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    V = eigvecs[:, :rank] * np.sqrt(np.maximum(eigvals[:rank], 0))
    V_complex = V.astype(np.complex128)

    print(f"Torus: {p}x{q} = {n} nodes, {G.number_of_edges()} edges")
    print(f"Top eigenvalues: {eigvals[:rank+2]}")
    print(f"Eigengap (lambda_{rank} - lambda_{rank+1}): {eigvals[rank-1] - eigvals[rank]:.6f}")
    print(f"Top-{rank} energy: {sum(eigvals[:rank])/sum(np.maximum(eigvals,0))*100:.1f}%")

    return L, V_complex


# Grid dimensions for target n values (p ~ aspect_ratio * q)
TORUS_DIMS = {
    250: (21, 12),    # 252
    500: (28, 18),    # 504
    750: (36, 21),    # 756
    1000: (42, 24),   # 1008
    1250: (50, 25),   # 1250
    1500: (50, 30),   # 1500
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="Target number of nodes")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=0.01, help="Weight perturbation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="test_instances/torus")
    args = parser.parse_args()

    if args.n not in TORUS_DIMS:
        # Find closest
        closest = min(TORUS_DIMS.keys(), key=lambda x: abs(x - args.n))
        print(f"No exact dims for n={args.n}, using n={closest}")
        args.n = closest

    p, q = TORUS_DIMS[args.n]
    actual_n = p * q

    os.makedirs(args.outdir, exist_ok=True)

    Q, V = generate_torus_instance(p, q, args.rank, args.epsilon, args.seed)

    tag = f"torus_{actual_n}_seed_{args.seed}"
    q_path = os.path.join(args.outdir, f"Q_{tag}.npy")
    v_path = os.path.join(args.outdir, f"V_{tag}.npy")

    np.save(q_path, Q)
    np.save(v_path, V)
    print(f"Saved: {q_path} ({Q.shape}), {v_path} ({V.shape})")


if __name__ == "__main__":
    main()
