

import os
                   # another optimisation library
import networkx as nx                    # graph generation / algorithms
import numpy as np

import matplotlib.pyplot as plt

import math

import pandas as pd
from scipy import sparse

def low_rank_matrix(Q, eigvals, eigvecs, r: int = 2):
    """
    Construct a rank-r approximation of Q from its largest eigenvalues.
    Works correctly for both real and complex PSD matrices.

    Parameters
    ----------
    Q : ndarray (n,n)
        The matrix to approximate, must be Hermitian (or symmetric for real matrices)
    eigvals, eigvecs : ndarray
        Full eigendecomposition of Q.
    r : int
        Desired rank of the approximation.

    Returns
    -------
    Q_hat : ndarray
        Rank-r approximation  Σ_{i≤r} λ_i q_i q_i†
    V     : ndarray (n,r)
        Matrix whose columns are √λ_i q_i  (useful for low-rank SDP forms).
    """
    import numpy as np

    # Check that eigenvalues are real (as they should be for a Hermitian matrix)
    if not np.allclose(np.imag(eigvals), 0):
        raise ValueError("Eigenvalues should be real for a Hermitian matrix")

    # Convert to real if the imaginary parts are effectively zero
    eigvals = np.real(eigvals)

    # indices that would sort eigenvalues in decreasing order
    ldas_idxs = np.argsort(-eigvals)  # Negative to sort in decreasing order

    # top-r eigenvalues / eigenvectors
    ldas = eigvals[ldas_idxs[:r]]

    # Ensure eigenvalues are non-negative for PSD matrix
    if np.any(ldas < -1e-10):  # Allow small negative values due to numerical precision
        raise ValueError("Found significant negative eigenvalues in supposedly PSD matrix")

    # Zero out any tiny negative eigenvalues due to numerical issues
    ldas = np.maximum(ldas, 0)

    lda_mat = np.diag(ldas)
    lda_sqrt = np.sqrt(lda_mat)

    # Extract top-r eigenvectors as columns
    qs = eigvecs[:, ldas_idxs[:r]]

    # Construct V = U * sqrt(Λ)
    V = qs @ lda_sqrt

    # Explicit rank-r reconstruction using outer products
    Q_hat = np.zeros(Q.shape, dtype=complex if np.iscomplexobj(Q) else float)

    for i in range(r):
        q_i = qs[:, i].reshape(-1, 1)  # Column vector
        # For complex matrices, use conjugate transpose for the outer product
        if np.iscomplexobj(Q):
            op = q_i @ q_i.conj().T
        else:
            op = q_i @ q_i.T

        Q_hat += ldas[i] * op

    # Final check for numerical stability
    if np.iscomplexobj(Q_hat) and np.allclose(np.imag(Q_hat), 0, atol=1e-10):
        Q_hat = np.real(Q_hat)

    return Q_hat, V

def generate_Q(graph_param, size: int = 20, mode: str = 'reg', seed=42):
    """
    Sample a Laplacian matrix Q for various synthetic graph models.

    Parameters
    ----------
    graph_param : float | int
        Model-specific parameter (degree, probability, λ, …).
    size : int
        Number of vertices.
    mode : str
        Key specifying the random-graph family:
        - 'reg': Regular graph (param = degree)
        - 'weighted_reg': Weighted regular graph (param = degree)
        - 'erdos_renyi': Erdos-Renyi random graph (param = probability)
        - 'weighted_erdos_renyi': Weighted Erdos-Renyi graph (param = probability)
        - 'expander': Regular expander graph (param = degree)
        - 'weighted_expander': Weighted expander graph (param = degree)
        - 'spiked_wishart': Spiked Wishart model (param = spike strength)
        - 'power_law': Power-law spectrum (param = decay exponent)
        - 'random_gaus': Complex normal matrix (param = standard deviation)
        - 'hamiltonian': Quantum Hamiltonian-like matrix (param = interaction strength)
        - 'sparse_complex': Sparse complex matrix (param = sparsity percentage)
        - 'circulant': Circulant graph (param = connection distance)
        - 'barbell': Barbell graph (param = size of complete components)
        - 'small_world': Small-world graph (param = rewiring probability)
        - 'band_toeplitz': Banded Toeplitz matrix (param = bandwidth)
        - 'complex_community': Complex community structure (param = inter-community connection probability)
        - 'kronecker': Kronecker graph (param = scale factor)
        - 'hierarchical': Hierarchical network (param = branching factor)

    Returns
    -------
    Q : ndarray (n,n)
        Unweighted or weighted Laplacian or PSD matrix.
    """

    # --- 1. Regular graph -------------------------------------------------
    if mode == 'reg':
        if (size * graph_param) % 2 != 0:
            raise ValueError(f"Cannot create {graph_param}-regular graph with {size} nodes: product must be even")
        else:
            G = nx.random_regular_graph(graph_param, size, seed=None)
            Q = np.array(nx.laplacian_matrix(G).todense())

    # --- 2. Regular graph with random positive weights --------------------
    elif mode == 'weighted_reg':
        G = nx.random_regular_graph(graph_param, size, seed=None)
        for (u, v) in G.edges():
            G.edges[u,v]['weight'] = np.random.rand() * 10
        Q = np.array(nx.laplacian_matrix(G).todense())

    # --- 3. Erdos-Renyi ----------------------------------------------------
    elif mode == 'erdos_renyi':
        G = nx.erdos_renyi_graph(size, graph_param, seed)
        Q = np.array(nx.laplacian_matrix(G).todense())

    # --- 4. Erdos-Renyi weighted ------------------------------------------
    elif mode == 'weighted_erdos_renyi':
        G = nx.erdos_renyi_graph(size, graph_param)
        for (u, v) in G.edges():
            G.edges[u,v]['weight'] = np.random.rand() * 10
        Q = np.array(nx.laplacian_matrix(G).todense())

    # --- 5. Expander (d-regular) ------------------------------------------
    elif mode == 'expander':
        G = nx.random_regular_graph(graph_param, size)  # Approximate expander with regular graph
        # Make it more "expander-like" by ensuring connectivity
        while not nx.is_connected(G):
            G = nx.random_regular_graph(graph_param, size)
        Q = np.array(nx.laplacian_matrix(G).todense())

    # --- 6. Weighted expander ---------------------------------------------
    elif mode == 'weighted_expander':
        G = nx.random_regular_graph(graph_param, size)
        # Make it more "expander-like" by ensuring connectivity
        while not nx.is_connected(G):
            G = nx.random_regular_graph(graph_param, size)
        for (u, v) in G.edges():
            G.edges[u,v]['weight'] = np.random.rand() * 10
        Q = np.array(nx.laplacian_matrix(G).todense())

    # --- 7. Spiked Wishart model ------------------------------------------
    elif mode == 'spiked_wishart':
        L = np.random.randn(size, size)
        W = np.dot(L, L.T)

        x = np.random.randn(size)
        Q = W + graph_param * np.outer(x, x)

    # --- 8. Power-law spectrum --------------------------------------------
    elif mode == 'power_law':
        L = np.random.randn(size, size)
        W = np.dot(L, L.T)

        eigvals, eigvecs = np.linalg.eigh(W)
        power_law_eigvals = 10 * np.power(np.arange(1, size + 1), -graph_param)
        Q = eigvecs @ np.diag(power_law_eigvals) @ eigvecs.T

    # --- 9. Complex normal (optional) -------------------------------------
    elif mode == 'random_gaus':
        Re = np.random.randn(size, size) * graph_param
        Im = np.random.randn(size, size) * graph_param

        A = Re + Im * 1j
        Q = A @ A.conj().T

    # --- 10. Quantum Hamiltonian-like matrix ------------------------------
    elif mode == 'hamiltonian':
        # Create a sparse Hermitian matrix (similar to tight-binding models)
        H = np.zeros((size, size), dtype=complex)

        # Diagonal terms (site energies)
        for i in range(size):
            H[i, i] = np.random.normal(0, graph_param)

        # Off-diagonal terms (hopping terms)
        for i in range(size):
            for j in range(i+1, min(i+3, size)):  # Nearest-neighbor interactions
                coupling = np.random.normal(0, graph_param) + 1j * np.random.normal(0, graph_param/2)
                H[i, j] = coupling
                H[j, i] = np.conjugate(coupling)  # Ensure Hermitian

        # Add a global phase (breaking time-reversal symmetry like magnetic field)
        phase = np.exp(1j * graph_param * np.pi/4)
        for i in range(size-1):
            H[i, i+1] *= phase
            H[i+1, i] = np.conjugate(H[i, i+1])

        # Make it PSD
        Q = H @ H.conj().T

    # --- 11. Sparse complex matrix ----------------------------------------
    elif mode == 'sparse_complex':
        # Create a sparse complex matrix with controlled sparsity
        sparsity = min(max(0.01, graph_param), 0.99)  # Keep param between 0.01 and 0.99
        nnz = int((1 - sparsity) * size * size)  # Number of non-zero elements

        # Create index arrays for sparse matrix
        row_indices = np.random.randint(0, size, nnz)
        col_indices = np.random.randint(0, size, nnz)
        real_values = np.random.randn(nnz)
        imag_values = np.random.randn(nnz)

        # Create sparse matrix
        sparse_matrix = sparse.coo_matrix(
            (real_values + 1j * imag_values, (row_indices, col_indices)),
            shape=(size, size)
        ).toarray()

        # Make it Hermitian and PSD
        Q = sparse_matrix @ sparse_matrix.conj().T

    # --- 12. Circulant graph ----------------------------------------------
    elif mode == 'circulant':
        # Create a circulant graph where each node connects to nodes
        # at distance up to graph_param
        k = int(graph_param)  # Number of neighbors on each side
        connections = list(range(1, min(k+1, size//2 + 1)))
        G = nx.circulant_graph(size, connections)
        Q = np.array(nx.laplacian_matrix(G).todense())

    # --- 13. Barbell graph ------------------------------------------------
    elif mode == 'barbell':
        # Create a barbell graph - two complete graphs connected by a path
        m = max(2, int(graph_param))  # Size of each complete graph
        if 2*m < size:
            G = nx.barbell_graph(m, size - 2*m)
        else:
            G = nx.barbell_graph(size//2, 0)
        Q = np.array(nx.laplacian_matrix(G).todense())

    # --- 14. Small-world graph --------------------------------------------
    elif mode == 'small_world':
        # Create a Watts-Strogatz small-world graph
        k = min(6, size-1)  # Number of nearest neighbors (must be even)
        if k % 2 == 1:
            k -= 1
        p = graph_param  # Rewiring probability
        G = nx.watts_strogatz_graph(size, k, p)
        Q = np.array(nx.laplacian_matrix(G).todense())

    # --- 15. Banded Toeplitz matrix ---------------------------------------
    elif mode == 'band_toeplitz':
        # Create a banded Toeplitz matrix
        bandwidth = max(1, int(graph_param))

        # Create band elements
        band_elements = np.random.rand(bandwidth) + 0.5

        # Create diagonal and off-diagonal bands
        diagonals = [2 * sum(band_elements) * np.ones(size)]  # Main diagonal

        # Add off-diagonal bands
        positions = [0]  # Position of main diagonal

        for b in range(1, bandwidth + 1):
            band = -band_elements[b-1] * np.ones(size - b)
            diagonals.append(band)
            diagonals.append(band)  # Same for both off-diagonals
            positions.append(b)
            positions.append(-b)

        # Create the Toeplitz matrix
        Q = sparse.diags(diagonals, positions).toarray()

    # --- 16. Complex community structure ----------------------------------
    elif mode == 'complex_community':
        # Create a graph with community structure
        num_communities = max(2, size // 10)
        community_sizes = [size // num_communities] * num_communities

        # Adjust last community size to match total size
        community_sizes[-1] += size - sum(community_sizes)

        # Probability of connections within community is high
        p_in = 0.7
        # Probability between communities is controlled by graph_param
        p_out = min(max(0.01, graph_param), 0.5)

        G = nx.random_partition_graph(community_sizes, p_in, p_out)

        # Create a weighted Laplacian with complex weights for inter-community edges
        A = np.zeros((size, size), dtype=complex)

        for u, v in G.edges():
            weight = 1.0
            # Add complex phase to inter-community edges
            if G.nodes[u].get('block', 0) != G.nodes[v].get('block', 0):
                phase = np.random.uniform(0, 2*np.pi)
                weight = np.exp(1j * phase)
            A[u, v] = weight
            A[v, u] = np.conjugate(weight)

        # Create Laplacian: L = D - A
        D = np.diag(np.sum(np.abs(A), axis=1))
        Q = D - A

    # --- 17. Kronecker graph ---------------------------------------------
    elif mode == 'kronecker':
        # Create a Kronecker graph
        # Start with a small initiator matrix
        P = np.array([[0.9, 0.5], [0.5, 0.3]])

        # Scale parameter controls the number of Kronecker products
        scale = max(1, int(graph_param))
        scale = min(scale, int(np.log2(size)))  # Limit scale to avoid huge matrices

        # Perform Kronecker product iteratively
        kronecker_P = P
        for _ in range(scale - 1):
            kronecker_P = np.kron(kronecker_P, P)

        # Ensure matrix is not larger than size x size
        kronecker_size = 2 ** scale
        if kronecker_size > size:
            kronecker_P = kronecker_P[:size, :size]

        # Create a graph from this probability matrix
        G = nx.from_numpy_array(kronecker_P, create_using=nx.Graph)

        # Get Laplacian
        Q = np.array(nx.laplacian_matrix(G).todense())

    # --- 18. Hierarchical network ----------------------------------------
    elif mode == 'hierarchical':
        # Create a hierarchical network
        branching = max(2, int(graph_param))  # Branching factor
        depth = max(1, int(math.log(size, branching)))  # Depth of tree

        # Create a balanced tree
        G = nx.balanced_tree(branching, depth)

        # If the tree is too large, take a subgraph
        if G.number_of_nodes() > size:
            nodes = list(G.nodes())[:size]
            G = G.subgraph(nodes)

        # If the tree is too small, add random edges until we have size nodes
        while G.number_of_nodes() < size:
            G.add_node(G.number_of_nodes())
            # Add edge to a random existing node
            G.add_edge(G.number_of_nodes()-1, np.random.randint(0, G.number_of_nodes()-1))

        # Get Laplacian
        Q = np.array(nx.laplacian_matrix(G).todense())

    else:
        raise NotImplementedError(f"{mode} not implemented")

    return Q
