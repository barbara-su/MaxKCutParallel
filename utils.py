

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

def generate_Q(graph_param, size: int = 20, mode: str = 'reg'):
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
        G = nx.erdos_renyi_graph(size, graph_param)
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

# Updated analyze_matrix_properties function with a fixed radar chart implementation
def analyze_matrix_properties(size=20, display_count=3, save_dir='./', show_plots=True):
    """
    Analyze and visualize properties of different Q matrices.

    Parameters
    ----------
    size : int
        Size of the matrices to generate
    display_count : int
        Number of different parameter values to test for each model
    save_dir : str
        Directory to save the generated plots
    show_plots : bool
        Whether to display plots (in addition to saving them)

    Returns
    -------
    matrices : dict
        Dictionary of generated matrices for each model
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.rcParams.update({'font.size': 18})

    # Define the graph models to test
    models = [
        ('reg', 4),
        ('weighted_reg', 4),
        ('erdos_renyi', 0.3),
        ('weighted_erdos_renyi', 0.3),
        ('expander', 3),
        ('weighted_expander', 3),
        ('spiked_wishart', 5.0),
        ('power_law', 1.0),
        ('random_gaus', 1.0),
        ('hamiltonian', 1.0),
        ('sparse_complex', 0.8),
        ('circulant', 2),
        ('barbell', 5),
        ('small_world', 0.1),
        ('band_toeplitz', 3),
        ('complex_community', 0.1),
        ('kronecker', 2),
        ('hierarchical', 3)
    ]

    # Create different parameter values for each model
    model_params = {}
    for model, default_param in models:
        if model in ['reg', 'weighted_reg', 'expander', 'weighted_expander', 'circulant', 'band_toeplitz', 'kronecker', 'hierarchical']:
            # Integer parameters
            if model in ['reg', 'weighted_reg', 'expander', 'weighted_expander'] and size % 2 != 0:
                params = [3, 5, 7, 9, 11][:display_count]  # Odd size needs odd degree
            else:
                params = [2, 4, 6, 8, 10][:display_count]
        elif model in ['barbell']:
            params = [max(2, int(size/i)) for i in [4, 6, 8, 10, 12][:display_count]]
        elif model in ['erdos_renyi', 'weighted_erdos_renyi', 'small_world', 'complex_community']:
            # Probability parameters
            params = np.linspace(0.1, 0.9, display_count)
        elif model in ['sparse_complex']:
            # Sparsity parameters (higher = more sparse)
            params = np.linspace(0.5, 0.95, display_count)
        elif model in ['power_law']:
            # Decay parameters
            params = np.linspace(0.5, 2.5, display_count)
        elif model in ['spiked_wishart', 'hamiltonian']:
            # Scaling parameters
            params = np.linspace(1.0, 5.0, display_count)
        elif model in ['random_gaus']:
            # Standard deviation parameters
            params = np.linspace(0.1, 2.0, display_count)
        else:
            params = [default_param]

        model_params[model] = params

    # Generate representative matrices for each model (using default parameters)
    matrices = {}
    for model, default_param in models:
        try:
            matrices[model] = generate_Q(default_param, size, model)
        except Exception as e:
            print(f"Error generating matrix for {model}: {e}")
            matrices[model] = np.eye(size)  # Fallback to identity matrix

    # 1. Visualize sparsity patterns
    plt.figure(figsize=(20, 16))
    plt.suptitle('Sparsity Patterns of Different Matrix Types', fontsize=30)

    rows = (len(models) + 3) // 4  # Calculate number of rows needed (4 cols)

    for i, (model, _) in enumerate(models):
        plt.subplot(rows, 4, i+1)

        if np.iscomplexobj(matrices[model]):
            # For complex matrices, show magnitude
            plt.spy(np.abs(matrices[model]), precision=1e-10, markersize=3)
            plt.title(f'{model} (complex)')
        else:
            plt.spy(matrices[model], precision=1e-10, markersize=3)
            plt.title(f'{model}')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for main title
    plt.savefig(os.path.join(save_dir, 'sparsity_patterns.png'), dpi=300)
    if not show_plots:
        plt.close()

    # 2. Plot eigenvalue spectrum for each model
    complex_models = ['random_gaus', 'hamiltonian', 'sparse_complex', 'complex_community']
    real_models = [m for m, _ in models if m not in complex_models]

    # 2.1 Real eigenvalue spectrum
    plt.figure(figsize=(20, 16))
    plt.suptitle('Eigenvalue Spectrum of Real Matrices', fontsize=30)

    rows = (len(real_models) + 3) // 4  # Calculate rows (4 cols)

    for i, model in enumerate(real_models):
        plt.subplot(rows, 4, i+1)

        for param in model_params[model]:
            try:
                Q = generate_Q(param, size, model)
                eigvals = np.sort(np.real(np.linalg.eigvals(Q)))
                plt.plot(eigvals, label=f'param={param:.2f}')
            except Exception as e:
                print(f"Error with {model}, param={param}: {e}")

        plt.title(f'{model}')
        plt.xlabel('Index')
        plt.ylabel('Eigenvalue')
        plt.legend()
        plt.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for main title
    plt.savefig(os.path.join(save_dir, 'eigenvalue_spectrum_real.png'), dpi=300)
    if not show_plots:
        plt.close()

    # 2.2 Complex eigenvalue scatterplots
    plt.figure(figsize=(15, 10))
    plt.suptitle('Complex Eigenvalues in Complex Plane', fontsize=30)

    for i, model in enumerate(complex_models):
        plt.subplot(2, 2, i+1)

        for param in model_params[model][:3]:  # Limit to 3 parameters for clarity
            try:
                Q = generate_Q(param, size, model)
                eigvals = np.linalg.eigvals(Q)

                plt.scatter(np.real(eigvals), np.imag(eigvals), label=f'param={param:.2f}', alpha=0.7)
            except Exception as e:
                print(f"Error with {model}, param={param}: {e}")

        plt.title(f'{model}')
        plt.xlabel('Real part')
        plt.ylabel('Imaginary part')
        plt.legend()
        plt.grid(True)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for main title
    plt.savefig(os.path.join(save_dir, 'complex_eigenvalues.png'), dpi=300)
    if not show_plots:
        plt.close()

    # 3. Eigenvalue decay patterns (log scale)
    plt.figure(figsize=(20, 16))
    plt.suptitle('Eigenvalue Decay Patterns (Log Scale)', fontsize=30)

    rows = (len(models) + 3) // 4  # Calculate rows (4 cols)

    for i, (model, default_param) in enumerate(models):
        plt.subplot(rows, 4, i+1)

        try:
            Q = generate_Q(default_param, size, model)
            eigvals = np.sort(np.abs(np.linalg.eigvals(Q)))[::-1]  # Sort by magnitude, descending
            plt.semilogy(eigvals, marker='o', markersize=3)

            plt.title(f'{model}')
            plt.xlabel('Index')
            plt.ylabel('|Eigenvalue| (log)')
            plt.grid(True)
        except Exception as e:
            print(f"Error plotting eigenvalue decay for {model}: {e}")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for main title
    plt.savefig(os.path.join(save_dir, 'eigenvalue_decay.png'), dpi=300)
    if not show_plots:
        plt.close()

    # 4. Visualize complex matrices
    plt.figure(figsize=(15, 15))
    plt.suptitle('Complex Matrix Visualization', fontsize=30)

    for i, model in enumerate(complex_models):
        # Magnitude
        plt.subplot(4, 3, 3*i+1)
        im = plt.imshow(np.abs(matrices[model]), cmap='viridis')
        plt.colorbar(im)
        plt.title(f'{model}: Magnitude')

        # Real part
        plt.subplot(4, 3, 3*i+2)
        im = plt.imshow(np.real(matrices[model]), cmap='RdBu', vmin=-5, vmax=5)
        plt.colorbar(im)
        plt.title(f'{model}: Real Part')

        # Imaginary part
        plt.subplot(4, 3, 3*i+3)
        im = plt.imshow(np.imag(matrices[model]), cmap='RdBu', vmin=-5, vmax=5)
        plt.colorbar(im)
        plt.title(f'{model}: Imaginary Part')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for main title
    plt.savefig(os.path.join(save_dir, 'complex_matrix_visualization.png'), dpi=300)
    if not show_plots:
        plt.close()

    # 5. Spider plot of matrix properties
    properties = ['Rank', 'Eigenvalue Spread', 'Sparsity', 'Symmetry', 'Condition Number']

    results = {}

    for model, default_param in models:
        Q = matrices[model]

        # Compute properties
        rank = np.linalg.matrix_rank(Q)
        eigvals = np.real(np.linalg.eigvals(Q))
        eig_spread = max(eigvals) - min(eigvals)
        sparsity = 1.0 - (np.count_nonzero(Q) / (size * size))
        symmetry = np.linalg.norm(Q - Q.T.conj()) / np.linalg.norm(Q) if np.linalg.norm(Q) > 0 else 0
        # For condition number, use pseudoinverse if matrix is singular
        if rank < size:
            cond = 1e5  # High value for singular matrices
        else:
            cond = min(1e5, np.linalg.cond(Q))  # Cap at 1e5 to avoid inf

        results[model] = {
            'Rank': rank / size,  # Normalize to [0,1]
            'Eigenvalue Spread': min(1.0, eig_spread / (size * 10)),  # Normalize with cap
            'Sparsity': sparsity,
            'Symmetry': 1.0 - symmetry,  # 1 = perfectly symmetric
            'Condition Number': 1.0 - min(1.0, np.log10(cond) / 5.0)  # Normalize log(cond) with cap
        }

    # Fixed Radar Chart Implementation
    def plot_radar_chart(model_set, title, save_path):
        """Create a radar chart for a set of models"""
        # Number of variables
        num_vars = len(properties)

        # Calculate angles for each property (in radians)
        angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()

        # Make the plot circular by appending the first angle at the end
        angles += angles[:1]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        # Use a colormap for multiple lines
        colors = plt.cm.tab20(np.linspace(0, 1, len(model_set)))

        # Plot each model
        for i, model in enumerate(model_set):
            # Get values for this model
            values = [results[model][prop] for prop in properties]

            # Make the plot circular by appending the first value at the end
            values += values[:1]

            # Plot values
            ax.plot(angles, values, color=colors[i], linewidth=2, label=model)
            ax.fill(angles, values, color=colors[i], alpha=0.1)

        # Set property labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(properties)

        # Add grid and legend
        ax.grid(True)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.title(title, fontsize=15, pad=20)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if not show_plots:
            plt.close()

    # Group models by type
    real_graph_models = ['reg', 'weighted_reg', 'erdos_renyi', 'weighted_erdos_renyi',
                         'expander', 'weighted_expander', 'circulant', 'barbell',
                         'small_world', 'hierarchical', 'kronecker']

    complex_matrix_models = ['random_gaus', 'hamiltonian', 'sparse_complex',
                            'spiked_wishart', 'power_law', 'band_toeplitz',
                            'complex_community']

    # Create radar charts
    plot_radar_chart(
        real_graph_models,
        'Matrix Properties: Graph-Based Models',
        os.path.join(save_dir, 'matrix_properties_radar_graph_based_models.png')
    )

    plot_radar_chart(
        complex_matrix_models,
        'Matrix Properties: Matrix-Based Models',
        os.path.join(save_dir, 'matrix_properties_radar_matrix_based_models.png')
    )

    # 6. Compare spectral properties across all models
    spectral_properties = {}

    for model, default_param in models:
        Q = matrices[model]
        eigvals = np.real(np.linalg.eigvals(Q))

        spectral_properties[model] = {
            'Mean': np.mean(eigvals),
            'Median': np.median(eigvals),
            'Min': np.min(eigvals),
            'Max': np.max(eigvals),
            'Std Dev': np.std(eigvals),
            'Gap': eigvals[-1] - eigvals[-2] if len(eigvals) > 1 else 0
        }

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(spectral_properties).T

    # Plot spectral properties comparison
    plt.figure(figsize=(20, 15))
    plt.suptitle('Spectral Properties Comparison', fontsize=16)

    for i, property_name in enumerate(['Mean', 'Median', 'Min', 'Max', 'Std Dev', 'Gap']):
        plt.subplot(2, 3, i+1)

        # Sort for better visualization
        sorted_df = df.sort_values(by=property_name)
        sorted_df[property_name].plot(kind='bar', color=plt.cm.viridis(np.linspace(0, 1, len(sorted_df))))

        plt.title(f'Spectral Property: {property_name}')
        plt.ylabel('Value')
        plt.grid(True, axis='y')
        plt.xticks(rotation=90)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for main title
    plt.savefig(os.path.join(save_dir, 'spectral_properties_comparison.png'), dpi=300)
    if not show_plots:
        plt.close()

    # 7. Rank vs. Parameter plots for selected models
    plt.figure(figsize=(15, 10))
    plt.suptitle('Rank vs. Parameter', fontsize=30)

    selected_models = ['power_law', 'sparse_complex', 'band_toeplitz', 'erdos_renyi']

    for i, model in enumerate(selected_models):
        plt.subplot(2, 2, i+1)

        ranks = []
        params = model_params[model]

        for param in params:
            try:
                Q = generate_Q(param, size, model)
                rank = np.linalg.matrix_rank(Q) / size  # Normalized rank
                ranks.append(rank)
            except Exception as e:
                print(f"Error computing rank for {model}, param={param}: {e}")
                ranks.append(0)

        plt.plot(params, ranks, 'o-', markersize=8)
        plt.title(f'Rank vs. Parameter: {model}')
        plt.xlabel('Parameter value')
        plt.ylabel('Normalized Rank')
        plt.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for main title
    plt.savefig(os.path.join(save_dir, 'rank_vs_parameter.png'), dpi=300)
    if not show_plots:
        plt.close()

    # 8. Visualize eigenvalue distribution histograms
    plt.figure(figsize=(15, 12))
    plt.suptitle('Eigenvalue Distribution Histograms', fontsize=16)

    # Select a diverse set of models
    diverse_models = ['reg', 'weighted_reg', 'erdos_renyi', 'power_law',
                      'random_gaus', 'hamiltonian', 'complex_community',
                      'band_toeplitz']

    for i, model in enumerate(diverse_models):
        plt.subplot(2, 4, i+1)

        try:
            Q = matrices[model]
            eigvals = np.real(np.linalg.eigvals(Q))

            plt.hist(eigvals, bins=20, alpha=0.7, color=plt.cm.viridis(i/len(diverse_models)))
            plt.title(f'{model}')
            plt.xlabel('Eigenvalue')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
        except Exception as e:
            print(f"Error plotting histogram for {model}: {e}")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for main title
    plt.savefig(os.path.join(save_dir, 'eigenvalue_histograms.png'), dpi=300)
    if not show_plots:
        plt.close()

    # Return the generated matrices for further inspection if needed
    return matrices

# Example usage:
matrices = analyze_matrix_properties(size=30, display_count=3, save_dir='./matrix_analysis', show_plots=True)
