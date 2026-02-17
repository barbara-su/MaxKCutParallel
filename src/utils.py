from __future__ import annotations

import networkx as nx
import numpy as np
import math
from scipy import sparse
from itertools import product
import os
import random

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

def compute_vtilde(V):
    """
    Compute the V_tilde matrix for M=3 (Max-3-Cut problem).

    This function implements the transformation described in equation (14):
    1. First creates the V_hat matrix by rotating V by angles π/3, π, 5π/3
    2. Then concatenates the real and imaginary parts to form V_tilde

    Parameters
    ----------
    V : ndarray (n, r)
        Complex matrix where Q = V @ V†, with dimensions (n, r)
        where n is the number of vertices and r is the rank

    Returns
    -------
    V_tilde : ndarray (3n, 2r)
        Real matrix representing the decision boundaries
    """
    import numpy as np

    n, r = V.shape  # n = number of vertices, r = rank

    # Initialize V_tilde matrix
    V_tilde = np.zeros((3*n, 2*r))

    # Step 1: Compute the V_hat matrix
    # V_hat = [e^(-jπ/3)V; -V; e^(-j5π/3)V]

    # Define the rotation angles θ_k = {π/3, π, 5π/3}
    thetas = np.array([np.pi/3, np.pi, 5*np.pi/3])

    # Initialize V_hat with dimensions (3p, r)
    V_hat = np.zeros((3*n, r), dtype=complex)

    # Fill V_hat with rotated versions of V
    for k in range(3):
        rotation = np.exp(-1j * thetas[k])
        V_hat[k*n:(k+1)*n, :] = rotation * V

    # Step 2: Construct V_tilde by concatenating real and imaginary parts

    # Interleave real and imaginary parts
    # V_tilde = [Re{V_hat_:,1} Im{V_hat_:,1} Re{V_hat_:,2} Im{V_hat_:,2} ... Re{V_hat_:,r} Im{V_hat_:,r}]
    for j in range(r):
        V_tilde[:, 2*j] = np.real(V_hat[:, j])     # Real part in even columns
        V_tilde[:, 2*j+1] = np.imag(V_hat[:, j])   # Imaginary part in odd columns

    return V_tilde

def find_intersection(VI):
    """
    Find the intersection point of hyperplanes defined by the rows of VI.

    Parameters
    ----------
    VI : ndarray
        Matrix where each row represents a hyperplane in the space.

    Returns
    -------
    c_tilde : ndarray
        Normalized vector in the null space of VI, representing the
        intersection point of the hyperplanes.
    """
    # One SVD pass: use singular values for rank check and right-singular vector
    # for the null direction.
    rows, cols = VI.shape
    U, S, Vh = np.linalg.svd(VI, full_matrices=True)

    min_dim = min(rows, cols)
    if S.size > 0:
        sv_tol = max(rows, cols) * np.finfo(S.dtype).eps * S[0]
        matrix_rank = int(np.sum(S > sv_tol))
    else:
        matrix_rank = 0

    if matrix_rank != min_dim:
        raise ValueError("VI matrix is not full rank")

    # With full rank and rows >= cols, null space is empty.
    if rows >= cols:
        raise ValueError("No intersection found - null space is empty")

    c_tilde = Vh[-1, :]
    nrm = np.linalg.norm(c_tilde)
    if nrm <= 1e-12:
        raise ValueError("No stable intersection, near-zero null vector")
    return c_tilde / nrm

def convert_ctilde_to_complex(c_tilde, r):
    """
    Convert c_tilde from real representation to complex form.

    Parameters
    ----------
    c_tilde : ndarray
        Real vector representing the intersection point.
    r : int
        Rank of the approximation.

    Returns
    -------
    c : ndarray
        Complex vector representation.
    """
    c_tilde = np.asarray(c_tilde)
    out_dtype = np.result_type(c_tilde.dtype, np.complex64)
    c = np.zeros(r, dtype=out_dtype)
    m = min(r, c_tilde.size // 2)
    if m > 0:
        c[:m] = c_tilde[0 : 2 * m : 2] + 1j * c_tilde[1 : 2 * m : 2]
    return c

def determine_phi_sign_c(c_tilde, eps: float = 1e-10):
    """
    Given the real hyperspherical coordinates c_tilde, compute the angles phi and
    determine the sign correction factor.

    Handles numerical precision issues gracefully.
    """
    c_tilde = np.asarray(c_tilde)
    D = c_tilde.size  # Should be 2*r
    phi = np.zeros(D - 1, dtype=c_tilde.dtype)

    # Track product of cosines incrementally (O(D) instead of O(D^2)).
    prod_cos = 1.0
    for i in range(D - 1):
        if abs(prod_cos) < eps:
            phi[i] = 0.0
            continue

        arg = np.clip(c_tilde[i] / prod_cos, -1.0, 1.0)
        phi[i] = np.arcsin(arg)

        if i < D - 2:
            ci = np.cos(phi[i])
            if abs(ci) < eps:
                prod_cos = 0.0
            else:
                prod_cos *= ci

    # Determine the sign correction (same logic as before, but reuse cos).
    j = D - 2
    if phi[j] == 0 or c_tilde[j] == 0:
        sign_c = 1
    else:
        cos_j = np.cos(phi[j])
        if abs(cos_j) < eps:
            sign_c = 1
        else:
            tan_j = np.sin(phi[j]) / cos_j
            sign_c = np.sign(tan_j * c_tilde[j] * c_tilde[j + 1])

    return phi, sign_c

def get_row_mapping(n, K):
    """
    Create mapping from V_tilde indices to original V indices.

    Parameters
    ----------
    n : int
        Number of rows in V
    K : int
        Number of partitions

    Returns
    -------
    mapping : dict
        Dictionary mapping V_tilde indices to (V_row, rotation) pairs
    inverse_mapping : dict
        Dictionary mapping V_row to list of V_tilde indices
    """
    mapping = {}
    inverse_mapping = {}

    for i in range(n):
        inverse_mapping[i] = []
        for j in range(K):
            v_tilde_idx = i*K + j
            mapping[v_tilde_idx] = (i, j)
            inverse_mapping[i].append(v_tilde_idx)

    return mapping, inverse_mapping

def construct_ctilde_from_phi(phi_reduced, r, K, sin_last=None, cos_last=None):
    """
    Construct the c_tilde vector with specific phi angles and the last angle fixed at pi/K.

    Parameters
    ----------
    phi_reduced : ndarray
        The first 2*r-2 phi angles
    r : int
        Rank of the approximation
    K : int
        Number of partitions

    Returns
    -------
    c_tilde : ndarray
        The constructed c_tilde vector with last angle fixed at pi/K
    """
    phi_reduced = np.asarray(phi_reduced)
    c_tilde = np.zeros(2 * r, dtype=phi_reduced.dtype if phi_reduced.size > 0 else float)

    if sin_last is None or cos_last is None:
        ang = np.pi / K
        sin_last = np.sin(ang)
        cos_last = np.cos(ang)

    if phi_reduced.size > 0:
        cosv = np.cos(phi_reduced)
        sinv = np.sin(phi_reduced)
        prefix = np.empty(phi_reduced.size + 1, dtype=phi_reduced.dtype)
        prefix[0] = 1.0
        prefix[1:] = np.cumprod(cosv)
        c_tilde[: phi_reduced.size] = prefix[: phi_reduced.size] * sinv
        prod_cos = prefix[-1]
    else:
        prod_cos = 1.0

    c_tilde[2 * r - 2] = prod_cos * sin_last
    c_tilde[2 * r - 1] = prod_cos * cos_last
    return c_tilde

def find_intersection_fixed_angle(VI_minus, r, K, sin_last=None, cos_last=None):
    """
    Find the intersection point with the last angle fixed at pi/K.

    Parameters
    ----------
    VI_minus : ndarray
        Matrix with one row removed from VI
    r : int
        Rank of the approximation
    K : int
        Number of partitions

    Returns
    -------
    c_tilde : ndarray
        The constructed c_tilde vector with fixed last angle
    """
    if sin_last is None or cos_last is None:
        ang = np.pi / K
        sin_last = np.sin(ang)
        cos_last = np.cos(ang)

    # Extract the constraints for the first 2r-2 variables
    A = VI_minus[:, :2*r-2]

    # Extract the constraints for the last two variables
    tail = VI_minus[:, 2*r-2:]
    if tail.shape[1] == 2:
        b = -(tail[:, 0] * sin_last + tail[:, 1] * cos_last)
    else:
        b = -tail @ np.array([sin_last, cos_last], dtype=VI_minus.dtype)

    # Solve the system A * phi = b
    try:
        if A.shape[0] == A.shape[1]:
            phi_reduced = np.linalg.solve(A, b)
        else:
            phi_reduced, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        try:
            phi_reduced, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError as e:
            raise ValueError("Could not find intersection with fixed angle") from e

    # Construct c_tilde from the reduced phi
    c_tilde = construct_ctilde_from_phi(
        phi_reduced,
        r,
        K,
        sin_last=sin_last,
        cos_last=cos_last,
    )

    return c_tilde


def complex_to_partition(z, K=3):
    """
    Convert a complex vector of K-th roots of unity to a partition vector (0,1,...,K-1).

    Parameters
    ----------
    z : ndarray
        Complex vector where each element is approximately a K-th root of unity.
    K : int, default=3
        Number of partitions.

    Returns
    -------
    partition : ndarray
        Integer vector where each element is in {0,1,...,K-1}.
    """
    n = len(z)
    partition = np.zeros(n, dtype=int)

    # K-th roots of unity
    roots = np.exp(1j * 2 * np.pi * np.arange(K) / K)

    # For each vertex, find the closest root of unity
    for i in range(n):
        distances = np.abs(z[i] - roots)
        partition[i] = np.argmin(distances)

    return partition


def generate_debug_QV(n=10, rank=1, seed=42):
    """
    Generate a random symmetric PSD matrix Q = A A^T of (intended) rank = rank,
    and return Q along with its (complex) factor V = A.

    If 1 <= rank <= n, then rank(Q) = rank with probability 1 under Gaussian A.
    """
    if not (isinstance(rank, int) and rank >= 1):
        raise ValueError("rank must be a positive integer.")
    if rank > n:
        raise ValueError("rank must satisfy rank <= n.")

    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, rank))   # shape (n, rank)
    Q = A @ A.T                      # shape (n, n), symmetric PSD
    V = A.astype(complex)            # shape (n, rank)

    return Q, V

def opt_K_cut(Q, K=3):
    """
    Optimal max-k cut computation
    """
    n = Q.shape[0]  # Number of vertices
    groups = range(K)  # Possible colors/groups (0 to K-1)
    candidate_colors = list(product(groups, repeat=n))
    best_score = float('-inf')  # Initialize with worst possible score
    best_colors = None  # Will store the optimal coloring

    # Evaluate each possible coloring
    for colors in candidate_colors:
        zs = np.exp(2 * np.pi * 1j * np.array(colors) / K)
        score = zs.conj() @ Q @ zs

        if np.real(score) > best_score:
            best_score = np.real(score) 
            best_colors = colors

    return best_score, best_colors


def set_numpy_precision(precision: int):
    """
    Simple "global" precision switch.
    - 16 -> float16 / complex64 (complex32 is not supported in NumPy)
    - 32 -> float32 / complex64
    - 64 -> float64 / complex128
    """
    if precision not in (16, 32, 64):
        raise ValueError("--precision must be one of {16, 32, 64}")

    if precision == 16:
        float_dtype = np.float16
        complex_dtype = np.complex64
    elif precision == 32:
        float_dtype = np.float32
        complex_dtype = np.complex64
    else:
        float_dtype = np.float64
        complex_dtype = np.complex128

    return float_dtype, complex_dtype


def set_seed(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except Exception:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Only touch cudnn via torch.backends inside the function body
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
