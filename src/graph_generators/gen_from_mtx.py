"""
Load graphs from Matrix Market (.mtx) format and SNAP edge lists.
Computes Laplacian and top-r eigenvectors for Max-K-Cut experiments.

Supports:
- SuiteSparse Matrix Collection (.mtx files)
- SNAP edge lists (.txt files)
- Automatic download from URLs
"""
import argparse
import os
import sys
import time
import urllib.request

import numpy as np
from scipy import sparse
from scipy.io import mmread
from scipy.sparse.linalg import eigsh


def load_mtx(path):
    """Load adjacency matrix from Matrix Market file."""
    A = mmread(path)
    if sparse.issparse(A):
        A = A.tocsr()
    else:
        A = sparse.csr_matrix(A)
    # Ensure symmetric (take max of A and A.T)
    A = A.maximum(A.T)
    # Remove self-loops
    A.setdiag(0)
    A.eliminate_zeros()
    # Make binary (unweighted)
    A.data[:] = 1.0
    return A


def load_edgelist(path, delimiter=None, comments='#'):
    """Load graph from edge list file (SNAP format)."""
    import networkx as nx
    G = nx.read_edgelist(path, delimiter=delimiter, comments=comments,
                         nodetype=int, create_using=nx.Graph)
    # Relabel nodes to 0..n-1
    G = nx.convert_node_labels_to_integers(G)
    A = nx.adjacency_matrix(G).tocsr()
    A.data[:] = 1.0
    return A


def adjacency_to_laplacian(A):
    """Compute graph Laplacian L = D - A from adjacency matrix."""
    degrees = np.array(A.sum(axis=1)).flatten()
    D = sparse.diags(degrees)
    L = D - A
    return L.tocsr()


def compute_eigenvectors(L, rank=2, tol=1e-6, maxiter=2000):
    """Compute top-r eigenvectors of Laplacian."""
    eigvals, eigvecs = eigsh(L, k=rank, which='LM', maxiter=maxiter, tol=tol)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvals, eigvecs


def spectral_diagnostics(eigvals, rank=2):
    """Compute spectral diagnostics."""
    pos = eigvals[eigvals > 1e-10]
    total = float(np.sum(pos))
    top_r = float(np.sum(pos[:rank]))
    return {
        "energy_ratio": top_r / total if total > 0 else 0,
        "eigengap": float(pos[rank - 1] - pos[rank]) if len(pos) > rank else 0,
        "lambda_1": float(pos[0]),
        "lambda_r": float(pos[rank - 1]) if len(pos) >= rank else 0,
        "avg_degree": float(np.mean(np.array(eigvals))),
    }


def download_if_needed(url, dest_path):
    """Download file if it doesn't exist locally."""
    if os.path.exists(dest_path):
        return dest_path
    print(f"  Downloading {url}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    urllib.request.urlretrieve(url, dest_path)
    # Handle .tar.gz or .gz
    if dest_path.endswith('.tar.gz'):
        import tarfile
        with tarfile.open(dest_path) as tf:
            tf.extractall(os.path.dirname(dest_path))
    elif dest_path.endswith('.gz') and not dest_path.endswith('.tar.gz'):
        import gzip, shutil
        out_path = dest_path[:-3]
        with gzip.open(dest_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        return out_path
    return dest_path


# Registry of known datasets with download URLs
DATASETS = {
    # SuiteSparse DIMACS10 - Delaunay meshes
    "delaunay_n10": {
        "url": "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n10.tar.gz",
        "n": 1024, "type": "mesh", "structured": True,
    },
    "delaunay_n13": {
        "url": "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n13.tar.gz",
        "n": 8192, "type": "mesh", "structured": True,
    },
    "delaunay_n16": {
        "url": "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n16.tar.gz",
        "n": 65536, "type": "mesh", "structured": True,
    },
    "delaunay_n19": {
        "url": "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n19.tar.gz",
        "n": 524288, "type": "mesh", "structured": True,
    },
    # SNAP road networks
    "roadNet-PA": {
        "url": "https://snap.stanford.edu/data/roadNet-PA.txt.gz",
        "n": 1088092, "type": "road", "structured": True,
    },
    "roadNet-CA": {
        "url": "https://snap.stanford.edu/data/roadNet-CA.txt.gz",
        "n": 1965206, "type": "road", "structured": True,
    },
}


def process_graph(name, data_dir, rank=2, save=True):
    """Download (if needed), load, and process a graph."""
    print(f"\n=== {name} ===")

    if name in DATASETS:
        info = DATASETS[name]
        url = info["url"]
        ext = ".tar.gz" if url.endswith(".tar.gz") else ".txt.gz"
        archive_path = os.path.join(data_dir, f"{name}{ext}")
        download_if_needed(url, archive_path)

        # Find the .mtx or .txt file
        import glob
        mtx_files = glob.glob(os.path.join(data_dir, f"{name}*/*.mtx")) + \
                     glob.glob(os.path.join(data_dir, f"{name}*.mtx"))
        txt_files = glob.glob(os.path.join(data_dir, f"{name}*.txt"))

        if mtx_files:
            A = load_mtx(mtx_files[0])
        elif txt_files:
            A = load_edgelist(txt_files[0])
        else:
            raise FileNotFoundError(f"No .mtx or .txt file found for {name}")
    else:
        # Try loading directly as .mtx
        mtx_path = os.path.join(data_dir, f"{name}.mtx")
        if os.path.exists(mtx_path):
            A = load_mtx(mtx_path)
        else:
            raise FileNotFoundError(f"Unknown dataset: {name}")

    n = A.shape[0]
    nnz = A.nnz
    avg_deg = nnz / n
    print(f"  Loaded: {n:,} nodes, {nnz:,} edges, avg_degree={avg_deg:.1f}")

    # Compute Laplacian
    t0 = time.time()
    L = adjacency_to_laplacian(A)
    lap_time = time.time() - t0
    print(f"  Laplacian: {lap_time:.1f}s")

    # Eigensolve
    t0 = time.time()
    eigvals, eigvecs = compute_eigenvectors(L, rank=rank)
    eig_time = time.time() - t0
    print(f"  Eigensolve: {eig_time:.1f}s (lambda_1={eigvals[0]:.4f})")

    # Spectral diagnostics
    diag = spectral_diagnostics(eigvals, rank)
    print(f"  Energy ratio (rank-{rank}): {diag['energy_ratio']*100:.2f}%")
    print(f"  Eigengap: {diag['eigengap']:.4f}")

    if save:
        out_dir = os.path.join(data_dir, "processed")
        os.makedirs(out_dir, exist_ok=True)

        # Save sparse Laplacian
        sparse.save_npz(os.path.join(out_dir, f"L_{name}.npz"), L)

        # Save eigenvectors
        V = eigvecs[:, :rank] * np.sqrt(np.maximum(eigvals[:rank], 0))
        V_complex = V.astype(np.complex128)
        np.save(os.path.join(out_dir, f"V_{name}.npy"), V_complex)

        print(f"  Saved: L_{name}.npz, V_{name}.npy")

    return L, eigvecs, eigvals, diag


def main():
    parser = argparse.ArgumentParser(description="Load and process real-world graphs")
    parser.add_argument("--datasets", type=str, default="delaunay_n10,delaunay_n13",
                        help="Comma-separated dataset names")
    parser.add_argument("--data_dir", type=str, default="realworld_data")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--list", action="store_true", help="List available datasets")
    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for name, info in DATASETS.items():
            print(f"  {name}: n={info['n']:,}, type={info['type']}, structured={info['structured']}")
        return

    os.makedirs(args.data_dir, exist_ok=True)

    for name in args.datasets.split(","):
        name = name.strip()
        try:
            process_graph(name, args.data_dir, rank=args.rank)
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
