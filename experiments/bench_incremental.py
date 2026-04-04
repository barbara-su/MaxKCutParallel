"""
Benchmark: incremental vs naive rank-1 phase sweep.
Tests on sparse graphs at n=10K, 50K, 100K, 500K.
"""
import os, sys, time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

os.environ.setdefault("TMPDIR", os.environ.get("TMPDIR", "/tmp"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from hybrid import rank1_phase_sweep

def generate_sparse_regular(n, degree=5, seed=42):
    rng = np.random.RandomState(seed)
    if n * degree % 2 != 0:
        n += 1
    stubs = np.repeat(np.arange(n), degree)
    rng.shuffle(stubs)
    rows, cols = [], []
    for i in range(0, len(stubs) - 1, 2):
        u, v = stubs[i], stubs[i + 1]
        if u != v:
            rows.extend([u, v])
            cols.extend([v, u])
    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A.data[:] = 1.0
    D = sparse.diags(np.array(A.sum(1)).flatten())
    L = D - A
    return L

print("=" * 60)
print("  INCREMENTAL RANK-1 BENCHMARK")
print("=" * 60)

for n in [10000, 50000, 100000, 500000, 1000000]:
    print(f"\n--- n = {n:,} ---")

    t0 = time.time()
    L = generate_sparse_regular(n, degree=5, seed=42)
    gen_time = time.time() - t0
    print(f"  Graph generated in {gen_time:.2f}s ({L.nnz:,} nnz)")

    t0 = time.time()
    eigval, eigvec = eigsh(L, k=1, which='LM', maxiter=1000, tol=1e-6)
    eig_time = time.time() - t0
    print(f"  Eigensolve: {eig_time:.2f}s (lambda_1 = {eigval[0]:.4f})")

    V = eigvec.astype(np.complex128)

    t0 = time.time()
    score, best_k, best_z, sweep_time = rank1_phase_sweep(L, V, K=3)
    total = time.time() - t0
    print(f"  Incremental sweep: {sweep_time:.2f}s (score = {score:.0f})")
    print(f"  TOTAL: {gen_time + eig_time + sweep_time:.2f}s")

print(f"\n{'=' * 60}")
print("COMPLETE")
