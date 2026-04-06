"""
Run rank-1, hybrid, greedy, SA on real-world graphs.
Phase 1: Small validation graphs (Delaunay n10, n13).
"""
import json
import os
import sys
import time

os.environ.setdefault("TMPDIR", os.environ.get("TMPDIR", "/tmp"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from baselines import random_cut, greedy_cut_incremental, sa_cut, score_cut
from hybrid import rank1_phase_sweep


def run_on_graph(name, L, V, K=3, time_budget=600):
    """Run all methods on one graph."""
    n = L.shape[0]
    roots = np.exp(2j * np.pi * np.arange(K) / K)

    results = {"name": name, "n": n, "K": K}

    # Random
    print(f"  Random...", flush=True)
    rng = np.random.RandomState(42)
    best_rs = -np.inf
    t0 = time.time()
    for _ in range(min(n + 1, 1000)):
        k_rand = rng.randint(0, K, size=n)
        z_rand = roots[k_rand]
        s = np.real(z_rand.conj() @ L.dot(z_rand))
        if s > best_rs:
            best_rs = s
    r_time = time.time() - t0
    results["random"] = {"score": float(best_rs), "time": round(r_time, 2)}
    print(f"    Score={best_rs:.0f}, Time={r_time:.1f}s")

    # Rank-1
    print(f"  Rank-1...", flush=True)
    r1_score, r1_k, r1_z, r1_time = rank1_phase_sweep(L, V, K=K)
    results["rank1"] = {"score": r1_score, "time": round(r1_time, 2)}
    print(f"    Score={r1_score:.0f}, Time={r1_time:.1f}s")

    # Greedy
    print(f"  Greedy...", flush=True)
    g_score, g_z, g_time, g_iters = greedy_cut_incremental(
        L, K=K, seed=42, max_time=time_budget, max_iters=20
    )
    results["greedy"] = {"score": g_score, "time": round(g_time, 2), "iterations": g_iters}
    print(f"    Score={g_score:.0f}, Time={g_time:.1f}s, Iters={g_iters}")

    # Hybrid
    print(f"  Hybrid...", flush=True)
    t0 = time.time()
    hw_score, _, hw_time, hw_iters = greedy_cut_incremental(
        L, K=K, seed=42, init_k=r1_k, max_time=time_budget, max_iters=20
    )
    hybrid_total = r1_time + hw_time
    results["hybrid"] = {
        "score": hw_score, "total_time": round(hybrid_total, 2),
        "rank1_time": round(r1_time, 2), "greedy_time": round(hw_time, 2),
        "iterations": hw_iters,
    }
    print(f"    Score={hw_score:.0f}, Time={hybrid_total:.1f}s (R1={r1_time:.1f}s + G={hw_time:.1f}s), Iters={hw_iters}")

    # SA
    print(f"  SA...", flush=True)
    sa_iters = max(20 * n, 100000)
    sa_score, _, sa_time, sa_acc = sa_cut(
        L, K=K, seed=42, max_iters=sa_iters, max_time=time_budget, cooling=0.99999
    )
    results["sa"] = {"score": sa_score, "time": round(sa_time, 2), "accepted": sa_acc}
    print(f"    Score={sa_score:.0f}, Time={sa_time:.1f}s")

    # Summary
    scores = {m: results[m]["score"] for m in ["random", "rank1", "greedy", "hybrid", "sa"]}
    winner = max(scores, key=scores.get)
    print(f"  Winner: {winner} ({scores[winner]:.0f})")
    results["winner"] = winner

    return results


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "realworld_data", "processed")
    out_dir = os.path.join(os.path.dirname(__file__), "realworld_results")
    os.makedirs(out_dir, exist_ok=True)

    # Find all processed graphs
    import glob
    v_files = sorted(glob.glob(os.path.join(data_dir, "V_*.npy")))

    if not v_files:
        print("No processed graphs found. Run gen_from_mtx.py first.")
        return

    for v_path in v_files:
        name = os.path.basename(v_path).replace("V_", "").replace(".npy", "")
        l_path = os.path.join(data_dir, f"L_{name}.npz")

        out_path = os.path.join(out_dir, f"{name}.json")
        if os.path.exists(out_path):
            print(f"SKIP: {name}")
            continue

        if not os.path.exists(l_path):
            print(f"SKIP: {name} (no Laplacian)")
            continue

        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        L = sparse.load_npz(l_path)
        V = np.load(v_path)
        if not np.iscomplexobj(V):
            V = V.astype(np.complex128)

        result = run_on_graph(name, L, V, K=3, time_budget=600)

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
