"""
Generate all instances for the blog post experiments.
3 families × 6 sizes × 3 seeds = 54 instances.
Also precomputes V_tilde for each and runs spectral diagnostics.
"""
import argparse
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from utils import compute_vtilde


def generate_regular(n, degree, rank, seed, outdir):
    """Generate random d-regular graph."""
    import networkx as nx
    G = nx.random_regular_graph(degree, n, seed=seed)
    L = nx.laplacian_matrix(G).toarray().astype(np.float64)
    eigvals, eigvecs = np.linalg.eigh(L)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    V = eigvecs[:, :rank] * np.sqrt(np.maximum(eigvals[:rank], 0))
    return L, V.astype(np.complex128), eigvals


def generate_sbm(n, blocks, p_in, p_out, rank, seed):
    """Generate stochastic block model graph."""
    import networkx as nx
    block_size = n // blocks
    sizes = [block_size] * blocks
    # Adjust last block to account for rounding
    sizes[-1] = n - block_size * (blocks - 1)
    probs = [[p_in if i == j else p_out for j in range(blocks)] for i in range(blocks)]
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    L = nx.laplacian_matrix(G).toarray().astype(np.float64)
    eigvals, eigvecs = np.linalg.eigh(L)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    V = eigvecs[:, :rank] * np.sqrt(np.maximum(eigvals[:rank], 0))
    return L, V.astype(np.complex128), eigvals


def generate_torus(target_n, rank, epsilon, seed):
    """Generate toroidal grid graph."""
    from gen_torus import TORUS_DIMS, generate_torus_instance
    if target_n not in TORUS_DIMS:
        closest = min(TORUS_DIMS.keys(), key=lambda x: abs(x - target_n))
        target_n = closest
    p, q = TORUS_DIMS[target_n]
    Q, V = generate_torus_instance(p, q, rank, epsilon, seed)
    eigvals = np.linalg.eigvalsh(Q)[::-1]
    return Q, V, eigvals


def spectral_diagnostics(eigvals, rank):
    """Compute spectral diagnostics."""
    pos = eigvals[eigvals > 1e-10]
    total_energy = float(np.sum(pos))
    top_r_energy = float(np.sum(pos[:rank]))
    energy_ratio = top_r_energy / total_energy if total_energy > 0 else 0
    eigengap = float(pos[rank - 1] - pos[rank]) if len(pos) > rank else 0
    condition = float(pos[0] / pos[rank - 1]) if pos[rank - 1] > 1e-10 else float('inf')
    return {
        "top_r_energy_ratio": energy_ratio,
        "eigengap": eigengap,
        "lambda_1": float(pos[0]),
        "lambda_r": float(pos[rank - 1]),
        "lambda_r_plus_1": float(pos[rank]) if len(pos) > rank else 0,
        "condition_number": condition,
        "total_energy": total_energy,
    }


def save_instance(Q, V, eigvals, family, n, seed, rank, base_dir):
    """Save Q, V, V_tilde, and spectral diagnostics."""
    outdir = os.path.join(base_dir, family)
    os.makedirs(outdir, exist_ok=True)

    actual_n = Q.shape[0]
    tag = f"{family}_{actual_n}_seed_{seed}"

    q_path = os.path.join(outdir, f"Q_{tag}.npy")
    v_path = os.path.join(outdir, f"V_{tag}.npy")
    vt_path = os.path.join(outdir, f"Vtilde_{tag}.npy")
    diag_path = os.path.join(outdir, f"diag_{tag}.json")

    np.save(q_path, Q)
    np.save(v_path, V[:, :rank])

    Vt = compute_vtilde(V[:, :rank]).astype(np.float32)
    np.save(vt_path, Vt)

    diag = spectral_diagnostics(eigvals, rank)
    diag["family"] = family
    diag["n"] = actual_n
    diag["seed"] = seed
    diag["rank"] = rank
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2)

    print(f"  {tag}: n={actual_n}, energy_ratio={diag['top_r_energy_ratio']:.3f}, "
          f"eigengap={diag['eigengap']:.4f}, lambda1={diag['lambda_1']:.3f}")
    return diag


def main():
    parser = argparse.ArgumentParser(description="Generate all blog experiment instances")
    parser.add_argument("--base_dir", type=str, default="blog_instances")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--families", type=str, default="all",
                        help="Comma-separated: regular,sbm,torus or 'all'")
    parser.add_argument("--sizes", type=str, default="250,500,750,1000,1250,1500")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    args = parser.parse_args()

    families = args.families.split(",") if args.families != "all" else ["regular", "sbm", "torus"]
    sizes = [int(x) for x in args.sizes.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    all_diags = []
    os.makedirs(args.base_dir, exist_ok=True)

    for family in families:
        print(f"\n{'='*60}")
        print(f"  {family.upper()}")
        print(f"{'='*60}")

        for n in sizes:
            for seed in seeds:
                try:
                    if family == "regular":
                        Q, V, eigvals = generate_regular(n, 5, args.rank, seed, args.base_dir)
                    elif family == "sbm":
                        Q, V, eigvals = generate_sbm(n, 3, 0.3, 0.01, args.rank, seed)
                    elif family == "torus":
                        Q, V, eigvals = generate_torus(n, args.rank, 0.01, seed)
                    else:
                        print(f"  Unknown family: {family}, skipping")
                        continue

                    diag = save_instance(Q, V, eigvals, family, n, seed, args.rank, args.base_dir)
                    all_diags.append(diag)
                except Exception as e:
                    print(f"  ERROR generating {family} n={n} seed={seed}: {e}")

    # Save summary
    summary_path = os.path.join(args.base_dir, "spectral_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_diags, f, indent=2)
    print(f"\nSaved spectral summary: {summary_path}")
    print(f"Total instances: {len(all_diags)}")

    # Print summary table
    print(f"\n{'Family':<10} {'n':>6} {'seed':>5} {'energy%':>8} {'eigengap':>10} {'lambda1':>9}")
    print("-" * 55)
    for d in all_diags:
        print(f"{d['family']:<10} {d['n']:>6} {d['seed']:>5} {d['top_r_energy_ratio']*100:>7.1f}% "
              f"{d['eigengap']:>10.4f} {d['lambda_1']:>9.3f}")


if __name__ == "__main__":
    main()
