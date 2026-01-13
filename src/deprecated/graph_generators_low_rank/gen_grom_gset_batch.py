import argparse
import os
import re
import numpy as np
import networkx as nx
from gen_v import *

def _parse_bool(x):
    # robust CLI bool parsing
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {x}")

def _safe_float_tag(v: float) -> str:
    # for filenames, e.g., 0.2 -> "0p2"
    return f"{v:.6g}".replace(".", "p").replace("-", "m")

def load_gset_graph(gset_id: int, in_dir: str, seed: int,
                    random_weights: bool, random_low: float, random_high: float) -> nx.Graph:
    path = os.path.join(in_dir, f"G{gset_id}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    rng = np.random.default_rng(seed)

    G = nx.Graph()
    with open(path, "r") as f:
        first = next(f).split()
        num_nodes = int(first[0])
        G.add_nodes_from(range(num_nodes))

        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            i, j, w = map(int, parts[:3])

            base_w = float(w)
            if not random_weights:
                w_final = base_w
            else:
                mult = (random_high - random_low) * rng.random() + random_low
                w_final = base_w * mult

            # GSet is 1-indexed
            G.add_edge(i - 1, j - 1, weight=w_final)

    return G

def main():
    parser = argparse.ArgumentParser(
        description="Batch-generate Q/V from GSet graphs G0..G80 (only those that exist)."
    )
    parser.add_argument(
        "--rank", type=int, default=1,
        help="Rank r for the low-rank eigenvector matrix V."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed."
    )
    parser.add_argument(
        "--random_weights", type=_parse_bool, default=False,
        help="Whether to randomize edge weights multiplicatively (0/1, true/false)."
    )
    parser.add_argument(
        "--random_low", type=float, default=0.0,
        help="Lower bound of random multiplier."
    )
    parser.add_argument(
        "--random_high", type=float, default=1.0,
        help="Upper bound of random multiplier."
    )
    parser.add_argument(
        "--in_dir", type=str, default="../graphs/gset",
        help="Directory containing GSet files like G1.txt, G2.txt, ..."
    )
    parser.add_argument(
        "--out_dir", type=str, default="graphs_gset",
        help="Directory to store generated Q/V .npy files."
    )
    parser.add_argument(
        "--max_id", type=int, default=81,
        help="Maximum GSet id to consider (inclusive). Default 80."
    )
    parser.add_argument(
        "--skip_existing", type=_parse_bool, default=True,
        help="Skip generation if output files already exist."
    )
    args = parser.parse_args()

    r = args.rank
    base_seed = args.seed
    random_weights = args.random_weights
    random_low = args.random_low
    random_high = args.random_high
    in_dir = args.in_dir
    out_dir = args.out_dir
    max_id = args.max_id
    skip_existing = args.skip_existing

    if random_weights and random_high < random_low:
        raise ValueError(f"--random_high must be >= --random_low, got {random_high} < {random_low}")

    os.makedirs(out_dir, exist_ok=True)

    # Optional filename tags (to avoid collisions across settings)
    rw_tag = "rand" if random_weights else "orig"
    low_tag = _safe_float_tag(random_low)
    high_tag = _safe_float_tag(random_high)

    generated = 0
    skipped_missing = 0
    skipped_existing = 0

    for gset_id in range(0, max_id + 1):
        in_path = os.path.join(in_dir, f"G{gset_id}.txt")
        if not os.path.exists(in_path):
            print(f"[skip missing] {in_path}")
            skipped_missing += 1
            continue

        # Make per-graph seed deterministic but distinct
        seed = base_seed + gset_id

        q_path = os.path.join(out_dir, f"Q_gset_{gset_id}.npy")
        v_path = os.path.join(out_dir, f"V_gset_{gset_id}.npy")

        if skip_existing and os.path.exists(q_path) and os.path.exists(v_path):
            print(f"[skip existing] gset={gset_id} -> {q_path}, {v_path}")
            skipped_existing += 1
            continue

        print(f"[gen] gset={gset_id} seed={seed} rank={r} random_weights={random_weights}")

        G = load_gset_graph(
            gset_id=gset_id,
            in_dir=in_dir,
            seed=seed,
            random_weights=random_weights,
            random_low=random_low,
            random_high=random_high,
        )

        Q = np.array(nx.laplacian_matrix(G).todense())
        V = gen_V_given_Q(Q, r)
        Q_hat = gen_Q_hat_given_V(V)

        np.save(q_path, Q_hat)
        np.save(v_path, V)

        print(f"  saved Q: {q_path} (shape {Q.shape})")
        print(f"  saved V: {v_path} (shape {V.shape})")
        generated += 1

    print("\nDone.")
    print(f"Generated: {generated}")
    print(f"Skipped missing: {skipped_missing}")
    print(f"Skipped existing: {skipped_existing}")

if __name__ == "__main__":
    main()
