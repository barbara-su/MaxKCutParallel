#!/usr/bin/env python3
import argparse
import os
import numpy as np
import networkx as nx

from gen_v import gen_V_given_Q


def load_gset_edges(path: str):
    """
    Returns:
      n: int
      edges: list of (i, j, w_int) with 0-indexed endpoints, original integer weight
    """
    edges = []
    with open(path, "r") as f:
        first = next(f).split()
        n = int(first[0])
        for line in f:
            if not line.strip():
                continue
            i, j, w = map(int, line.split()[:3])
            edges.append((i - 1, j - 1, w))
    return n, edges


def build_graph(n: int, edges, rng: np.random.RandomState, random_weights: bool, low: float, high: float):
    """
    Build a NetworkX graph from base edges, optionally rescaling each edge weight by a random factor in [low, high).
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))

    if random_weights:
        span = float(high - low)
        for i, j, w in edges:
            factor = span * rng.rand() + float(low)
            G.add_edge(i, j, weight=float(w) * factor)
    else:
        for i, j, w in edges:
            G.add_edge(i, j, weight=float(w))

    return G


def main():
    parser = argparse.ArgumentParser(
        description="Generate Laplacian Q (and optional low-rank V) for a GSet graph under many seeds."
    )
    parser.add_argument("--gset", type=int, default=10000, help="GSet instance id, loads G{gset}.txt")
    parser.add_argument("--in_dir", type=str, default="../graphs/gset", help="Directory containing GSet txt files")
    parser.add_argument("--out_dir", type=str, default="graphs", help="Directory to store outputs")

    # Seeds: either provide explicit list or (start, count)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Explicit seed list, e.g. --seeds 0 1 2 ... If omitted, uses seed_start..seed_start+count-1.",
    )
    parser.add_argument("--seed_start", type=int, default=0, help="Start seed if --seeds not provided")
    parser.add_argument("--count", type=int, default=20, help="How many seeds if --seeds not provided")

    # Random edge weights
    parser.add_argument(
        "--random_weights",
        action="store_true",
        help="If set, rescale each edge weight by U([random_low, random_high)) per seed.",
    )
    parser.add_argument("--random_low", type=float, default=0.0, help="Lower bound for random rescale factor")
    parser.add_argument("--random_high", type=float, default=1.0, help="Upper bound for random rescale factor")

    # Optional V generation
    parser.add_argument("--rank", type=int, default=1, help="Rank r for V if saving V")
    parser.add_argument(
        "--save_v",
        action="store_true",
        help="If set, also save V_gset_{gset}_seed_{seed}.npy alongside Q.",
    )

    args = parser.parse_args()

    gset = args.gset
    in_dir = args.in_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    gset_path = os.path.join(in_dir, f"G{gset}.txt")
    if not os.path.exists(gset_path):
        raise FileNotFoundError(f"Could not find {gset_path}")

    n, base_edges = load_gset_edges(gset_path)

    # Choose seeds
    if args.seeds is None or len(args.seeds) == 0:
        seeds = list(range(args.seed_start, args.seed_start + args.count))
    else:
        seeds = list(args.seeds)

    print(f"Using G{gset} from GSet: n={n}, edges={len(base_edges)}")
    print(f"Generating {len(seeds)} instance(s): seeds={seeds[:5]}{'...' if len(seeds) > 5 else ''}")
    print(f"random_weights={args.random_weights}, low={args.random_low}, high={args.random_high}")
    print(f"save_v={args.save_v}, rank={args.rank}")

    for seed in seeds:
        rng = np.random.RandomState(seed)

        G = build_graph(
            n=n,
            edges=base_edges,
            rng=rng,
            random_weights=args.random_weights,
            low=args.random_low,
            high=args.random_high,
        )

        Q = np.array(nx.laplacian_matrix(G).todense())

        q_path = os.path.join(out_dir, f"Q_gset_{gset}_seed_{seed}.npy")
        np.save(q_path, Q)

        if args.save_v:
            V = gen_V_given_Q(Q, args.rank)
            v_path = os.path.join(out_dir, f"V_gset_{gset}_seed_{seed}.npy")
            np.save(v_path, V)

        print(f"[seed={seed}] saved {q_path}" + (f" and V(rank={args.rank})" if args.save_v else ""))


if __name__ == "__main__":
    main()
