#!/usr/bin/env python3

import argparse
import numpy as np
from utils import generate_Q, low_rank_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="graphs")
    args = parser.parse_args()

    np.random.seed(args.seed)
    print("Generating Q for n={}".format(args.n))
    Q = generate_Q(0.5, args.n, 'erdos_renyi', seed=args.seed)
    q_path = f"{args.outdir}/Q_n{args.n}.npy"
    np.save(q_path, Q)
    print("Saved Q to {}".format(q_path))


if __name__ == "__main__":
    main()
