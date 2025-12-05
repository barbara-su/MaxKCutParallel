import argparse
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic low-rank PSD matrix Q and its eigenvector matrix V."
    )
    parser.add_argument(
        "--n", type=int, default=10000,
        help="Dimension of the matrix."
    )
    parser.add_argument(
        "--rank", type=int, default=1,
        help="Target rank of Q and number of eigenvectors."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed."
    )
    parser.add_argument(
        "--out_dir", type=str, default="graphs_lowrank",
        help="Directory to store Q_n.npy and V_n.npy."
    )
    args = parser.parse_args()

    n = args.n
    r = args.rank
    seed = args.seed
    out_dir = args.out_dir

    print(f"Generating low-rank Q with n = {n}, rank = {r}, seed = {seed}")
    os.makedirs(out_dir, exist_ok=True)

    # reproducibility
    rng = np.random.default_rng(seed)

    # generate orthonormal eigenvectors V
    A = rng.normal(size=(n, r))
    Q, _ = np.linalg.qr(A)                # Q is n x r orthonormal
    V = Q

    # generate positive eigenvalues
    eigvals = rng.uniform(low=0.5, high=2.0, size=r)

    # construct PSD matrix of rank r
    Q_lowrank = V @ np.diag(eigvals) @ V.T

    print(f"Q_lowrank shape: {Q_lowrank.shape}")
    print(f"V shape: {V.shape}")

    # save outputs
    q_path = os.path.join(out_dir, f"Q_{n}.npy")
    v_path = os.path.join(out_dir, f"V_{n}.npy")

    np.save(q_path, Q_lowrank)
    np.save(v_path, V)

    print(f"Saved Q_lowrank to {q_path}")
    print(f"Saved V to {v_path}")


if __name__ == "__main__":
    main()
