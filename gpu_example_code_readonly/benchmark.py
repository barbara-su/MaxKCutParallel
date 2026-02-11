import time
from math import comb
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gpu_solver import gpu_rank_r_solver
from ray_solver_wrapper import ray_rank_r_solver


def generate_test_problem(n: int, r: int, seed: int = 42) -> Dict:
    rng = np.random.default_rng(seed)
    V = (rng.standard_normal((n, r)) + 1j * rng.standard_normal((n, r))) / np.sqrt(2)
    row_norms = np.linalg.norm(V, axis=1, keepdims=True)
    V_tilde = V / row_norms  # simple normalization
    M = V @ V.conj().T
    return {"M": M, "V": V, "V_tilde": V_tilde, "n": n, "r": r}


def benchmark_solvers(
    n_values: List[int],
    r_values: List[int],
    K: int = 3,
    num_trials: int = 3,
    batch_size: int = 4096,
    candidates_per_task: int = 1000,
) -> pd.DataFrame:
    results = []

    for n in n_values:
        for r in r_values:
            num_combinations = comb(n, 2 * r - 1)
            if num_combinations > 1e7:  # cap for practicality
                print(f"Skipping n={n}, r={r}: {num_combinations:.2e} combinations")
                continue

            print(f"\nBenchmarking n={n}, r={r} ({num_combinations:.2e} combos)")

            gpu_times = []
            ray_times = []
            solutions_match = True

            for trial in range(num_trials):
                problem = generate_test_problem(n, r, seed=42 + trial)

                # GPU solver
                start = time.perf_counter()
                x_gpu, obj_gpu = gpu_rank_r_solver(
                    problem["V_tilde"], problem["V"], K, batch_size=batch_size
                )
                gpu_times.append(time.perf_counter() - start)

                # Ray solver
                start = time.perf_counter()
                x_ray, obj_ray = ray_rank_r_solver(
                    problem["V_tilde"], problem["V"], K, candidates_per_task=candidates_per_task
                )
                ray_times.append(time.perf_counter() - start)

                if not np.isclose(obj_gpu, obj_ray, rtol=1e-6, atol=1e-8):
                    solutions_match = False
                    print(f"  WARNING: Objectives differ! GPU={obj_gpu}, Ray={obj_ray}")

            results.append(
                {
                    "n": n,
                    "r": r,
                    "K": K,
                    "num_combinations": num_combinations,
                    "gpu_time_mean": np.mean(gpu_times),
                    "gpu_time_std": np.std(gpu_times),
                    "ray_time_mean": np.mean(ray_times),
                    "ray_time_std": np.std(ray_times),
                    "speedup": np.mean(ray_times) / np.mean(gpu_times),
                    "solutions_match": solutions_match,
                }
            )

            print(
                f"  GPU: {np.mean(gpu_times):.3f}s, Ray: {np.mean(ray_times):.3f}s, "
                f"Speedup: {results[-1]['speedup']:.2f}x"
            )

    return pd.DataFrame(results)


def plot_results(df: pd.DataFrame, save_path: str = "benchmark_results.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax1 = axes[0]
    for r in df["r"].unique():
        subset = df[df["r"] == r]
        ax1.semilogy(subset["n"], subset["gpu_time_mean"], "o-", label=f"GPU (r={r})")
        ax1.semilogy(subset["n"], subset["ray_time_mean"], "s--", label=f"Ray (r={r})")
    ax1.set_xlabel("Problem size n")
    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("Execution Time vs Problem Size")
    ax1.legend()
    ax1.grid(True)

    ax2 = axes[1]
    for r in df["r"].unique():
        subset = df[df["r"] == r]
        ax2.plot(subset["n"], subset["speedup"], "o-", label=f"r={r}")
    ax2.axhline(y=1, color="k", linestyle="--", label="Break-even")
    ax2.set_xlabel("Problem size n")
    ax2.set_ylabel("Speedup (Ray / GPU)")
    ax2.set_title("GPU Speedup over Ray")
    ax2.legend()
    ax2.grid(True)

    ax3 = axes[2]
    ax3.semilogx(df["num_combinations"], df["speedup"], "o")
    ax3.axhline(y=1, color="k", linestyle="--", label="Break-even")
    ax3.set_xlabel("Number of combinations")
    ax3.set_ylabel("Speedup")
    ax3.set_title("GPU Speedup vs Combinatorial Complexity")
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


if __name__ == "__main__":
    n_values = [20, 30, 50, 75]  # keep moderate for demo
    r_values = [2, 3]
    K = 3

    print("=" * 60)
    print("RANK-r ALGORITHM: GPU vs RAY BENCHMARK")
    print("=" * 60)

    results_df = benchmark_solvers(
        n_values, r_values, K, num_trials=2, batch_size=2048, candidates_per_task=500
    )
    results_df.to_csv("benchmark_results.csv", index=False)
    print("\nResults saved to benchmark_results.csv")
    print("\nSummary:\n", results_df.to_string())

    plot_results(results_df)

    avg_speedup = results_df["speedup"].mean()
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    if avg_speedup > 2:
        print(f"GPU implementation is {avg_speedup:.1f}x faster on average. RECOMMENDED.")
    elif avg_speedup > 1:
        print(f"GPU implementation is {avg_speedup:.1f}x faster on average. Marginal benefit.")
    else:
        print(f"Ray implementation is faster. GPU speedup = {avg_speedup:.2f}x. NOT RECOMMENDED.")