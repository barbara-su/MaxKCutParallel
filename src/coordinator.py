"""
Cross-machine coordinator for rank-r Max-K-Cut on multiple GPU clusters.

Splits the combination space across machines proportional to GPU count,
launches worker.py on each via SSH, collects results, picks global best.

Usage:
    python coordinator.py --q_path Q.npy --v_path V.npy --rank 2 --K 3

Machines are configured in MACHINES list below.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np


# ── Machine configuration ──────────────────────────────────────────────────

MACHINES = [
    {
        "name": "anton.k0",
        "host": "tasos@anton.k0.rice.edu",
        "num_gpus": 4,
        "work_dir": "/data/tasos/max-k-cut",
        "python": "python3",
        "env_prefix": "export PATH=$HOME/.local/bin:$PATH; TMPDIR=/data/tasos/tmp",
    },
    {
        "name": "anton.k1",
        "host": "tasos@anton.k1.rice.edu",
        "num_gpus": 4,
        "work_dir": "/data/tasos/max-k-cut",
        "python": "python3",
        "env_prefix": "export PATH=$HOME/.local/bin:$PATH; TMPDIR=/data/tasos/tmp",
    },
    {
        "name": "kp001",
        "host": "ak85@kp001.rice.edu",
        "num_gpus": 4,
        "work_dir": "/data.local/home/ak85/max-k-cut",
        "python": "source ~/miniconda3/etc/profile.d/conda.sh && conda activate gpu && python",
        "env_prefix": "",
    },
    {
        "name": "kp002",
        "host": "ak85@kp002.rice.edu",
        "num_gpus": 3,
        "work_dir": "/data.local/home/ak85/max-k-cut",
        "python": "source ~/miniforge3/etc/profile.d/conda.sh && conda activate torch-new-clone1 && python",
        "env_prefix": "",
    },
]


def sync_code_to_machine(machine, local_src_dir):
    """Rsync source code and data to a remote machine."""
    remote = f"{machine['host']}:{machine['work_dir']}/"
    cmd = [
        "rsync", "-avz", "--exclude=.git", "--exclude=__pycache__",
        "--exclude=gset", "--exclude=graph_gen_logs",
        "--exclude=results", "--exclude=test_results",
        f"{local_src_dir}/", remote,
    ]
    print(f"  Syncing code to {machine['name']}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"    WARNING: rsync to {machine['name']} failed: {result.stderr[:200]}")
        return False
    return True


def sync_data_to_machine(machine, q_path, v_path, vtilde_path):
    """Sync Q, V, V_tilde files to a remote machine."""
    remote_dir = f"{machine['host']}:{machine['work_dir']}/run_data/"

    # Ensure remote dir exists
    subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", machine["host"],
         f"mkdir -p {machine['work_dir']}/run_data"],
        capture_output=True, timeout=30,
    )

    for local_path in [q_path, v_path, vtilde_path]:
        if local_path and os.path.exists(local_path):
            cmd = ["scp", "-o", "ConnectTimeout=10", local_path, remote_dir]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                print(f"    WARNING: scp {os.path.basename(local_path)} to {machine['name']} failed")
                return False
    return True


def launch_worker(machine, q_name, v_name, vtilde_name, start_rank, end_rank, r, K, chunk_size, precision):
    """Launch worker.py on a remote machine via SSH. Returns Popen object."""
    work_dir = machine["work_dir"]
    result_path = f"{work_dir}/run_data/result_{machine['name']}.json"

    parts = []
    if machine["env_prefix"]:
        parts.append(machine["env_prefix"])

    parts.append(f"cd {work_dir}/src")

    worker_cmd = (
        f"{machine['python']} worker.py"
        f" --q_path {work_dir}/run_data/{q_name}"
        f" --v_path {work_dir}/run_data/{v_name}"
        f" --vtilde_path {work_dir}/run_data/{vtilde_name}"
        f" --start_rank {start_rank} --end_rank {end_rank}"
        f" --rank {r} --K {K}"
        f" --num_gpus {machine['num_gpus']}"
        f" --chunk_size {chunk_size}"
        f" --precision {precision}"
        f" --out {result_path}"
    )
    parts.append(worker_cmd)

    full_cmd = " && ".join(parts)
    ssh_cmd = ["ssh", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30", machine["host"], full_cmd]

    print(f"  Launching on {machine['name']} ({machine['num_gpus']} GPUs): [{start_rank:,}, {end_rank:,})")
    proc = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc, result_path


def collect_result(machine, result_path):
    """Download result JSON from remote machine."""
    local_path = f"/tmp/result_{machine['name']}.json"
    cmd = ["scp", "-o", "ConnectTimeout=10", f"{machine['host']}:{result_path}", local_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print(f"    WARNING: Could not fetch result from {machine['name']}")
        return None
    with open(local_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Coordinate rank-r Max-K-Cut across multiple GPU machines")
    parser.add_argument("--q_path", type=str, required=True, help="Local path to Q.npy")
    parser.add_argument("--v_path", type=str, required=True, help="Local path to V.npy")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--chunk_size", type=int, default=50000)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--machines", type=str, default="all",
                        help="Comma-separated machine names or 'all'")
    parser.add_argument("--sync_code", action="store_true", help="Rsync code to all machines first")
    parser.add_argument("--out", type=str, default="final_result.json")
    args = parser.parse_args()

    # Select machines
    if args.machines == "all":
        machines = MACHINES
    else:
        names = set(args.machines.split(","))
        machines = [m for m in MACHINES if m["name"] in names]

    if not machines:
        print("ERROR: No machines selected")
        sys.exit(1)

    total_gpus = sum(m["num_gpus"] for m in machines)
    print(f"Machines: {[m['name'] for m in machines]} ({total_gpus} GPUs total)")

    # Load and validate
    Q = np.load(args.q_path)
    V = np.load(args.v_path)
    if V.ndim == 1:
        V = V.reshape(-1, 1)
    V = V[:, :args.rank]
    n = Q.shape[0]

    # Precompute V_tilde locally
    print("Computing V_tilde...")
    from utils import compute_vtilde
    V_tilde = compute_vtilde(V)
    vtilde_path = args.q_path.replace("Q_", "Vtilde_")
    if vtilde_path == args.q_path:
        vtilde_path = "/tmp/Vtilde_temp.npy"
    np.save(vtilde_path, V_tilde)

    N = args.K * n
    comb_size = 2 * args.rank - 1
    total_comb = math.comb(N, comb_size)
    print(f"Instance: n={n}, r={args.rank}, K={args.K}")
    print(f"Total combinations: C({N},{comb_size}) = {total_comb:,}")

    # Sync code if requested
    local_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.sync_code:
        for m in machines:
            sync_code_to_machine(m, local_src)

    # Sync data
    q_name = os.path.basename(args.q_path)
    v_name = os.path.basename(args.v_path)
    vtilde_name = os.path.basename(vtilde_path)
    print("\nSyncing data to machines...")
    for m in machines:
        sync_data_to_machine(m, args.q_path, args.v_path, vtilde_path)

    # Split work proportionally to GPU count
    assignments = []
    cursor = 0
    for m in machines:
        share = int(total_comb * m["num_gpus"] / total_gpus)
        end = min(cursor + share, total_comb)
        if m == machines[-1]:
            end = total_comb  # last machine gets remainder
        assignments.append((m, cursor, end))
        cursor = end

    # Launch all workers
    print(f"\nLaunching workers...")
    t_start = time.time()
    procs = []
    for m, start, end in assignments:
        proc, result_path = launch_worker(
            m, q_name, v_name, vtilde_name,
            start, end, args.rank, args.K, args.chunk_size, args.precision,
        )
        procs.append((m, proc, result_path))

    # Wait for all to complete, streaming output
    for m, proc, _ in procs:
        for line in proc.stdout:
            print(f"[{m['name']}] {line}", end="")
        proc.wait()

    total_elapsed = time.time() - t_start
    print(f"\nAll workers finished in {total_elapsed:.1f}s")

    # Collect and merge results
    print("\nCollecting results...")
    best_score = float("-inf")
    best_k = None
    best_z = None
    total_feasible = 0
    total_processed = 0
    per_machine = {}

    for m, _, result_path in procs:
        res = collect_result(m, result_path)
        if res is None:
            print(f"  {m['name']}: FAILED (no result)")
            continue

        per_machine[m["name"]] = {
            "score": res["best_score"],
            "time": res["time_seconds"],
            "processed": res["combinations_processed"],
            "feasible": res["feasible_count"],
            "gpus": res["num_gpus"],
        }
        total_feasible += res["feasible_count"]
        total_processed += res["combinations_processed"]

        if res["best_k"] is not None and res["best_score"] > best_score:
            best_score = res["best_score"]
            best_k = res["best_k"]
            best_z_real = res["best_z_real"]
            best_z_imag = res["best_z_imag"]

        print(f"  {m['name']}: score={res['best_score']:.0f}, time={res['time_seconds']:.1f}s, "
              f"processed={res['combinations_processed']:,}")

    # Final output
    print(f"\n{'='*60}")
    print(f"FINAL RESULT")
    print(f"{'='*60}")
    print(f"  Score: {best_score:.0f}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Total GPUs: {total_gpus}")
    print(f"  Processed: {total_processed:,}")
    print(f"  Feasible: {total_feasible:,} ({total_feasible/max(1,total_processed)*100:.1f}%)")
    print(f"  Throughput: {total_processed/total_elapsed:,.0f} candidates/sec")

    output = {
        "best_score": best_score,
        "total_time_seconds": total_elapsed,
        "total_gpus": total_gpus,
        "total_processed": total_processed,
        "total_feasible": total_feasible,
        "n": n,
        "rank": args.rank,
        "K": args.K,
        "per_machine": per_machine,
        "best_k": best_k,
        "best_z_real": best_z_real if best_k else None,
        "best_z_imag": best_z_imag if best_k else None,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
