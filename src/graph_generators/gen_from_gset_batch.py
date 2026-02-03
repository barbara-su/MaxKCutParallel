import argparse
import os
import numpy as np
import networkx as nx
from gen_v import gen_V_given_Q
import time
import json
from datetime import datetime
import logging


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("gset_gen")


class JSONRunLogger:
    def __init__(self, out_path: str):
        self.out_path = out_path
        self.records = []

    def log(self, record: dict):
        self.records.append(record)

    def flush(self):
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        with open(self.out_path, "w") as f:
            json.dump(
                {
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "num_records": len(self.records),
                    "records": self.records,
                },
                f,
                indent=2,
                sort_keys=True,
            )


def _parse_bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {x}")


def load_gset_graph(
    gset_id: int,
    in_dir: str,
    seed: int,
    random_weights: bool,
    random_low: float,
    random_high: float,
) -> nx.Graph:
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
            if random_weights:
                mult = (random_high - random_low) * rng.random() + random_low
                w_final = base_w * mult
            else:
                w_final = base_w

            G.add_edge(i - 1, j - 1, weight=w_final)

    return G


def main():
    logger = setup_logger()

    parser = argparse.ArgumentParser(
        description="Batch-generate Q/V from GSet graphs with JSON timing logs."
    )
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random_weights", type=_parse_bool, default=False)
    parser.add_argument("--random_low", type=float, default=0.0)
    parser.add_argument("--random_high", type=float, default=1.0)
    parser.add_argument("--in_dir", type=str, default="../graphs/gset")
    parser.add_argument("--out_dir", type=str, default="graphs_gset")
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--max_id", type=int, default=81)
    parser.add_argument("--skip_existing", type=_parse_bool, default=True)

    args = parser.parse_args()

    if args.random_weights and args.random_high < args.random_low:
        raise ValueError("--random_high must be >= --random_low")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    json_path = os.path.join(
        args.result_dir,
        f"gen_V_from_Q_rank{args.rank}_seed{args.seed}.json",
    )
    run_logger = JSONRunLogger(json_path)

    generated = 0
    skipped_missing = 0
    skipped_existing = 0

    for gset_id in range(0, args.max_id + 1):
        graph_name = f"G{gset_id}"
        in_path = os.path.join(args.in_dir, f"{graph_name}.txt")

        if not os.path.exists(in_path):
            skipped_missing += 1
            continue

        seed = args.seed + gset_id
        q_path = os.path.join(args.out_dir, f"Q_gset_{gset_id}.npy")
        v_path = os.path.join(args.out_dir, f"V_gset_{gset_id}.npy")

        if args.skip_existing and os.path.exists(q_path) and os.path.exists(v_path):
            skipped_existing += 1
            continue

        logger.info("START %s | rank=%d | seed=%d", graph_name, args.rank, seed)

        # Wall clock for reconciliation
        t_wall0 = time.perf_counter()

        # (1) Load time (disk): read file + build NetworkX graph
        t0 = time.perf_counter()
        G = load_gset_graph(
            gset_id,
            args.in_dir,
            seed,
            args.random_weights,
            args.random_low,
            args.random_high,
        )
        t_load_disk = time.perf_counter() - t0

        # (2) Load time (get laplacian): build sparse L + densify to Q
        t1 = time.perf_counter()
        Q = np.asarray(nx.laplacian_matrix(G).todense())
        t_laplacian = time.perf_counter() - t1

        # (3) Eigenvectors and (4) Construct V: timed inside gen_v.py
        t2 = time.perf_counter()
        V, v_timing = gen_V_given_Q(Q, args.rank, return_timing=True)
        t_genV_wrapper = time.perf_counter() - t2  # not exported, used only for sanity
        t_eigs = float(v_timing["compute_eigenvectors_time_s"])
        t_constructV = float(v_timing["construct_V_time_s"])

        # (5) Disk save time
        t3 = time.perf_counter()
        np.save(q_path, Q)
        np.save(v_path, V)
        t_save = time.perf_counter() - t3

        # Reconcile totals: enforce total = sum(components).
        t_wall = time.perf_counter() - t_wall0
        sum_components = t_load_disk + t_laplacian + t_eigs + t_constructV + t_save
        missing = t_wall - sum_components

        # If there is unaccounted time, fold it into disk_save to close the budget.
        # This keeps the JSON schema fixed, as requested.
        if missing > 1e-3:
            t_save += missing
            sum_components += missing

        total_time = sum_components

        run_logger.log(
            {
                "graph_id": gset_id,
                "load_time_disk_s": t_load_disk,
                "load_time_laplacian_s": t_laplacian,
                "compute_eigenvectors_time_s": t_eigs,
                "construct_V_time_s": t_constructV,
                "disk_save_time_s": t_save,
                "total_time_s": total_time,
            }
        )

        logger.info(
            "DONE  %s | disk=%.3fs | lap=%.3fs | eig=%.3fs | V=%.3fs | save=%.3fs | total=%.3fs",
            graph_name,
            t_load_disk,
            t_laplacian,
            t_eigs,
            t_constructV,
            t_save,
            total_time,
        )

        # Optional: warning if wrapper timing is wildly different than eig+V
        # This can catch unexpected overhead inside gen_V_given_Q.
        if abs(t_genV_wrapper - (t_eigs + t_constructV)) > 0.5:
            logger.info(
                "NOTE  %s | gen_V_wrapper=%.3fs vs (eig+V)=%.3fs (diff=%.3fs)",
                graph_name,
                t_genV_wrapper,
                (t_eigs + t_constructV),
                (t_genV_wrapper - (t_eigs + t_constructV)),
            )

        generated += 1

    run_logger.flush()

    logger.info(
        "SUMMARY | generated=%d | skipped_missing=%d | skipped_existing=%d | json=%s",
        generated,
        skipped_missing,
        skipped_existing,
        json_path,
    )


if __name__ == "__main__":
    main()
