"""Benchmark full-GPU rank-2 tuning knobs on a single multi-GPU node.

This script measures:
- `gpu_inner_batch_size` over a fixed number of logical rank-2 tasks
- `max_in_flight_gpu_requests` for the best-performing inner batch

It is intended for empirical tuning on one node with local Ray resources.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import ray

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from parallel_rank_r_dir_gpu_fullgpu import RankRGPUActor, _resolve_max_in_flight_gpu_requests
from utils import compute_vtilde


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Benchmark rank-2 full-GPU tuning settings on one node.")
    ap.add_argument("--q_path", type=str, required=True)
    ap.add_argument("--v_path", type=str, required=True)
    ap.add_argument("--output_path", type=str, required=True)
    ap.add_argument("--num_gpus", type=int, default=4)
    ap.add_argument("--num_cpus", type=int, default=16)
    ap.add_argument("--candidates_per_task", type=int, default=100_000_000)
    ap.add_argument("--num_tasks", type=int, default=8)
    ap.add_argument(
        "--inner_batch_values",
        type=int,
        nargs="+",
        default=[262144, 335544, 524288, 1048576, 0],
        help="Explicit inner-batch values to test. Use 0 to test actor-managed auto mode.",
    )
    ap.add_argument(
        "--queue_depth_values",
        type=int,
        nargs="+",
        default=[4, 8, 12],
        help="Queue-depth values to test after choosing the best inner-batch result.",
    )
    return ap.parse_args()


def run_case(
    actors: List["ray.actor.ActorHandle"],
    candidates_per_task: int,
    num_tasks: int,
    inner_batch: int,
    max_in_flight: int,
) -> Dict[str, object]:
    start_ranks = [i * candidates_per_task for i in range(num_tasks)]
    in_flight = []
    submit_meta = {}
    compute_secs = []
    queue_wall_secs = []
    feasible_counts = []
    best_score = float("-inf")
    best_batch = None
    t0 = time.perf_counter()

    def submit(batch_id: int, start_rank: int):
        actor = actors[batch_id % len(actors)]
        submit_t = time.perf_counter()
        fut = actor.score_rank_batch.remote(
            int(start_rank),
            int(candidates_per_task),
            2,
            int(inner_batch),
        )
        submit_meta[fut] = (batch_id, submit_t)
        return fut

    def handle_done(fut):
        nonlocal best_score, best_batch
        batch_id, submit_t = submit_meta.pop(fut)
        score, _best_k, _best_z, feasible_count, compute_sec = ray.get(fut)
        compute_secs.append(float(compute_sec))
        queue_wall_secs.append(float(time.perf_counter() - submit_t))
        feasible_counts.append(int(feasible_count))
        if score > best_score:
            best_score = float(score)
            best_batch = int(batch_id)

    for batch_id, start_rank in enumerate(start_ranks):
        in_flight.append(submit(batch_id, start_rank))
        if len(in_flight) >= max_in_flight:
            done, in_flight = ray.wait(in_flight, num_returns=1)
            handle_done(done[0])

    while in_flight:
        done, in_flight = ray.wait(in_flight, num_returns=1)
        handle_done(done[0])

    total_wall = float(time.perf_counter() - t0)
    return {
        "inner_batch": int(inner_batch),
        "max_in_flight_gpu_requests": int(max_in_flight),
        "num_tasks": int(num_tasks),
        "total_wall_seconds": total_wall,
        "avg_compute_seconds": float(sum(compute_secs) / len(compute_secs)),
        "avg_queue_wall_seconds": float(sum(queue_wall_secs) / len(queue_wall_secs)),
        "best_score_seen": float(best_score),
        "best_batch_seen": int(best_batch) if best_batch is not None else None,
        "total_feasible": int(sum(feasible_counts)),
        "per_task_compute_seconds": compute_secs,
    }


def write_payload(
    output_path: Path,
    q_path: Path,
    v_path: Path,
    args: argparse.Namespace,
    auto_inner_batch_per_actor: List[int],
    default_queue_depth: int,
    inner_batch_results: List[Dict[str, object]],
    queue_depth_results: List[Dict[str, object]],
) -> None:
    payload = {
        "graph_q": str(q_path),
        "graph_v": str(v_path),
        "num_gpus": int(args.num_gpus),
        "num_cpus": int(args.num_cpus),
        "candidates_per_task": int(args.candidates_per_task),
        "num_tasks": int(args.num_tasks),
        "auto_gpu_inner_batch_size_per_actor": [int(x) for x in auto_inner_batch_per_actor],
        "default_auto_queue_depth": int(default_queue_depth),
        "inner_batch_results": inner_batch_results,
        "queue_depth_results": queue_depth_results,
    }
    output_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    q_path = Path(args.q_path).expanduser().resolve()
    v_path = Path(args.v_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    Q = np.asarray(np.load(q_path), dtype=np.float32)
    V_full = np.asarray(np.load(v_path), dtype=np.complex64)
    V = V_full[:, :2]
    V_tilde = compute_vtilde(V)

    ray.init(num_cpus=int(args.num_cpus), num_gpus=int(args.num_gpus), include_dashboard=False, ignore_reinit_error=True)
    actors = [RankRGPUActor.remote(K=3, precision=32) for _ in range(int(args.num_gpus))]
    ray.get([a.set_instance.remote(V, Q, V_tilde) for a in actors])
    auto_inner_batch_per_actor = ray.get([a.get_effective_gpu_inner_batch_size.remote() for a in actors])
    print(
        "AUTO",
        json.dumps({"auto_gpu_inner_batch_size_per_actor": [int(x) for x in auto_inner_batch_per_actor]}, sort_keys=True),
        flush=True,
    )

    # Warm one actor once so kernel startup does not dominate the first measured case.
    ray.get(actors[0].score_rank_batch.remote(0, min(1_000_000, int(args.candidates_per_task)), 2, 0))

    default_queue_depth = _resolve_max_in_flight_gpu_requests(0, len(actors))
    inner_batch_results = []
    queue_depth_results: List[Dict[str, object]] = []
    for inner_batch in args.inner_batch_values:
        result = run_case(
            actors=actors,
            candidates_per_task=int(args.candidates_per_task),
            num_tasks=int(args.num_tasks),
            inner_batch=int(inner_batch),
            max_in_flight=int(default_queue_depth),
        )
        inner_batch_results.append(result)
        write_payload(
            output_path=output_path,
            q_path=q_path,
            v_path=v_path,
            args=args,
            auto_inner_batch_per_actor=auto_inner_batch_per_actor,
            default_queue_depth=default_queue_depth,
            inner_batch_results=inner_batch_results,
            queue_depth_results=queue_depth_results,
        )
        print("INNER", json.dumps(result, sort_keys=True), flush=True)

    best_inner_batch = min(inner_batch_results, key=lambda x: x["total_wall_seconds"])["inner_batch"]

    for queue_depth in args.queue_depth_values:
        result = run_case(
            actors=actors,
            candidates_per_task=int(args.candidates_per_task),
            num_tasks=int(args.num_tasks),
            inner_batch=int(best_inner_batch),
            max_in_flight=int(queue_depth),
        )
        queue_depth_results.append(result)
        write_payload(
            output_path=output_path,
            q_path=q_path,
            v_path=v_path,
            args=args,
            auto_inner_batch_per_actor=auto_inner_batch_per_actor,
            default_queue_depth=default_queue_depth,
            inner_batch_results=inner_batch_results,
            queue_depth_results=queue_depth_results,
        )
        print("QUEUE", json.dumps(result, sort_keys=True), flush=True)

    write_payload(
        output_path=output_path,
        q_path=q_path,
        v_path=v_path,
        args=args,
        auto_inner_batch_per_actor=auto_inner_batch_per_actor,
        default_queue_depth=default_queue_depth,
        inner_batch_results=inner_batch_results,
        queue_depth_results=queue_depth_results,
    )
    print(f"WROTE {output_path}", flush=True)
    ray.shutdown()


if __name__ == "__main__":
    main()
