# max-k-cut-parallel

## Latest Full-GPU Rank-2 Status

On 2026-03-14, the first graph from `/scratch/bs82/graphs/gset_rank_2` (`Q_gset_1.npy`, `V_gset_1.npy`, `n=800`) was used to benchmark the current full-GPU rank-2 path on 4 H200 GPUs.

- Current fixed auto run: `results/gpu/gset_rank_2_h2004_q1_auto_fixed/Q_gset_1_r2.json`
- Best score: `41939`
- End-to-end runtime: `336.53s` (`5m 37s`)
- Rank-2 search time: `330.62s`
- Average rank-2 logical batch time at `candidates_per_task=100000000`: `46.48s`
- `max_in_flight_gpu_requests_requested=0` -> `effective_max_in_flight_gpu_requests=8`
- `gpu_inner_batch_size_requested=0` -> `effective_gpu_inner_batch_size=1048576`

Current JSON reporting is explicit: it records both the requested value and the resolved effective value, and it also marks each knob as `auto` or `manual`. The older pre-fix run is still available at `results/gpu/gset_rank_2_h2004_q1/Q_gset_1_r2.json`, but its `gpu_inner_batch_size: 0` field only recorded the requested CLI value and not the resolved actor-side auto chunk.

## H200 Tuning For `gpu_inner_batch_size`

`gpu_inner_batch_size` is the per-GPU-actor inner scoring chunk used inside `RankRGPUActor.score_rank_batch()`. A logical rank-2 task can contain `100000000` candidates, but the actor still processes that logical task in smaller GPU chunks so workspace size stays bounded. This knob controls the size of those inner chunks, so it directly changes GEMM size and launch overhead.

`gpu_inner_batch_size=0` now means "use the actor's automatic memory-aware chunk size". That behavior was fixed in `src/parallel_rank_r_dir_gpu_fullgpu.py` on 2026-03-14.

Measured on the first `gset_rank_2` graph, with 4 H200 GPUs, `candidates_per_task=100000000`, `num_tasks=8`, and auto queue depth `8`:

- `262144` -> `295.26s`
- `335544` -> `242.91s`
- `524288` -> `174.81s`
- `1048576` -> `112.00s`
- `0` -> `111.72s`

Artifact:

- `results/gpu/tuning/rank2_inner_batch_h2004_q1_20260314.json`

For this node class, actor auto resolved to:

- `auto_gpu_inner_batch_size_per_actor = [1048576, 1048576, 1048576, 1048576]`

Recommendation for 4xH200 rank-2 runs like `Q_gset_1.npy`:

- Leave `gpu_inner_batch_size=0`.
- If you want a fixed explicit value instead of auto, use `1048576`.

How to set it empirically on a new GPU type or graph family:

1. Run one representative graph with `gpu_inner_batch_size=0` and record the resolved auto value.
2. Sweep a small band around that value, for example `auto/4`, `auto/2`, `auto`, and `2*auto` if memory allows.
3. Compare `total_wall_seconds`, not just per-task compute time.
4. Keep `0` unless a fixed explicit value is measurably better on your hardware.

Example tuning command:

```bash
python -u src/others/benchmark_fullgpu_rank2_tuning.py \
  --q_path /scratch/bs82/graphs/gset_rank_2/Q_gset_1.npy \
  --v_path /scratch/bs82/graphs/gset_rank_2/V_gset_1.npy \
  --output_path results/gpu/tuning/rank2_inner_batch_manual.json \
  --num_gpus 4 \
  --num_cpus 16 \
  --candidates_per_task 100000000 \
  --num_tasks 8 \
  --inner_batch_values 262144 335544 524288 1048576 0 \
  --queue_depth_values 4 8 12
```

## `max_in_flight_gpu_requests`

In the full-GPU path, this is only the queue depth for outstanding GPU actor requests. It is not a CPU-worker count and it does not change the rank-2 math. The misleading `max_in_flight_cpu` alias has been removed from the current full-GPU CLI.

Measured on the same 4xH200 tuning run with auto inner batch:

- queue depth `4` -> `111.26s`
- queue depth `8` -> `111.38s`
- queue depth `12` -> `112.07s`

That spread is negligible, so this is not a useful primary tuning knob for the current rank-2 H200 path. Practical recommendation:

- Leave `max_in_flight_gpu_requests=0`.
- Tune this only after `gpu_inner_batch_size` is already settled.

## ETA For This Setting

The old README ETA was too optimistic for large `n` because it only scaled with the number of rank-2 combinations. For dense `Q`, the dominant rank-2 score kernel also pays an `O(n^2)` matrix multiplication cost per chunk, so a better rough model is:

```python
import math

CALIBRATED_N = 800
CALIBRATED_SECONDS = 336.53439927101135

def estimate_rank2_fullgpu_eta_seconds(num_vertices: int) -> float:
    """Conservative ETA for 4xH200, rank=2, K=3, precision=32.

    `num_vertices` means graph vertices, not Slurm nodes.
    Calibration uses the current completed n=800 full run above.
    """
    combination_ratio = (
        math.comb(3 * num_vertices, 3)
        / math.comb(3 * CALIBRATED_N, 3)
    )
    dense_score_ratio = (num_vertices / CALIBRATED_N) ** 2
    return CALIBRATED_SECONDS * combination_ratio * dense_score_ratio
```

This is still only an estimate, but it is much closer to the actual dense full-GPU scaling because it includes both:

- `C(3n, 3)` rank-2 search growth
- the extra `n^2` cost from dense `Q @ Zcat` scoring

Examples from that calibration:

- `n=800` -> `336.53s` (`5m 37s`)
- `n=1000` -> `1027.28s` (`17m 7s`)
- `n=1200` -> `2556.62s` (`42m 37s`)
- `n=2000` -> `32889.36s` (`9h 8m 9s`)
- `n=5000` -> `3212815.18s` (`37d 4h 26m`)
- `n=10000` -> `102820367.99s` (`1190d 1h 12m`, about `3.26 years`)

Treat these as rough dense-model ETAs, not hard guarantees. Graph structure, feasibility pruning, scheduler noise, and later rank-1 recursion can move the final wall time.

## Using `multi_node_rank_r_dir_gpu_fullgpu.sh` Effectively

The launcher scans sorted `Q*.npy` / `V*.npy` pairs and processes them in order.

- Use `max_instances=1` and `start_index=0` to run only the first graph in a directory.
- Keep `precision=32` for the current full-GPU rank-2 path.
- `candidates_per_task=100000000` worked for the first `gset_rank_2` graph on 4 H200 GPUs.
- Leave `gpu_inner_batch_size=0`. On this H200 benchmark, auto resolved to `1048576` and matched the best wall time.
- Leave `max_in_flight_gpu_requests=0`. On this H200 benchmark, queue depths `4`, `8`, and `12` were effectively tied.
- Use `skip_existing=1` for resumable reruns.

Example: run only the first graph in `/scratch/bs82/graphs/gset_rank_2` and save results under `results/gpu`:

```bash
sbatch --nodelist=bg2u24g1 --gres=gpu:h200:4 \
  experiments/multi_node_rank_r_dir_gpu_fullgpu.sh \
  /scratch/bs82/graphs/gset_rank_2 \
  results/gpu/gset_rank_2_h2004_q1_direct \
  2 3 32 100000000 0 4 0 0 1 0 1
```

Argument mapping for the tail of that command:

- `0`: `max_in_flight_gpu_requests`
- `0`: `gpu_inner_batch_size` (`0` means actor auto; on 4xH200 it resolved to `1048576`)
- `1`: `max_instances`
- `0`: `start_index`
- `1`: `skip_existing`
