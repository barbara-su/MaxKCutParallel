# Technical Report: `parallel_rank_r_dir_gpu_fullgpu.py`

## 1) Objective and Context

This report explains how we moved from the older CPU-heavy rank-`r` directory solver (`src/parallel_rank_r_dir.py`) to the current full-GPU candidate-generation solver (`src/parallel_rank_r_dir_gpu_fullgpu.py`).

The goal is to solve, for each instance `(Q, V)`, a discrete assignment problem over `K` roots of unity:

$$
z_i \in \{\omega^0,\omega^1,\ldots,\omega^{K-1}\}, \quad \omega = e^{2\pi i/K}
$$

with score

$$
\text{score}(z) = \operatorname{Re}(z^H Q z).
$$

For rank-`r>1`, candidate generation is combinatorial over \((2r-1)\)-tuples from `K*n` rows of `V_tilde`, so runtime is dominated by how fast we can:

1. enumerate combinations,
2. build feasible candidate directions,
3. quantize to roots,
4. score at high throughput.

---

## 2) High-Level Evolution (Why This File Is Faster)

Compared to `src/parallel_rank_r_dir.py`, the full-GPU solver shifts almost all heavy work onto GPU:

- **Old path**: CPU Ray tasks build candidates and do expensive per-candidate math, then GPU scores.
- **Full-GPU path**: GPU actor builds index batches, generates candidates, applies feasibility logic, quantizes, scores, and does final exact refinement.

The major design shifts are:

1. **GPU-resident static tensors** (`V`, `Q`, `V_tilde`) per actor.
2. **GPU-side combination unranking** (no Python tuple enumeration in hot path).
3. **Batched small linear solves on GPU** for null-vector construction.
4. **Vectorized quantization + one-GEMM scoring**.
5. **Exact baseline refinement for the top candidate** to preserve correctness.

---

## 3) System Architecture

### 3.1 Driver Layer

Main entrypoint: `main()` and recursive solver `process_rankr_recursive_fullgpu(...)`.

- Discovers `(Q, V)` instance pairs.
- Creates one Ray GPU actor per visible GPU.
- For `r>=2`, calls `process_rankr_single_fullgpu(...)`.
- For `r==1`, reuses optimized rank-1 GPU path (`process_rank_1_parallel_gpu`).

Relevant code:
- `src/parallel_rank_r_dir_gpu_fullgpu.py`: `process_rankr_single_fullgpu`, `process_rankr_recursive_fullgpu`, `main`.

### 3.2 Actor Layer

`RankRGPUActor` owns all compute-heavy kernels and persistent GPU state:

- `set_instance(V, Q, V_tilde)` uploads tensors.
- `score_rank_batch(start_rank, batch_size, r)` builds combinations on GPU and scores.
- `score_batch(...)` and `score_k_batch(...)` keep compatibility with other paths.

Relevant code:
- `src/parallel_rank_r_dir_gpu_fullgpu.py`: class `RankRGPUActor`.

---

## 4) Step-by-Step Pipeline in `process_rankr_single_fullgpu`

### Step A: Precompute and broadcast `V_tilde`

`V_tilde = compute_vtilde(V)` is built once per rank-level and sent to each GPU actor.

Why: avoid rebuilding per batch; keep candidate-generation inputs resident on device.

### Step B: Split the total combination space into rank ranges

Total combinations:

$$
N_{\text{comb}} = \binom{K n}{2r-1}
$$

Each task represents a contiguous rank interval:

$$
[\text{start\_rank},\, \text{start\_rank}+B)
$$

with `B = candidates_per_task` (capped on last batch).

### Step C: Submit GPU actor requests directly

The driver does **not** launch an extra CPU Ray task layer for candidate construction. It directly submits actor calls (`score_rank_batch.remote(...)`) and drains with `ray.wait`.

This removes nested scheduling and extra serialization overhead.

### Step D: Aggregate best score and timing stats

For each completed batch, update:

- global best `(score, k, z)`,
- feasible ratio,
- average GPU batch wall time.

Progress is logged every 10 completed tasks.

---

## 5) Low-Level GPU Techniques (Core Optimizations)

## 5.1 TF32 acceleration for GEMM-heavy sections

In actor init:

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
```

Effect: faster matmul kernels on Ampere+ while maintaining sufficient accuracy for this workload.

---

## 5.2 GPU combination unranking (`_build_index_batch_gpu`)

Instead of Python `itertools.combinations`, the actor un-ranks combinations directly on GPU using a precomputed binomial LUT (`self._comb_lut`).

For each output position, it advances `x` until

$$
\text{remaining} < \binom{N-x-1}{k-pos-1}.
$$

This yields a tensor `I` with shape `(B, 2r-1)` on GPU.

Why it matters:

- avoids Python tuple generation,
- avoids host-side batch assembly,
- keeps enumeration data-path on device.

---

## 5.3 Batched null-space construction by pivoted solve (`_build_null_vectors_pivot`)

For each candidate row-block `VI` with shape `(m, d)` where `m=d-1=2r-1`:

- For each pivot column `p`, solve
  $$
  A x = b,
  $$
  where `A` is `VI` without pivot column, `b = -VI[:, p]`.
- Reconstruct `c` with `c[p]=1`, remaining entries from `x`.
- Pick pivot with lowest residual `||VI c||_2`.
- Normalize `c`.

This is fully batched across `B` candidates via `torch.linalg.solve_ex`.

Why it matters:

- replaces slow per-candidate CPU null-space routines,
- exploits GPU for many tiny solves in parallel,
- keeps validity mask (`valid_null`) in tensor form.

---

## 5.4 Batched phase/sign feasibility logic (`_determine_phi_sign_torch`)

Implements a batched equivalent of baseline `determine_phi_sign_c` for small dimension (`r<=3`).

Feasibility gate uses:

$$
-\frac{\pi}{K} < \phi_{2r-2} \le \frac{\pi}{K}.
$$

Only candidates passing null-solve + angle feasibility move forward.

Why it matters:

- preserves baseline geometry constraints,
- stays on GPU and vectorized across batch.

---

## 5.5 Quantization to roots and fallback used-vertex override

Given complex candidates `C`:

1. Compute
   $$
   Y = V C^T.
   $$
2. Quantize each `Y_{i,b}` to nearest root index via angle rounding.
3. Apply baseline-compatible fallback used-vertex override using unique vertices from `I // K` (deduplicated per candidate).

This override is cheap and vectorized, and matches baseline semantics for the fallback path.

---

## 5.6 One-GEMM real scoring trick

Instead of two separate GEMMs for real/imag parts, stack them once:

```python
Zcat[:, :B] = zr
Zcat[:, B:2*B] = zi
QZcat = Q @ Zcat
```

Then

$$
\text{scores}_b = z_r^T Q z_r + z_i^T Q z_i.
$$

A reusable workspace buffer `self._zcat_buf` avoids repeated allocation.

Why it matters:

- higher kernel utilization,
- less launch overhead,
- lower allocation churn.

---

## 5.7 Exact baseline refinement of batch winner (`_exact_refine_best_candidate`)

To close residual correctness gaps, the actor re-runs exact baseline refinement for the **best candidate of the batch**:

- build `VI` for selected candidate,
- for each used vertex, try fixed-angle refit by removing one relevant row,
- if refit fails, fallback to base `c`,
- override corresponding `k[v]`, re-score exactly.

Only one candidate per batch is refined, so cost is controlled while correctness improves.

Why it matters:

- recovers baseline behavior where fixed-angle correction changes winner quality,
- avoids paying full exact-refinement cost for all candidates.

---

## 6) Rank-Recursive Strategy

`process_rankr_recursive_fullgpu` computes best for rank `r`, then recurses to `r-1` and keeps the better score.

This preserves the existing algorithmic structure while replacing each rank-level engine with faster full-GPU implementation.

---

## 7) Data Movement Strategy

The implementation minimizes transfer volume by design:

- Upload `V`, `Q`, `V_tilde` once per instance/rank stage.
- Send only scalar range metadata per batch (`start_rank`, `batch_size`, `r`).
- Build `I` and all downstream candidate tensors on GPU.
- Return only best result triplet `(score, best_k, best_z)` + feasible count.

This is a major reason the new path scales better than CPU-candidate pipelines.

---

## 8) Numerical Choices and Practical Details

- `precision=16/32/64` currently maps to `complex64` + `float32` in actor dtypes.
- Scores are rounded before returning best to absorb TF32-level noise and preserve discrete objective consistency.
- `Q` is assumed real in the fast path.

---

## 9) Complexity and Throughput Intuition

For each batch of size `B`, dominant costs are:

1. Batched small solves: roughly `O(B * d^3)` with small `d=2r`.
2. Projection GEMM: `Y = V C^T` with shape `(n, r) @ (r, B)`.
3. Scoring GEMM: `Q @ Zcat` with shape `(n, n) @ (n, 2B)`.

Because `r` is small and GEMMs are well-optimized, this design is much better matched to GPU hardware than Python-heavy CPU candidate loops.

---

## 10) Minimal Code Map

- High-level driver / recursion:
  - `process_rankr_single_fullgpu`
  - `process_rankr_recursive_fullgpu`
  - `main`
- GPU index generation:
  - `_build_index_batch_gpu`
  - `_comb_lut` creation in `set_instance`
- GPU candidate construction:
  - `_build_null_vectors_pivot`
  - `_determine_phi_sign_torch`
  - `_ctilde_to_complex_torch`
- Scoring and correctness:
  - `_score_index_batch_tensor`
  - `_exact_refine_best_candidate`
  - `_find_intersection_fixed_angle_single`

All of the above live in `src/parallel_rank_r_dir_gpu_fullgpu.py`.

---

## 11) Summary

`parallel_rank_r_dir_gpu_fullgpu.py` is fast because it restructures the pipeline around GPU locality and batch math:

- enumerate on GPU,
- construct candidates on GPU,
- quantize/score on GPU,
- refine winner for correctness.

It keeps the original recursive rank logic but replaces the slowest layer (CPU candidate generation + Python overhead) with tensorized GPU kernels and direct actor scheduling.
