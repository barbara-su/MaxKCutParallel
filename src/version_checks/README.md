# Version Checks

Historical full-GPU solver snapshots live here for regression checks and code archaeology.
The active implementation is [`src/parallel_rank_r_dir_gpu_fullgpu.py`](/scratch/bs82/max-k-cut-parallel/src/parallel_rank_r_dir_gpu_fullgpu.py).

Layout:

- `triton_trials/`: snapshots moved from the old top-level `version_checks/` directory on 2026-03-14.
- `v0/` through `v6/`: older archived implementation checkpoints.
- `numba/`: older Numba-focused variants.
