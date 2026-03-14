# Triton Trials

These files were moved from the old top-level `version_checks/` directory on 2026-03-14.
They are historical full-GPU snapshots from the Triton integration period and are kept for
regression checks and code archaeology.

- `parallel_rank_r_dir_gpu_fullgpu_pre_triton.py`: pure PyTorch snapshot before Triton quantization integration.
- `parallel_rank_r_dir_gpu_fullgpu_pre_next_fuse_20260303.py`: intermediate 2026-03-03 snapshot after Triton work but before the next fusion step.
- `parallel_rank_r_dir_gpu_fullgpu_post_triton.py`: post-Triton snapshot kept for comparison against earlier variants.
- `parallel_rank_r_dir_gpu_fullgpu_checkpoint_20260303_h100_bench.py`: checkpoint-era snapshot used during the 2026-03-03 H100 benchmarking pass.
