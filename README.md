# Low-Rank Max-K-Cut: GPU-Parallel Solver

A GPU-accelerated solver for the Max-K-Cut problem that exploits low-rank structure in the objective matrix. The algorithm decomposes the combinatorial search into a rank-1 base case (solved exactly via GPU-parallel enumeration) and a rank-2 extension (searched over direction pairs with GPU-batched scoring), then recurses to handle arbitrary rank. By distributing work across heterogeneous GPU clusters -- including older architectures like P100 -- the solver scales to graphs with thousands of vertices while producing solutions with provable approximation guarantees.

**Paper:** [Exploiting Low-Rank Structure in Max-K-Cut Problems](https://arxiv.org/pdf/2602.20376)

**Blog posts:**
- [Exploiting Low-Rank Structure in Max-K-Cut Problems](https://akyrillidis.github.io/explore-quantum/MaxKCut.html) — Algorithm overview, theory, and benchmark results
- [What Can 15 Obsolete GPUs Do for Combinatorial Optimization?](https://akyrillidis.github.io/explore-quantum/LowRankMaxCut_GPU.html) — GPU implementation, scaling experiments, and interactive visualizations

---

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.12+ (works on PyTorch 1.12 through 2.4+)
- CUDA-capable GPU (P100 or newer; no Tensor Cores required)
- [Ray](https://docs.ray.io/) (for single-node multi-GPU runs)
- NumPy, SciPy

Install dependencies:

```bash
pip install torch numpy scipy ray
```

### Single-Node Rank-1 Solve

Run the exact rank-1 solver on a single machine with multiple GPUs:

```bash
python src/parallel_rank_1_gpu.py \
    --q_path data/Q.npy \
    --v_path data/V.npy \
    --K 3 \
    --num_gpus 4 \
    --precision 32
```

### Single-Node Rank-2 Solve (Full GPU)

Run the rank-2 search entirely on GPU using Ray actors:

```bash
python src/parallel_rank_r_dir_gpu_fullgpu.py \
    --q_path data/Q.npy \
    --v_path data/V.npy \
    --rank 2 --K 3 \
    --num_gpus 4 \
    --precision 32 \
    --candidates_per_task 100000000
```

### Multi-Machine Heterogeneous Cluster

For distributing work across multiple machines (e.g., a mix of P100 and other GPUs), use the coordinator/worker architecture:

1. **Configure machines** in `src/coordinator.py` (edit the `MACHINES` list with hostnames, GPU counts, and paths).

2. **Launch the coordinator:**

```bash
python src/coordinator.py \
    --q_path data/Q.npy \
    --v_path data/V.npy \
    --rank 2 --K 3
```

The coordinator splits the combination space proportionally across machines, launches `worker.py` on each via SSH, and collects the global best solution.

3. **Direct worker invocation** (on a single machine):

```bash
python src/worker.py \
    --q_path Q.npy --v_path V.npy --vtilde_path Vtilde.npy \
    --start_rank 0 --end_rank 1000000 \
    --rank 2 --K 3 \
    --num_gpus 4 --out result.json
```

---

## Hardware Requirements

- **Minimum:** Any single CUDA GPU with 8+ GB VRAM (e.g., GTX 1080, P100)
- **Recommended:** Multiple GPUs for parallel speedup
- **No Tensor Cores needed** -- the solver uses standard FP32 matrix operations
- **Tested on:** NVIDIA P100 (16 GB), RTX 6000 Ada (48 GB), H200 (80 GB)
- **Heterogeneous clusters supported:** The coordinator/worker architecture distributes work across machines with different GPU types

---

## File Structure

```
.
├── src/
│   ├── parallel_rank_1_gpu.py          # Exact rank-1 solver (GPU-parallel)
│   ├── parallel_rank_r_dir_gpu_fullgpu.py  # Full-GPU rank-2 search with Ray
│   ├── worker.py                       # Ray-free multi-GPU worker (PyTorch 1.12+)
│   ├── coordinator.py                  # Cross-machine coordinator (SSH-based)
│   ├── utils.py                        # Shared utilities (Q generation, scoring, etc.)
│   ├── baselines.py                    # SDP and other baseline solvers
│   ├── graph_generators/               # Graph instance generators
│   ├── post_process/                   # Result analysis and plotting
│   └── others/                         # Benchmarking and tuning scripts
├── experiments/
│   ├── single_node_rank_1.sh           # Single-node rank-1 launcher
│   ├── multi_node_rank_r_dir_gpu_fullgpu.sh  # Multi-node full-GPU launcher
│   ├── generate_graphs/                # Graph generation scripts
│   └── tests/                          # Correctness tests
├── gset/                               # G-set benchmark instances
├── results/                            # Output directory for solutions
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{stevens2025exploiting,
  title={Exploiting Low-Rank Structure in Max-K-Cut Problems},
  author={Stevens, Ria and Liao, Fangshuo and Su, Barbara and Li, Jianqiang and Kyrillidis, Anastasios},
  journal={arXiv preprint arXiv:2602.20376},
  year={2025}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Authors

- **Ria Stevens**
- **Fangshuo Liao**
- **Barbara Su**
- **Jianqiang Li**
- **Anastasios Kyrillidis**
