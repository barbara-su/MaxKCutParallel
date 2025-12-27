#!/bin/bash
#SBATCH --job-name=rank1-maxcut-dir
#SBATCH --output=logs/rank-1-maxcut-dir-%j.out
#SBATCH --error=logs/rank-1-maxcut-dir-%j.err

#SBATCH --partition=commons
#SBATCH --cpus-per-task=100
#SBATCH --mem=200G
#SBATCH --time=23:00:00

# Single-node rank-1 over ALL seeded Q/V pairs in a directory.
#
# Your files look like:
#   Q_<n>_seed_<s>.npy
#   V_<n>_seed_<s>.npy
#
# But python expects inside --graph_dir:
#   Q_<n>.npy
#   V_<n>.npy
#
# So we loop, symlink to expected names in a temp dir, and run python once per (n, seed).
#
# Args:
#   1: qv_dir (default "graphs/graphs_rank_1")
#   2: out_root (default "results_rank_1_dir")
#   3: precision in {16,32,64} (default 64)
#   4: candidates_per_task (default 10)
#
# Example:
#   sbatch single_node_rank_1.sh graphs/graphs_rank_1 results/rank1_single 64 10

set -euo pipefail

QV_DIR=${1:-graphs/graphs_rank_1}
OUT_ROOT=${2:-results_rank_1_dir}
PRECISION=${3:-64}
CANDIDATES_PER_TASK=${4:-10}

echo "Job started on $(hostname)"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-}"
echo "Using qv_dir = $QV_DIR"
echo "Using out_root = $OUT_ROOT"
echo "Using precision = $PRECISION"
echo "Using candidates_per_task = $CANDIDATES_PER_TASK"

# openblas
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Clean environment
module purge || true
unset LD_PRELOAD || true
export XALT_EXECUTABLE_TRACKING=0
export XALT_RUNNABLE=0

# OPTIONAL but very helpful for quota: push caches/tmp off $HOME.
# Uncomment and adjust if you have /scratch.
# export TMPDIR=/scratch/$USER/tmp
# export XDG_CACHE_HOME=/scratch/$USER/.cache
# export PIP_CACHE_DIR=/scratch/$USER/pip-cache
# mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR"

# Conda
cd /home/bs82/max-k-cut-parallel/ || exit 1
mkdir -p logs
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ./rayenv

QV_DIR=$(readlink -f "$QV_DIR")
OUT_ROOT=$(readlink -m "$OUT_ROOT")
mkdir -p "$OUT_ROOT"

if [[ ! -d "$QV_DIR" ]]; then
  echo "ERROR: qv_dir does not exist: $QV_DIR" >&2
  exit 2
fi

# Start ray (single-node)
ray stop || true
ray start --head --port=5050

shopt -s nullglob
q_files=("$QV_DIR"/Q_*_seed_*.npy)
if [[ ${#q_files[@]} -eq 0 ]]; then
  echo "ERROR: No Q_*_seed_*.npy found in $QV_DIR" >&2
  exit 3
fi

echo "Found ${#q_files[@]} Q files in $QV_DIR"

TMP_ROOT="${TMPDIR:-/tmp}/rank1_qv_symlinks_${SLURM_JOB_ID}"
mkdir -p "$TMP_ROOT"

for q_path in "${q_files[@]}"; do
  base=$(basename "$q_path")  # Q_20_seed_7.npy

  if [[ "$base" =~ ^Q_([0-9]+)_seed_([0-9]+)\.npy$ ]]; then
    n="${BASH_REMATCH[1]}"
    seed="${BASH_REMATCH[2]}"
  else
    echo "WARN: unexpected filename format (skipping): $base"
    continue
  fi

  v_path="$QV_DIR/V_${n}_seed_${seed}.npy"
  if [[ ! -f "$v_path" ]]; then
    echo "WARN: missing V for n=$n seed=$seed: $v_path (skipping)"
    continue
  fi

  run_tmp="$TMP_ROOT/n${n}_seed${seed}"
  mkdir -p "$run_tmp"
  ln -sf "$q_path" "$run_tmp/Q_${n}.npy"
  ln -sf "$v_path" "$run_tmp/V_${n}.npy"

  out_dir="$OUT_ROOT"
  mkdir -p "$out_dir"

  echo "============================================================"
  echo "Running n=$n seed=$seed"
  echo "TMP=$run_tmp"
  echo "OUT=$out_dir"
  echo "============================================================"

  python -u src/parallel_rank_1.py \
    --n "$n" \
    --seed 42 \
    --graph_dir "$run_tmp" \
    --results_dir "$out_dir" \
    --precision "$PRECISION" \
    --candidates_per_task "$CANDIDATES_PER_TASK"

  rm -rf "$run_tmp"
done

echo "Job complete."

# sbatch experiments/single_node_rank_1_dir.sh graphs/erdos_renyi/rank_1/p01/n20 results/erdos_renyi/rank_1/p01/n20 32 10
# sbatch experiments/single_node_rank_1_dir.sh graphs/erdos_renyi/rank_1/p01/n50 results/erdos_renyi/rank_1/p01/n50 32 10
# sbatch experiments/single_node_rank_1_dir.sh graphs/erdos_renyi/rank_1/p01/n100 results/erdos_renyi/rank_1/p01/n100 32 10

# sbatch experiments/single_node_rank_1_dir.sh graphs/erdos_renyi/rank_1/p025/n20 results/erdos_renyi/rank_1/p025/n20 32 10
# sbatch experiments/single_node_rank_1_dir.sh graphs/erdos_renyi/rank_1/p025/n50 results/erdos_renyi/rank_1/p025/n50 32 10
# sbatch experiments/single_node_rank_1_dir.sh graphs/erdos_renyi/rank_1/p025/n100 results/erdos_renyi/rank_1/p025/n100 32 10

# sbatch experiments/single_node_rank_1_dir.sh graphs/erdos_renyi/rank_1/p05/n20 results/erdos_renyi/rank_1/p05/n20 32 10
# sbatch experiments/single_node_rank_1_dir.sh graphs/erdos_renyi/rank_1/p05/n50 results/erdos_renyi/rank_1/p05/n50 32 10
# sbatch experiments/single_node_rank_1_dir.sh graphs/erdos_renyi/rank_1/p05/n100 results/erdos_renyi/rank_1/p05/n100 32 10

# sbatch experiments/single_node_rank_1_dir.sh graphs/erdos_renyi/rank_1/p075/n20 results/erdos_renyi/rank_1/p075/n20 32 10
# sbatch experiments/single_node_rank_1_dir.sh graphs/erdos_renyi/rank_1/p075/n50 results/erdos_renyi/rank_1/p075/n50 32 10
# sbatch experiments/single_node_rank_1_dir.sh graphs/erdos_renyi/rank_1/p075/n100 results/erdos_renyi/rank_1/p075/n100 32 10

# sbatch experiments/single_node_rank_1_dir.sh graphs/regular_graph/3_regular_graph_rank_1/n20 results/regular_graph/3_regular_graph_rank_1/n20 32 10
# sbatch experiments/single_node_rank_1_dir.sh graphs/regular_graph/3_regular_graph_rank_1/n50 results/regular_graph/3_regular_graph_rank_1/n50 32 10
# sbatch experiments/single_node_rank_1_dir.sh graphs/regular_graph/3_regular_graph_rank_1/n100 results/regular_graph/3_regular_graph_rank_1/n100 32 10

# sbatch experiments/single_node_rank_1_dir.sh graphs/regular_graph/5_regular_graph_rank_1/n20 results/regular_graph/5_regular_graph_rank_1/n20 32 10
# sbatch experiments/single_node_rank_1_dir.sh graphs/regular_graph/5_regular_graph_rank_1/n50 results/regular_graph/5_regular_graph_rank_1/n50 32 10
# sbatch experiments/single_node_rank_1_dir.sh graphs/regular_graph/5_regular_graph_rank_1/n100 results/regular_graph/5_regular_graph_rank_1/n100 32 10

