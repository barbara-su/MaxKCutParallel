#!/bin/bash
#SBATCH --job-name=multi-nodes-rank-1-dir
#SBATCH --output=logs/multi-nodes-rank-1-dir-%j.out
#SBATCH --error=logs/multi-nodes-rank-1-dir-%j.err

#SBATCH --nodes=4
#SBATCH --tasks-per-node=1
#SBATCH --partition=commons
#SBATCH --cpus-per-task=100
#SBATCH --mem=300G
#SBATCH --time=23:00:00

# Loop over Q_<n>_seed_<s>.npy / V_<n>_seed_<s>.npy pairs and run rank-1.
#
# Args:
#   1: qv_dir (default "graphs/graphs_rank_1")
#   2: out_root (default "results_rank_1_dir")
#   3: precision in {16,32,64} (default 64)
#   4: candidates_per_task (default 10)
#
# Example:
#   sbatch multi_node_rank_1_dir.sh graphs/graphs_rank_1 results/rank1_all 64 10

set -euo pipefail

QV_DIR=${1:-graphs/graphs_rank_1}
OUT_ROOT=${2:-results_rank_1_dir}
PRECISION=${3:-64}
CANDIDATES_PER_TASK=${4:-10}

echo "Job started on $(hostname)"
echo "SLURM_JOB_NUM_NODES: ${SLURM_JOB_NUM_NODES:-}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-}"
echo "Using qv_dir = $QV_DIR"
echo "Using out_root = $OUT_ROOT"
echo "Using precision = $PRECISION"
echo "Using candidates_per_task = $CANDIDATES_PER_TASK"

# BLAS threading: avoid oversubscription (Ray parallelism provides concurrency)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Clean environment
module purge || true
unset LD_PRELOAD || true
export XALT_EXECUTABLE_TRACKING=0
export XALT_RUNNABLE=0

# Repo + conda
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

# Ray cluster setup info
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
port=5050

head_ip=$(srun -N1 -n1 -w "$head_node" hostname -I | awk '{print $1}')
ip_head="${head_ip}:${port}"
export ip_head
echo "Ray head GCS address: $ip_head"

# Only one Slurm process should run the directory loop.
if [[ "${SLURM_PROCID:-0}" -ne 0 ]]; then
  echo "SLURM_PROCID=${SLURM_PROCID} not 0; exiting to avoid duplicate directory loop."
  exit 0
fi

shopt -s nullglob
q_files=("$QV_DIR"/Q_*_seed_*.npy)
if [[ ${#q_files[@]} -eq 0 ]]; then
  echo "ERROR: No Q_*_seed_*.npy found in $QV_DIR" >&2
  exit 3
fi

echo "Found ${#q_files[@]} Q files in $QV_DIR"

# Temp root (prefer node-local tmp if available)
TMP_ROOT="${TMPDIR:-/tmp}/rank1_qv_symlinks_${SLURM_JOB_ID}"
mkdir -p "$TMP_ROOT"

for q_path in "${q_files[@]}"; do
  base=$(basename "$q_path")  # e.g., Q_20_seed_7.npy

  # Parse n and seed from filename
  # Expect: Q_<n>_seed_<seed>.npy
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

  # Per-run temp directory with the names your python expects: Q_<n>.npy and V_<n>.npy
  run_tmp="$TMP_ROOT/n${n}_seed${seed}"
  mkdir -p "$run_tmp"
  ln -sf "$q_path" "$run_tmp/Q_${n}.npy"
  ln -sf "$v_path" "$run_tmp/V_${n}.npy"

  out_dir="$OUT_ROOT/n${n}/seed${seed}"
  mkdir -p "$out_dir"

  echo "============================================================"
  echo "Running n=$n seed=$seed"
  echo "Q=$q_path"
  echo "V=$v_path"
  echo "TMP=$run_tmp"
  echo "OUT=$out_dir"
  echo "============================================================"

  # Run the exact same symmetric_run pattern per instance
  srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
    python -m read_only.symmetric_run \
      --address "$ip_head" \
      --min-nodes "$SLURM_JOB_NUM_NODES" \
      --num-cpus "$SLURM_CPUS_PER_TASK" \
      -- \
      python -u src/parallel_rank_1.py \
        --n "$n" \
        --seed "$seed" \
        --graph_dir "$run_tmp" \
        --results_dir "$out_dir" \
        --precision "$PRECISION" \
        --candidates_per_task "$CANDIDATES_PER_TASK"

  # Optional cleanup per run to keep tmp small
  rm -rf "$run_tmp"
done

echo "All done."


# QV_DIR=${1:-graphs/graphs_rank_1}
# OUT_ROOT=${2:-results_rank_1_dir}
# PRECISION=${3:-64}
# CANDIDATES_PER_TASK=${4:-10}

# sbatch experiments/multi_node_rank_1_dir.sh graphs/erdos_renyi/rank_1/p01/n20 results/erdos_renyi/rank_1/p01/n20 32 10
# sbatch experiments/multi_node_rank_1_dir.sh graphs/erdos_renyi/rank_1/p01/n50 results/erdos_renyi/rank_1/p01/n50 32 10
# sbatch experiments/multi_node_rank_1_dir.sh graphs/erdos_renyi/rank_1/p01/n100 results/erdos_renyi/rank_1/p01/n100 32 10

# sbatch experiments/multi_node_rank_1_dir.sh graphs/erdos_renyi/rank_1/p025/n20 results/erdos_renyi/rank_1/p025/n20 32 10
# sbatch experiments/multi_node_rank_1_dir.sh graphs/erdos_renyi/rank_1/p025/n50 results/erdos_renyi/rank_1/p025/n50 32 10
# sbatch experiments/multi_node_rank_1_dir.sh graphs/erdos_renyi/rank_1/p025/n100 results/erdos_renyi/rank_1/p025/n100 32 10

# sbatch experiments/multi_node_rank_1_dir.sh graphs/erdos_renyi/rank_1/p05/n20 results/erdos_renyi/rank_1/p05/n20 32 10
# sbatch experiments/multi_node_rank_1_dir.sh graphs/erdos_renyi/rank_1/p05/n50 results/erdos_renyi/rank_1/p05/n50 32 10
# sbatch experiments/multi_node_rank_1_dir.sh graphs/erdos_renyi/rank_1/p05/n100 results/erdos_renyi/rank_1/p05/n100 32 10

# sbatch experiments/multi_node_rank_1_dir.sh graphs/erdos_renyi/rank_1/p075/n20 results/erdos_renyi/rank_1/p075/n20 32 10
# sbatch experiments/multi_node_rank_1_dir.sh graphs/erdos_renyi/rank_1/p075/n50 results/erdos_renyi/rank_1/p075/n50 32 10
# sbatch experiments/multi_node_rank_1_dir.sh graphs/erdos_renyi/rank_1/p075/n100 results/erdos_renyi/rank_1/p075/n100 32 10
