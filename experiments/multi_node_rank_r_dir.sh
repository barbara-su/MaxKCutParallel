#!/bin/bash

#SBATCH --job-name=multi-nodes-rank-r-dir
#SBATCH --output=logs/multi-nodes-rank-r-dir-%j.out
#SBATCH --error=logs/multi-nodes-rank-r-dir-%j.err

#SBATCH --nodes=6
#SBATCH --tasks-per-node=1
#SBATCH --partition=commons
#SBATCH --cpus-per-task=70
#SBATCH --mem=40G
#SBATCH --time=23:00:00
#SBATCH --exclude=bg2u24g1,bg3u16g1

# One Ray cluster, many instances.
# This calls src/parallel_rank_r_dir.py ONCE, which iterates the directory internally
# without restarting Ray.
#
# Args:
#   1: qv_dir (default "graphs")
#   2: results_dir (default "results_rank_r_dir")   (kept simple, no per-n subdirs here)
#   3: rank (default 2)
#   4: precision in {16,32,64} (default 64)
#   5: candidates_per_task (default 1000)
#   6: debug flag (0/1, default 0)
#
# Example:
#   sbatch experiments/multi_node_rank_r_dir.sh graphs/erdos_renyi/rank_2/p01/n20 results/erdos_renyi/rank_2/p01/n20 2 32 1000 0

set -euo pipefail

QV_DIR=${1:-graphs}
RESULTS_DIR=${2:-results_rank_r_dir}
R=${3:-2}
PRECISION=${4:-64}
CANDIDATES_PER_TASK=${5:-1000}
DEBUG_FLAG=${6:-0}

DEBUG_ARG=""
if [[ "$DEBUG_FLAG" -eq 1 ]]; then
  DEBUG_ARG="--debug"
fi

echo "Job started on $(hostname)"
echo "SLURM_JOB_NUM_NODES: ${SLURM_JOB_NUM_NODES:-}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-}"
echo "Using qv_dir = $QV_DIR"
echo "Using results_dir = $RESULTS_DIR"
echo "Using rank = $R"
echo "Using precision = $PRECISION"
echo "Using candidates_per_task = $CANDIDATES_PER_TASK"
echo "Debug enabled: $DEBUG_FLAG"

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

echo "Allocated nodes:"
for n in "${nodes_array[@]}"; do
  echo "  $n"
done

# BLAS threading: avoid oversubscription
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Clean env
module purge || true
unset LD_PRELOAD || true
export XALT_EXECUTABLE_TRACKING=0
export XALT_RUNNABLE=0

# Conda
cd /home/bs82/max-k-cut-parallel/ || exit 1
mkdir -p logs
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ./rayenv

QV_DIR=$(readlink -f "$QV_DIR")
RESULTS_DIR=$(readlink -m "$RESULTS_DIR")
mkdir -p "$RESULTS_DIR"

if [[ ! -d "$QV_DIR" ]]; then
  echo "ERROR: qv_dir does not exist: $QV_DIR" >&2
  exit 2
fi

# Ray cluster setup (for symmetric_run)
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
port=5050

head_ip=$(srun -N1 -n1 -w "$head_node" hostname -I | awk '{print $1}')
ip_head="${head_ip}:${port}"
export ip_head
echo "Ray head GCS address: $ip_head"

# Run ONCE via symmetric_run.
# parallel_rank_r_dir.py will:
# - ray.init(address="auto") once
# - iterate all Q_*_seed_*.npy / V_*_seed_*.npy pairs
# - write jsons into RESULTS_DIR
srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
  python -m read_only.symmetric_run \
    --address "$ip_head" \
    --min-nodes "$SLURM_JOB_NUM_NODES" \
    --num-cpus "$SLURM_CPUS_PER_TASK" \
    -- \
    python -u src/parallel_rank_r_dir.py \
      --qv_dir "$QV_DIR" \
      --results_dir "$RESULTS_DIR" \
      --rank "$R" \
      --precision "$PRECISION" \
      --candidates_per_task "$CANDIDATES_PER_TASK" \
      --skip_existing
      $DEBUG_ARG

echo "Job complete."