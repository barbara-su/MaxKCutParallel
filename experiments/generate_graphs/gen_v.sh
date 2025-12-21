#!/bin/bash
#SBATCH --job-name=gen_v
#SBATCH --output=logs/gen-v-%j.out
#SBATCH --error=logs/gen-v-%j.err

#SBATCH --partition=commons
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=23:00:00

# Args:
#   1: q_path (required)
#   2: v_path (required)
#   3: rank (default 1)
Q_PATH=${1:?Usage: sbatch gen_v.sh <q_path> <v_path> [rank]}
V_PATH=${2:?Usage: sbatch gen_v.sh <q_path> <v_path> [rank]}
R=${3:-1}

echo "Job started on $(hostname)"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "Using q_path = $Q_PATH"
echo "Using v_path = $V_PATH"
echo "Using rank = $R"

# openblas
# Use all allocated CPUs for the linear algebra
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=1

# Clean environment
module purge || true
unset LD_PRELOAD || true
export XALT_EXECUTABLE_TRACKING=0
export XALT_RUNNABLE=0

# Conda
cd /home/bs82/max-k-cut-parallel/ || exit 1
mkdir -p logs
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ./rayenv

# Run python code
python src/graph_generators/gen_v.py \
  --q_path "$Q_PATH" \
  --v_path "$V_PATH" \
  --rank "$R"

echo "Job complete."
