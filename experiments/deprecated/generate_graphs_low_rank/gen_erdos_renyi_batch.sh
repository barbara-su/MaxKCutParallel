#!/bin/bash
#SBATCH --job-name=gen_er_qv_many
#SBATCH --output=logs/gen-er-qv-many-%j.out
#SBATCH --error=logs/gen-er-qv-many-%j.err

#SBATCH --partition=commons
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=23:00:00

# Args:
#   1: n (default 20000)
#   2: p (required in practice, but default provided)
#   3: rank (default 1)
#   4: num_seeds (default 20)
#   5: out_dir (default "graphs_er")
N=${1:-20000}
P=${2:-0.001}
R=${3:-1}
NUM_SEEDS=${4:-20}
OUT_DIR=${5:-graphs_er}

echo "Job started on $(hostname)"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "Using n = $N"
echo "Using p = $P"
echo "Using rank = $R"
echo "Using num_seeds = $NUM_SEEDS"
echo "Using out_dir = $OUT_DIR"

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

# Run python code (ER multi-seed generator script)
python src/graph_generators_low_rank/gen_erdos_renyi_batch.py \
  --n "$N" \
  --p "$P" \
  --rank "$R" \
  --num_seeds "$NUM_SEEDS" \
  --out_dir "$OUT_DIR"

echo "Job complete."

