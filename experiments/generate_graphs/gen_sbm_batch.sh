#!/bin/bash
#SBATCH --job-name=gen_sbm_qv_many
#SBATCH --output=logs/gen-sbm-qv-many-%j.out
#SBATCH --error=logs/gen-sbm-qv-many-%j.err

#SBATCH --partition=commons
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=23:00:00

# Args:
#   1: n (default 1000)
#   2: blocks (default 2)
#   3: block_sizes (space-separated, default "500 500")
#   4: prob_within (default 0.5)
#   5: prob_between (default 0.01)
#   6: rank (default 1)
#   7: num_seeds (default 20)
#   8: out_dir (default graphs_sbm)

N=${1:-1000}
BLOCKS=${2:-2}
BLOCK_SIZES=${3:-"500 500"}
P_IN=${4:-0.5}
P_OUT=${5:-0.01}
R=${6:-1}
NUM_SEEDS=${7:-20}
OUT_DIR=${8:-graphs_sbm}

echo "Job started on $(hostname)"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "n = $N"
echo "blocks = $BLOCKS"
echo "block_sizes = $BLOCK_SIZES"
echo "prob_within = $P_IN"
echo "prob_between = $P_OUT"
echo "rank = $R"
echo "num_seeds = $NUM_SEEDS"
echo "out_dir = $OUT_DIR"

mkdir -p logs

# Linear algebra threading
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
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ./rayenv

# Run
python src/graph_generators/gen_sbm_batch.py \
  --n "$N" \
  --blocks "$BLOCKS" \
  --block_sizes $BLOCK_SIZES \
  --prob_within "$P_IN" \
  --prob_between "$P_OUT" \
  --rank "$R" \
  --num_seeds "$NUM_SEEDS" \
  --out_dir "$OUT_DIR"

echo "Job complete."

# Example
# sbatch experiments/generate_graphs/gen_sbm_batch.sh 1000 2 "500 500" 0.5 0.01 1 20 /scratch/bs82/graphs/graphs_sbm/example
