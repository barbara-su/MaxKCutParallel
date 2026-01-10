#!/bin/bash
#SBATCH --job-name=sbm_gen
#SBATCH --output=logs/sbm-gen-%j.out
#SBATCH --error=logs/sbm-gen-%j.err

#SBATCH --partition=commons
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=23:00:00

# Args:
#   1: n (default: 1000)
#   2: blocks (default: 2)
#   3: block_sizes (space-separated, default: "500 500")
#   4: prob_within (default: 0.5)
#   5: prob_between (default: 0.01)
#   6: rank (default: 1)
#   7: seed (default: 42)
#   8: out_dir (default: graphs_sbm)

N=${1:-1000}
BLOCKS=${2:-2}
BLOCK_SIZES=${3:-"500 500"}
PROB_WITHIN=${4:-0.5}
PROB_BETWEEN=${5:-0.01}
RANK=${6:-1}
SEED=${7:-42}
OUT_DIR=${8:-graphs_sbm}

echo "Job started on $(hostname)"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "n = $N"
echo "blocks = $BLOCKS"
echo "block_sizes = $BLOCK_SIZES"
echo "prob_within = $PROB_WITHIN"
echo "prob_between = $PROB_BETWEEN"
echo "rank = $RANK"
echo "seed = $SEED"
echo "out_dir = $OUT_DIR"

mkdir -p logs

# Threading for Laplacian + eigendecomposition
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
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
python -u src/graph_generators/gen_sbm.py \
  --n "$N" \
  --blocks "$BLOCKS" \
  --block_sizes $BLOCK_SIZES \
  --prob_within "$PROB_WITHIN" \
  --prob_between "$PROB_BETWEEN" \
  --rank "$RANK" \
  --seed "$SEED" \
  --out_dir "$OUT_DIR"

echo "Job complete."

# sbatch gen_sbm.sh 1000 2 "500 500" 0.5 0.01 1 42 /scratch/bs82/graphs/sbm
