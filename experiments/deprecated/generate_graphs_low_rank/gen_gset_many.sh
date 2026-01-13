#!/bin/bash
#SBATCH --job-name=gset_many
#SBATCH --output=logs/gset-many-%j.out
#SBATCH --error=logs/gset-many-%j.err

#SBATCH --partition=commons
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=23:00:00

# Args:
#   1: gset_id (required)   e.g. 1, 22, 80
#   2: out_dir (default: graphs_gset_many)
#   3: rank (default: 1)
#   4: seed_start (default: 0)
#   5: count (default: 20)
#   6: random_weights (default: 0) (0/1)
#   7: random_low (default: 0.0)
#   8: random_high (default: 1.0)
#   9: in_dir (default: ../graphs/gset)
#  10: save_v (default: 1) (0/1)
#
# Notes:
# - This script calls: src/graph_generators_low_rank/gen_from_gset_many.py
# - Outputs:
#     Q_gset_{gset}_seed_{seed}.npy
#   and if save_v=1:
#     V_gset_{gset}_seed_{seed}.npy

GSET_ID=${1:?Usage: sbatch gen_gset_many.sh <gset_id> [out_dir] [rank] [seed_start] [count] [random_weights] [random_low] [random_high] [in_dir] [save_v]}
OUT_DIR=${2:-graphs_gset_many}
RANK=${3:-1}
SEED_START=${4:-0}
COUNT=${5:-20}
RANDOM_WEIGHTS=${6:-0}
RANDOM_LOW=${7:-0.0}
RANDOM_HIGH=${8:-1.0}
IN_DIR=${9:-../graphs/gset}
SAVE_V=${10:-1}

echo "Job started on $(hostname)"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "Using gset_id = $GSET_ID"
echo "Using out_dir = $OUT_DIR"
echo "Using rank = $RANK"
echo "Using seed_start = $SEED_START"
echo "Using count = $COUNT"
echo "Using random_weights = $RANDOM_WEIGHTS"
echo "Using random_low = $RANDOM_LOW"
echo "Using random_high = $RANDOM_HIGH"
echo "Using in_dir = $IN_DIR"
echo "Using save_v = $SAVE_V"

mkdir -p logs

# Threading: use your allocated CPUs for BLAS/LAPACK during eigh
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

# Build python flags for booleans.
# Your gen_from_gset_many.py uses flags: --random_weights, --save_v
RW_FLAG=""
if [[ "$RANDOM_WEIGHTS" == "1" || "$RANDOM_WEIGHTS" == "true" || "$RANDOM_WEIGHTS" == "True" ]]; then
  RW_FLAG="--random_weights"
fi

SV_FLAG=""
if [[ "$SAVE_V" == "1" || "$SAVE_V" == "true" || "$SAVE_V" == "True" ]]; then
  SV_FLAG="--save_v"
fi

python -u src/graph_generators_low_rank/gen_from_gset_many.py \
  --gset "$GSET_ID" \
  --out_dir "$OUT_DIR" \
  --in_dir "$IN_DIR" \
  --rank "$RANK" \
  --seed_start "$SEED_START" \
  --count "$COUNT" \
  $RW_FLAG \
  --random_low "$RANDOM_LOW" \
  --random_high "$RANDOM_HIGH" \
  $SV_FLAG

echo "Job complete."


# GSET_ID=${1:?Usage: sbatch gen_gset_many.sh <gset_id> [out_dir] [rank] [seed_start] [count] [random_weights] [random_low] [random_high] [in_dir] [save_v]}
# OUT_DIR=${2:-graphs_gset_many}
# RANK=${3:-1}
# SEED_START=${4:-0}
# COUNT=${5:-20}
# RANDOM_WEIGHTS=${6:-0}
# RANDOM_LOW=${7:-0.0}
# RANDOM_HIGH=${8:-1.0}
# IN_DIR=${9:-../graphs/gset}
# SAVE_V=${10:-1}

# sbatch experiments/generate_graphs/gen_gset_many.sh 70 /scratch/bs82/graphs/gset_random_many/70/1_1 1 0 20 1 0.9 1.1 gset 1
# sbatch experiments/generate_graphs/gen_gset_many.sh 70 /scratch/bs82/graphs/gset_random_many/70/10  1 40 20 1 0   10  gset 1
# sbatch experiments/generate_graphs/gen_gset_many.sh 70 /scratch/bs82/graphs/gset_random_many/70/100 1 80 20 1 0   100 gset 1

# sbatch experiments/generate_graphs/gen_gset_many.sh 72 /scratch/bs82/graphs/gset_random_many/72/1_1 1 0 20 1 0.9 1.1 gset 1
# sbatch experiments/generate_graphs/gen_gset_many.sh 72 /scratch/bs82/graphs/gset_random_many/72/10  1 40 20 1 0   10  gset 1
# sbatch experiments/generate_graphs/gen_gset_many.sh 72 /scratch/bs82/graphs/gset_random_many/72/100 1 80 20 1 0   100 gset 1

# sbatch experiments/generate_graphs/gen_gset_many.sh 77 /scratch/bs82/graphs/gset_random_many/77/1_1 1 0 20 1 0.9 1.1 gset 1
# sbatch experiments/generate_graphs/gen_gset_many.sh 77 /scratch/bs82/graphs/gset_random_many/77/10  1 40 20 1 0   10  gset 1
# sbatch experiments/generate_graphs/gen_gset_many.sh 77 /scratch/bs82/graphs/gset_random_many/77/100 1 80 20 1 0   100 gset 1

# sbatch experiments/generate_graphs/gen_gset_many.sh 81 /scratch/bs82/graphs/gset_random_many/81/1_1 1 0 20 1 0.9 1.1 gset 1
# sbatch experiments/generate_graphs/gen_gset_many.sh 81 /scratch/bs82/graphs/gset_random_many/81/10  1 40 20 1 0   10  gset 1
# sbatch experiments/generate_graphs/gen_gset_many.sh 81 /scratch/bs82/graphs/gset_random_many/81/100 1 80 20 1 0   100 gset 1