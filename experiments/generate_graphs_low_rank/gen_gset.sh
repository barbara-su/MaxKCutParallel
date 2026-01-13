#!/bin/bash
#SBATCH --job-name=gset_one
#SBATCH --output=logs/gset-one-%j.out
#SBATCH --error=logs/gset-one-%j.err

#SBATCH --partition=commons
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=23:00:00

# Args:
#   1: gset_id (required)   e.g. 1, 22, 80
#   2: out_dir (default: graphs_gset_one)
#   3: rank (default: 1)
#   4: seed (default: 42)
#   5: random_weights (default: 0)  (0/1)
#   6: random_low (default: 0.0)
#   7: random_high (default: 1.0)
#   8: in_dir (default: ../graphs/gset)

GSET_ID=${1:?Usage: sbatch gset_one.sh <gset_id> [out_dir] [rank] [seed] [random_weights] [random_low] [random_high] [in_dir]}
OUT_DIR=${2:-graphs_gset_one}
RANK=${3:-1}
SEED=${4:-42}
RANDOM_WEIGHTS=${5:-0}
RANDOM_LOW=${6:-0.0}
RANDOM_HIGH=${7:-1.0}
IN_DIR=${8:-../graphs/gset}

echo "Job started on $(hostname)"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "Using gset_id = $GSET_ID"
echo "Using out_dir = $OUT_DIR"
echo "Using rank = $RANK"
echo "Using seed = $SEED"
echo "Using random_weights = $RANDOM_WEIGHTS"
echo "Using random_low = $RANDOM_LOW"
echo "Using random_high = $RANDOM_HIGH"
echo "Using in_dir = $IN_DIR"

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

# Run python
python -u src/graph_generators_low_rank/gen_from_gset.py \
  --gset "$GSET_ID" \
  --rank "$RANK" \
  --seed "$SEED" \
  --random_weights "$RANDOM_WEIGHTS" \
  --random_low "$RANDOM_LOW" \
  --random_high "$RANDOM_HIGH" \
  --in_dir "$IN_DIR" \
  --out_dir "$OUT_DIR"

echo "Job complete."

# GSET_ID=${1:?Usage: sbatch gen_gset.sh <gset_id> [out_dir] [rank] [seed] [random_weights] [random_low] [random_high] [in_dir]}
# OUT_DIR=${2:-graphs_gset_one}
# RANK=${3:-1}
# SEED=${4:-42}
# RANDOM_WEIGHTS=${5:-0}
# RANDOM_LOW=${6:-0.0}
# RANDOM_HIGH=${7:-1.0}
# IN_DIR=${8:-../graphs/gset}

# for G70, G72, G77, G81, weighted experiments with:
# random_low=0.9, random_high=1.1
# random_low=0, random_high=10
# random_low=0, random_high=100

# sbatch experiments/generate_graphs/gen_gset.sh 70 /scratch/bs82/graphs/gset_random/1_1 1 42 true 0.9 1.1 gset
# sbatch experiments/generate_graphs/gen_gset.sh 70 /scratch/bs82/graphs/gset_random/10 1 42 true 0 10 gset
# sbatch experiments/generate_graphs/gen_gset.sh 70 /scratch/bs82/graphs/gset_random/100 1 42 true 0 100 gset

# sbatch experiments/generate_graphs/gen_gset.sh 72 /scratch/bs82/graphs/gset_random/1_1 1 42 true 0.9 1.1 gset
# sbatch experiments/generate_graphs/gen_gset.sh 72 /scratch/bs82/graphs/gset_random/10 1 42 true 0 10 gset
# sbatch experiments/generate_graphs/gen_gset.sh 72 /scratch/bs82/graphs/gset_random/100 1 42 true 0 100 gset

# sbatch experiments/generate_graphs/gen_gset.sh 77 /scratch/bs82/graphs/gset_random/1_1 1 42 true 0.9 1.1 gset
# sbatch experiments/generate_graphs/gen_gset.sh 77 /scratch/bs82/graphs/gset_random/10 1 42 true 0 10 gset
# sbatch experiments/generate_graphs/gen_gset.sh 77 /scratch/bs82/graphs/gset_random/100 1 42 true 0 100 gset

# sbatch experiments/generate_graphs/gen_gset.sh 81 /scratch/bs82/graphs/gset_random/1_1 1 42 true 0.9 1.1 gset
# sbatch experiments/generate_graphs/gen_gset.sh 81 /scratch/bs82/graphs/gset_random/10 1 42 true 0 10 gset
# sbatch experiments/generate_graphs/gen_gset.sh 81 /scratch/bs82/graphs/gset_random/100 1 42 true 0 100 gset

# sbatch experiments/generate_graphs/gen_gset.sh 35 /scratch/bs82/graphs/ANYCSP_missing_gset 1 42 false 0 0 gset
