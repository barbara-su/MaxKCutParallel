#!/bin/bash
#SBATCH --job-name=gen_gset_qv
#SBATCH --output=logs/gen-gset-qv-%j.out
#SBATCH --error=logs/gen-gset-qv-%j.err

#SBATCH --partition=commons
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=23:00:00

# Args:
#   1: rank (default 1)
#   2: seed (default 42)
#   3: random_weights (default false; accepts 0/1/true/false)
#   4: random_low (default 0.0)
#   5: random_high (default 1.0)
#   6: in_dir (default ../graphs/gset)
#   7: out_dir (default graphs_gset)
#   8: max_id (default 81)
#   9: skip_existing (default true; accepts 0/1/true/false)

R=${1:-1}
SEED=${2:-42}
RANDOM_WEIGHTS=${3:-false}
RANDOM_LOW=${4:-0.0}
RANDOM_HIGH=${5:-1.0}
IN_DIR=${6:-../graphs/gset}
OUT_DIR=${7:-graphs_gset}
MAX_ID=${8:-81}
SKIP_EXISTING=${9:-true}

echo "Job started on $(hostname)"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "Using rank = $R"
echo "Using seed = $SEED"
echo "Using random_weights = $RANDOM_WEIGHTS"
echo "Using random_low = $RANDOM_LOW"
echo "Using random_high = $RANDOM_HIGH"
echo "Using in_dir = $IN_DIR"
echo "Using out_dir = $OUT_DIR"
echo "Using max_id = $MAX_ID"
echo "Using skip_existing = $SKIP_EXISTING"

# Threading: this job is single-process, heavy linear algebra
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
mkdir -p logs
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ./rayenv

# Run python code (adjust path to your script if needed)
python -u src/graph_generators/gen_grom_gset_batch.py \
  --rank "$R" \
  --seed "$SEED" \
  --random_weights "$RANDOM_WEIGHTS" \
  --random_low "$RANDOM_LOW" \
  --random_high "$RANDOM_HIGH" \
  --in_dir "$IN_DIR" \
  --out_dir "$OUT_DIR" \
  --max_id "$MAX_ID" \
  --skip_existing "$SKIP_EXISTING"

echo "Job complete."

# R=${1:-1}
# SEED=${2:-42}
# RANDOM_WEIGHTS=${3:-false}
# RANDOM_LOW=${4:-0.0}
# RANDOM_HIGH=${5:-1.0}
# IN_DIR=${6:-../graphs/gset}
# OUT_DIR=${7:-graphs_gset}
# MAX_ID=${8:-81}
# SKIP_EXISTING=${9:-true}

# to call it
# sbatch experiments/generate_graphs/gen_gset_batch.sh 1 42 false 0 0 gset /scratch/bs82/graphs/gset 81 true
# sbatch experiments/generate_graphs/gen_gset_batch.sh 2 42 false 0 0 gset /scratch/graphs/gset_rank_2 81 true