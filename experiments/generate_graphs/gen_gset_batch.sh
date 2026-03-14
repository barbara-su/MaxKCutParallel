#!/bin/bash
#SBATCH --job-name=gen_gset_qv
#SBATCH --output=logs/gen-gset-qv-%j.out
#SBATCH --error=logs/gen-gset-qv-%j.err

#SBATCH --partition=commons
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=23:00:00

# Args:
#   1: rank
#   2: seed
#   3: random_weights
#   4: random_low
#   5: random_high
#   6: in_dir
#   7: out_dir
#   8: max_id
#   9: skip_existing
#  10: result_dir

R=${1:-1}
SEED=${2:-42}
RANDOM_WEIGHTS=${3:-false}
RANDOM_LOW=${4:-0.0}
RANDOM_HIGH=${5:-1.0}
IN_DIR=${6:-../graphs/gset}
OUT_DIR=${7:-graphs_gset}
MAX_ID=${8:-81}
SKIP_EXISTING=${9:-true}
RESULT_DIR=${10:-results_json}

echo "Job started on $(hostname)"
echo "Rank           : $R"
echo "Seed           : $SEED"
echo "Random weights : $RANDOM_WEIGHTS"
echo "In dir         : $IN_DIR"
echo "Out dir        : $OUT_DIR"
echo "Result dir     : $RESULT_DIR"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=1

module purge || true
unset LD_PRELOAD || true

cd /home/bs82/max-k-cut-parallel/ || exit 1
mkdir -p logs
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ./rayenv

python -u src/graph_generators/gen_from_gset_batch.py \
  --rank "$R" \
  --seed "$SEED" \
  --random_weights "$RANDOM_WEIGHTS" \
  --random_low "$RANDOM_LOW" \
  --random_high "$RANDOM_HIGH" \
  --in_dir "$IN_DIR" \
  --out_dir "$OUT_DIR" \
  --result_dir "$RESULT_DIR" \
  --max_id "$MAX_ID" \
  --skip_existing "$SKIP_EXISTING"

echo "Job complete."

# to call it
# sbatch experiments/generate_graphs/gen_gset_batch.sh 1 42 false 0 0 gset /scratch/bs82/graphs/gset 81 false results/gset_rank_1_timing 
# sbatch experiments/generate_graphs/gen_gset_batch.sh 2 42 false 0 0 gset /scratch/bs82/graphs/gset_rank_2 81 true