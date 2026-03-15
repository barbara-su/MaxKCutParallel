#!/bin/bash
#SBATCH --job-name=gen_qv_many
#SBATCH --output=logs/gen-qv-many-%j.out
#SBATCH --error=logs/gen-qv-many-%j.err

#SBATCH --partition=commons
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=23:00:00

# Args:
#   1: n (default 20000)
#   2: d (default 3)
#   3: rank (default 1)
#   4: num_seeds (default 20)
#   5: out_dir (default "graphs")
N=${1:-20000}
D=${2:-3}
R=${3:-1}
NUM_SEEDS=${4:-20}
OUT_DIR=${5:-graphs}

echo "Job started on $(hostname)"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "Using n = $N"
echo "Using d = $D"
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
cd /scratch/bs82/max-k-cut-parallel/ || exit 1
mkdir -p logs
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ./rayenv

# Run python code (new multi-seed generator script)
python src/graph_generators/gen_regular_random_batch.py \
  --n "$N" \
  --d "$D" \
  --rank "$R" \
  --num_seeds "$NUM_SEEDS" \
  --out_dir "$OUT_DIR"

echo "Job complete."

# to run, call
# sbatch experiments/generate_graphs/gen_regular_graph_batch.sh 100 5 2 20 graphs/5_regular_graph_rank_2/n100

# sbatch experiments/generate_graphs/gen_regular_graph_batch.sh 20 3 3 20 graphs/3_regular_graph_rank_3/n20
# sbatch experiments/generate_graphs/gen_regular_graph_batch.sh 50 3 3 20 graphs/3_regular_graph_rank_3/n50
# sbatch experiments/generate_graphs/gen_regular_graph_batch.sh 100 3 3 20 graphs/3_regular_graph_rank_3/n100

# sbatch experiments/generate_graphs/gen_regular_graph_batch.sh 20 5 3 20 graphs/5_regular_graph_rank_3/n20
# sbatch experiments/generate_graphs/gen_regular_graph_batch.sh 50 5 3 20 graphs/5_regular_graph_rank_3/n50
# sbatch experiments/generate_graphs/gen_regular_graph_batch.sh 100 5 3 20 graphs/5_regular_graph_rank_3/n100
