#!/bin/bash
#SBATCH --job-name=rank1-maxcut
#SBATCH --output=logs/rank-1-maxcut-%j.out
#SBATCH --error=logs/rank-1-maxcut-%j.err

#SBATCH --partition=commons
#SBATCH --cpus-per-task=208
#SBATCH --mem=0
#SBATCH --time=23:00:00

# Args:
#   1: n (default 20000)
#   2: results_dir (default "results")
#   3: precision in {16,32,64} (default 64)
#   4: candidates_per_task (default 10)
#   5: seed (default 42)
N=${1:-20000}
RESULTS_DIR=${2:-results}
PRECISION=${3:-64}
CANDIDATES_PER_TASK=${4:-10}
SEED=${5:-42}

echo "Job started on $(hostname)"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Using n = $N"
echo "Using results_dir = $RESULTS_DIR"
echo "Using precision = $PRECISION"
echo "Using candidates_per_task = $CANDIDATES_PER_TASK"
echo "Using seed = $SEED"

# openblas 
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Clean environment
module purge || true
unset LD_PRELOAD || true
export XALT_EXECUTABLE_TRACKING=0
export XALT_RUNNABLE=0

# Conda
cd /home/bs82/max-k-cut-parallel/
mkdir -p logs
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ./rayenv

# start ray
ray stop
ray start --head --port=5050

# Run python code
python src/parallel_rank_1.py \
  --n "$N" \
  --seed "$SEED" \
  --graph_dir "graphs/graphs_rank_1" \
  --results_dir "$RESULTS_DIR" \
  --precision "$PRECISION" \
  --candidates_per_task "$CANDIDATES_PER_TASK"

echo "Job complete."
