#!/bin/bash
#SBATCH --job-name=rank-r-maxcut
#SBATCH --output=logs/rank-r-maxcut-%j.out
#SBATCH --error=logs/rank-r-maxcut-%j.err

#SBATCH --partition=commons
#SBATCH --cpus-per-task=208
#SBATCH --mem=0
#SBATCH --time=23:00:00

# Args:
#   1: n (default 20000)
#   2: rank (default 2)
#   3: graph_dir (default "graphs")
#   4: results_dir (default "results")
#   5: precision in {16,32,64} (default 64)
#   6: candidates_per_task (default 1000)
N=${1:-20000}
R=${2:-2}
GRAPH_DIR=${3:-graphs}
RESULTS_DIR=${4:-results}
PRECISION=${5:-64}
CANDIDATES_PER_TASK=${6:-1000}

echo "Using n = $N"
echo "Using rank = $R"
echo "Using graph_dir = $GRAPH_DIR"
echo "Using results_dir = $RESULTS_DIR"
echo "Using precision = $PRECISION"
echo "Using candidates_per_task = $CANDIDATES_PER_TASK"

# Clean env
module purge || true
unset LD_PRELOAD || true
export XALT_EXECUTABLE_TRACKING=0
export XALT_RUNNABLE=0

# openblas 
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Conda
cd /home/bs82/max-k-cut-parallel/
mkdir -p logs
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ./rayenv

# Start Ray
ray stop
ray start --head --port=5050

# Run python
python src/parallel_rank_r.py \
  --n "$N" \
  --rank "$R" \
  --graph_dir "$GRAPH_DIR" \
  --results_dir "$RESULTS_DIR" \
  --precision "$PRECISION" \
  --candidates_per_task "$CANDIDATES_PER_TASK"

echo "Job complete."
