#!/bin/bash
#SBATCH --job-name=rank-r-maxcut
#SBATCH --output=logs/rank-r-maxcut-%j.out
#SBATCH --error=logs/rank-r-maxcut-%j.err

#SBATCH --partition=commons
#SBATCH --cpus-per-task=100
#SBATCH --mem=500G
#SBATCH --time=23:00:00

# Read n, rank, and graph_dir from arguments
# Default n to 20000, rank to 2, graph_dir to "graphs" if not provided
N=${1:-20000}
R=${2:-2}
GRAPH_DIR=${3:-graphs}
RECOVERY_FLAG=${4:-0}

echo "Using n = $N"
echo "Using rank = $R"
echo "Using graph_dir = $GRAPH_DIR"
echo "Compute recovery: $RECOVERY_FLAG"

# Clean env
module purge || true
unset LD_PRELOAD || true
export XALT_EXECUTABLE_TRACKING=0
export XALT_RUNNABLE=0

# Conda
cd /home/bs82/max-k-cut-parallel/
mkdir -p logs
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ./rayenv

# Start Ray
ray stop
ray start --head --port=5050

# Run python
python src/parallel_rank_r.py --n "$N" --rank "$R" --debug

echo "Job complete."