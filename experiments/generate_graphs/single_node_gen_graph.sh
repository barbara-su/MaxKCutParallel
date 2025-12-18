#!/bin/bash
#SBATCH --job-name=gen_graph
#SBATCH --output=logs/gen-graph-%j.out
#SBATCH --error=logs/gen-graph-%j.err

#SBATCH --partition=commons
#SBATCH --cpus-per-task=2
#SBATCH --mem=400G
#SBATCH --time=23:00:00

# Read n, rank, and output directory from arguments
# Default n to 20000, rank to 2, out dir to "graphs" if not provided
N=${1:-20000}
R=${2:-2}
OUT_DIR=${3:-graphs}

echo "Job started on $(hostname)"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Using n = $N"
echo "Using rank = $R"
echo "Using output directory = $OUT_DIR"

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

# Run python code
python src/graph_generators/gen_qv.py --n "$N" --rank "$R" --out_dir "$OUT_DIR"

echo "Job complete."
