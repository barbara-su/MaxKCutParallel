#!/bin/bash
#SBATCH --job-name=gen_graph
#SBATCH --output=logs/gen-graph-%j.out
#SBATCH --error=logs/gen-graph-%j.err

#SBATCH --partition=commons
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=23:00:00

# Read n from first argument, default to 20000 if not provided
N=${1:-20000}

echo "Job started on $(hostname)"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Using n = $N"

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
python src/gen_qv.py --n="$N"

echo "Job complete."
