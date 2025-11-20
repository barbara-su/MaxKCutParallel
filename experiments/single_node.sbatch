#!/bin/bash
#SBATCH --job-name=rank1-maxcut
#SBATCH --output=logs/rank1-maxcut-%j.out
#SBATCH --error=logs/rank1-maxcut-%j.err

#SBATCH --partition=commons
#SBATCH --cpus-per-task=208
#SBATCH --mem=0          
#SBATCH --time=23:00:00

echo "Job started on $(hostname)"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"


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
python src/parallel_rank1.py --n 100000

echo "Job complete."
