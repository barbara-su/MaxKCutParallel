#!/bin/bash
#SBATCH --job-name=genQV-gaus
#SBATCH --output=logs/genQV-gaus-%j.out
#SBATCH --error=logs/genQV-gaus-%j.err

#SBATCH --partition=commons
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=01:00:00

# Args:
#   1: n (default 1000)
#   2: sigma (default 1.0)
#   3: seed (default 42)
#   4: rank (default 1)
#   5: out_dir (default graphs/graphs_random_gaus)
N=${1:-1000}
SIGMA=${2:-1.0}
SEED=${3:-42}
RANK=${4:-1}
OUTDIR=${5:-graphs/graphs_random_gaus}

echo "Job started on $(hostname)"
echo "n=$N sigma=$SIGMA seed=$SEED rank=$RANK out_dir=$OUTDIR"

module purge || true
unset LD_PRELOAD || true
export XALT_EXECUTABLE_TRACKING=0
export XALT_RUNNABLE=0

cd /home/bs82/max-k-cut-parallel/ || exit 1
mkdir -p logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate ./rayenv

# Limit BLAS threads so you don’t oversubscribe the 8 allocated cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

python src/tmpt_gen.py \
  --n "$N" \
  --sigma "$SIGMA" \
  --seed "$SEED" \
  --rank "$RANK" \
  --out_dir "$OUTDIR" \
  --dtype complex64

echo "Done."
