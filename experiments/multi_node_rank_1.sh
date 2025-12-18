#!/bin/bash
#SBATCH --job-name=multi-nodes-rank-1
#SBATCH --output=logs/multi-nodes-rank-1-%j.out
#SBATCH --error=logs/multi-nodes-rank-1-%j.err

#SBATCH --nodes=4
#SBATCH --tasks-per-node=1
#SBATCH --partition=commons
#SBATCH --cpus-per-task=102
#SBATCH --mem=350G
#SBATCH --time=23:00:00

# Args (match newest single_node_rank.sh):
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
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "Using n = $N"
echo "Using results_dir = $RESULTS_DIR"
echo "Using precision = $PRECISION"
echo "Using candidates_per_task = $CANDIDATES_PER_TASK"
echo "Using seed = $SEED"

# BLAS threading: avoid oversubscription (Ray parallelism provides concurrency)
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
cd /home/bs82/max-k-cut-parallel/ || exit 1
mkdir -p logs
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ./rayenv

# Ray cluster setup info
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
port=5050

# Get head node IP
head_ip=$(srun -N1 -n1 -w "$head_node" hostname -I | awk '{print $1}')
ip_head="${head_ip}:${port}"
export ip_head
echo "Ray head GCS address: $ip_head"

# Run via symmetric_run (keep this)
srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
  python -m read_only.symmetric_run \
    --address "$ip_head" \
    --min-nodes "$SLURM_JOB_NUM_NODES" \
    --num-cpus "$SLURM_CPUS_PER_TASK" \
    -- \
    python -u src/parallel_rank_1.py \
      --n "$N" \
      --seed "$SEED" \
      --graph_dir "graphs/graphs_rank_1" \
      --results_dir "$RESULTS_DIR" \
      --precision "$PRECISION" \
      --candidates_per_task "$CANDIDATES_PER_TASK"

echo "Job complete."
