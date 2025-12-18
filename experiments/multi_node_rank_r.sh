#!/bin/bash
#SBATCH --job-name=multi-nodes-rank-r
#SBATCH --output=logs/multi-nodes-rank-r-%j.out
#SBATCH --error=logs/multi-nodes-rank-r-%j.err

#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --partition=commons
#SBATCH --cpus-per-task=208
#SBATCH --mem=0
#SBATCH --time=23:00:00

# Args (match single-node rank-r script):
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

echo "Job started on $(hostname)"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "Using n = $N"
echo "Using rank = $R"
echo "Using graph_dir = $GRAPH_DIR"
echo "Using results_dir = $RESULTS_DIR"
echo "Using precision = $PRECISION"
echo "Using candidates_per_task = $CANDIDATES_PER_TASK"

# BLAS threading: avoid oversubscription
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Clean env
module purge || true
unset LD_PRELOAD || true
export XALT_EXECUTABLE_TRACKING=0
export XALT_RUNNABLE=0

# Conda
cd /home/bs82/max-k-cut-parallel/ || exit 1
mkdir -p logs
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ./rayenv

# Ray cluster setup
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
port=5050

head_ip=$(srun -N1 -n1 -w "$head_node" hostname -I | awk '{print $1}')
ip_head="${head_ip}:${port}"
export ip_head
echo "Ray head GCS address: $ip_head"

# Keep symmetric_run
srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
  python -m read_only.symmetric_run \
    --address "$ip_head" \
    --min-nodes "$SLURM_JOB_NUM_NODES" \
    --num-cpus "$SLURM_CPUS_PER_TASK" \
    -- \
    python -u src/parallel_rank_r.py \
      --n "$N" \
      --rank "$R" \
      --graph_dir "$GRAPH_DIR" \
      --results_dir "$RESULTS_DIR" \
      --precision "$PRECISION" \
      --candidates_per_task "$CANDIDATES_PER_TASK"

echo "Job complete."
