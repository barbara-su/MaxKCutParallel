#!/bin/bash
#SBATCH --job-name=multi-nodes-test
#SBATCH --output=logs/multi-nodes-test-%j.out
#SBATCH --error=logs/multi-nodes-test-%j.err
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1

#SBATCH --partition=commons
#SBATCH --cpus-per-task=208
#SBATCH --mem=0
#SBATCH --time=23:00:00

# Read argument
N=${1:-20000}

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

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
port=6379

# Ask Slurm to run hostname -I on the head node and grab the first IP
head_ip=$(srun -N1 -n1 -w "$head_node" hostname -I | awk '{print $1}')

ip_head="${head_ip}:${port}"
export ip_head
echo "Ray head GCS address: $ip_head"

srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
    python -m read_only.symmetric_run \
    --address "$ip_head" \
    --min-nodes "$SLURM_JOB_NUM_NODES" \
    --num-cpus="${SLURM_CPUS_PER_TASK}" \
    -- \
    python -u src/parallel_rank1.py --n "$N" --graph_dir "graphs"