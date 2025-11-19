#!/bin/bash
#SBATCH --job-name=rank1-maxcut
#SBATCH --output=logs/rank1-maxcut-%j.out
#SBATCH --error=logs/rank1-maxcut-%j.err

#SBATCH --partition=commons
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=208        # one Ray process per node, using all 208 cores
#SBATCH --mem=1024GB               # per node
#SBATCH --time=24:00:00

echo "Job started on $(hostname)"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"

# Clean environment
module purge || true
unset LD_PRELOAD || true
export XALT_EXECUTABLE_TRACKING=0
export XALT_RUNNABLE=0

# Project environment
cd /home/bs82/max-k-cut-parallel/
mkdir -p logs
source ./local_miniforge/etc/profile.d/conda.sh
conda activate /home/bs82/max-k-cut-parallel/ray311

# Get node list
nodes=($(scontrol show hostnames "$SLURM_NODELIST"))
NUM_NODES=${SLURM_NNODES:-${#nodes[@]}}

head_node=${nodes[0]}

echo "Nodes in allocation: ${nodes[@]}"
echo "Using NUM_NODES = $NUM_NODES"
echo "Head node: $head_node"

# Stop any existing Ray on the head node
srun -N1 -n1 -w "$head_node" ray stop || true

# Start Ray head
echo "Starting Ray head on $head_node"
srun -N1 -n1 -w "$head_node" \
  ray start --head \
    --port=5050 \
    --num-cpus="$SLURM_CPUS_PER_TASK" &

sleep 10

# Start Ray workers on remaining nodes
if [ "$NUM_NODES" -gt 1 ]; then
  for ((i=1; i<NUM_NODES; i++)); do
    worker_node=${nodes[$i]}
    echo "Starting Ray worker on $worker_node"
    srun -N1 -n1 -w "$worker_node" \
      ray start \
        --address="$head_node:5050" \
        --num-cpus="$SLURM_CPUS_PER_TASK" &
  done
fi

# Wait for all ray start commands
wait

echo "Ray cluster should be up. Running Python job on head node."

# Run Python code on head node for n = 100000
srun -N1 -n1 -w "$head_node" \
  python src/parallel_rank1.py --n 100000

echo "Python job complete. Stopping Ray cluster."

# Cleanly stop Ray on all nodes
srun -N1 -n1 -w "$head_node" ray stop || true
if [ "$NUM_NODES" -gt 1 ]; then
  for ((i=1; i<NUM_NODES; i++)); do
    worker_node=${nodes[$i]}
    srun -N1 -n1 -w "$worker_node" ray stop || true
  done
fi

echo "Job complete."
