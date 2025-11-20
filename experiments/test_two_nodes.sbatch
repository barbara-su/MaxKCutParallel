#!/bin/bash
#SBATCH --job-name=test-two-nodes
#SBATCH --output=logs/test-two-nodes-%j.out
#SBATCH --error=logs/test-two-nodes-%j.err

#SBATCH --partition=commons
#SBATCH --nodes=2
#SBATCH --nodelist=bg6u16g1,bg6u24g1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=208
#SBATCH --mem=0                  # use all memory on each node
#SBATCH --time=24:00:00

set -euo pipefail

mkdir -p logs

echo "Job started on $(hostname)"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"

# Resolve hostnames for the nodes in this allocation
nodes=($(scontrol show hostnames "$SLURM_NODELIST"))
head_node=${nodes[0]}

echo "Nodes: ${nodes[@]}"
echo "Head node: $head_node"

# Get the primary IP address of the head node
head_ip=$(srun -N1 -n1 -w "$head_node" bash -c "hostname -I | awk '{print \$1}'")
echo "Head node IP: $head_ip"

# Single line environment setup
setup_env_cmd="cd /home/bs82/max-k-cut-parallel/; module purge || true; unset LD_PRELOAD || true; unset RAY_ADDRESS || true; source ./local_miniforge/etc/profile.d/conda.sh; conda activate /home/bs82/max-k-cut-parallel/ray311"

echo "Stopping Ray on all nodes"
for node in "${nodes[@]}"; do
  echo "Stopping Ray on $node"
  srun -N1 -n1 -w "$node" bash -c "$setup_env_cmd; ray stop || true"
done

echo "Starting Ray head on $head_node"
srun -N1 -n1 -w "$head_node" bash -c "
  $setup_env_cmd;
  ray start --head --port=5050 --num-cpus=$SLURM_CPUS_PER_TASK
"
echo "Ray head started."

echo "Starting Ray worker nodes"
for node in "${nodes[@]:1}"; do
  echo "Starting worker on $node"
  srun -N1 -n1 -w "$node" bash -c "
    $setup_env_cmd;
    ray start --address=${head_ip}:5050 --num-cpus=$SLURM_CPUS_PER_TASK
  "
done

echo "Running Python test on $head_node"
srun -N1 -n1 -w "$head_node" bash -c "
  $setup_env_cmd;
  python src/test_multi_nodes.py
"

echo "Stopping Ray on all nodes"
for node in "${nodes[@]}"; do
  echo "Stopping Ray on $node"
  srun -N1 -n1 -w "$node" bash -c "$setup_env_cmd; ray stop || true"
done

echo "Done."
