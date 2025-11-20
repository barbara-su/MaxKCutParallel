#!/bin/bash
#SBATCH --job-name=ray-slurm-demo
#SBATCH --output=logs/ray-slurm-demo-%j.out
#SBATCH --error=logs/ray-slurm-demo-%j.err
#SBATCH --partition=commons
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=208
#SBATCH --mem=0
#SBATCH --time=01:00:00

set -euo pipefail

cd /home/bs82/max-k-cut-parallel/
unset LD_PRELOAD
source ./local_miniforge/etc/profile.d/conda.sh
conda activate /home/bs82/max-k-cut-parallel/ray311

nodes=($(scontrol show hostnames "$SLURM_NODELIST"))
head_node=${nodes[0]}
head_ip=$(srun -N1 -n1 -w "$head_node" hostname -I | awk '{print $1}')

# Stop lingering Ray processes
for node in "${nodes[@]}"; do
    srun -N1 -n1 -w "$node" ray stop || true
done

# Start Ray head node
srun -N1 -n1 -w "$head_node" ray start --head --node-ip-address="$head_ip" --port=6379 --num-cpus=$SLURM_CPUS_PER_TASK --block &

sleep 10

# Start Ray worker nodes
for node in "${nodes[@]:1}"; do
    worker_ip=$(srun -N1 -n1 -w "$node" hostname -I | awk '{print $1}')
    srun -N1 -n1 -w "$node" ray start --address="$head_ip:6379" --node-ip-address="$worker_ip" --num-cpus=$SLURM_CPUS_PER_TASK --block &
    sleep 5
done

sleep 10

# Run your Python script on the head node
srun -N1 -n1 -w "$head_node" python -u src/test_multi_nodes.py

# Stop Ray processes
for node in "${nodes[@]}"; do
    srun -N1 -n1 -w "$node" ray stop || true
done