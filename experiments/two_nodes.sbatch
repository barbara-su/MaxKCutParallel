#!/bin/bash
#SBATCH --job-name=rank1-maxcut
#SBATCH --output=logs/rank1-maxcut-%j.out
#SBATCH --error=logs/rank1-maxcut-%j.err

#SBATCH --partition=commons
#SBATCH --nodes=2
#SBATCH --nodelist=bg6u16g1,bg6u24g1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=208
#SBATCH --mem=0
#SBATCH --time=24:00:00

set -euo pipefail

echo "Job started on $(hostname)"
echo "SLURM_NODELIST: $SLURM_NODELIST"

nodes=($(scontrol show hostnames "$SLURM_NODELIST"))
head_node=${nodes[0]}

echo "Nodes: ${nodes[@]}"
echo "Head node: $head_node"

# Single line, safe for bash -c
setup_env_cmd="cd /home/bs82/max-k-cut-parallel/; module purge || true; unset LD_PRELOAD || true; unset RAY_ADDRESS || true; source ./local_miniforge/etc/profile.d/conda.sh; conda activate /home/bs82/max-k-cut-parallel/ray311"

echo "Stopping Ray on both nodes"
for node in "${nodes[@]}"; do
  srun -N1 -n1 -w "$node" bash -c "$setup_env_cmd; ray stop || true"
done

echo "Starting Ray head on $head_node"
srun -N1 -n1 -w "$head_node" bash -c "$setup_env_cmd; ray start --head --port=5050 --num-cpus=208"
echo "Ray head started."

worker=${nodes[1]}
echo "Starting Ray worker on $worker"
srun -N1 -n1 -w "$worker" bash -c "$setup_env_cmd; ray start --address=$head_node:5050 --num-cpus=208"
echo "Ray worker started."

echo "Running Python code on $head_node"
srun -N1 -n1 -w "$head_node" bash -c "$setup_env_cmd; python src/parallel_rank1.py --n 100000"

echo "Stopping Ray on both nodes"
for node in "${nodes[@]}"; do
  srun -N1 -n1 -w "$node" bash -c "$setup_env_cmd; ray stop || true"
done

echo "Done."
