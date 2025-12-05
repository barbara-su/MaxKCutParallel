### Alloate machine for test purpose
srun --pty --time=00:30:00 --gres=gpu:h100:1 --mem=80G $SHELL
lscpu | grep "^CPU(s):"

### set up 

module purge || true
unset LD_PRELOAD || true
export XALT_EXECUTABLE_TRACKING=0
export XALT_RUNNABLE=0
unset LD_LIBRARY_PATH
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ./rayenv

### start ray
on head: ray start --head --num-cpus=200 --include-dashboard=false
on worker: ray start --address='192.168.154.11:6379' --num-cpus=200

### ray test
python -c "import ray; ray.init(address='auto'); print(ray.cluster_resources())"

### check allocation status
sinfo -N -o "%N %c %m %e %C" \
    | awk 'NR==1{print; next} {print $1, $2, $3/1024, $4/1024, $5}'

### submit job
sbatch experiments/single_node_rank_r.sh 100 2


