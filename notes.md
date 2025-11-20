1. Allocate 2 h100 machines: srun --pty --time=00:30:00 --gres=gpu:h100:1 --mem=80G $SHELL
2. Check number of CPUs: lscpu | grep "^CPU(s):"
3. Load conda: 
    unset LD_PRELOAD
    source ./local_miniforge/etc/profile.d/conda.sh
    conda activate /home/bs82/max-k-cut-parallel/ray311
4. Start ray: ray start --head --port=6379
5. Connect to an open sbatch terminal: squeue -j 12209 -o "%N", ssh it
6. Run sbatch: sbatch test.sbatch

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


