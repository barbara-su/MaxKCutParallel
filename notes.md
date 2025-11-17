1. Allocate 2 h100 machines: srun --pty --time=00:30:00 --gres=gpu:h100:1 --mem=80G $SHELL
2. Check number of CPUs: lscpu | grep "^CPU(s):"
3. Load conda: 
    unset LD_PRELOAD
    source ./local_miniforge/etc/profile.d/conda.sh
    conda activate /home/bs82/max-k-cut-parallel/ray311
4. Start ray: ray start --head --port=6379
5. Connect to an open sbatch terminal: squeue -j 12209 -o "%N", ssh it
6. Run sbatch: sbatch test.sbatch


