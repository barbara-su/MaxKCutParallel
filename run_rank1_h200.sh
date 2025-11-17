#!/bin/bash

cd /home/bs82/max-k-cut-parallel
mkdir -p logs

# Clean module environment
module purge || true
unset LD_PRELOAD || true
export XALT_EXECUTABLE_TRACKING=0
export XALT_RUNNABLE=0

# Load conda setup
source ./local_miniforge/etc/profile.d/conda.sh
conda activate /home/bs82/max-k-cut-parallel/ray311

# Ray settings
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_memory_monitor_refresh_ms=0

# Launch the code
python parallel_rank_1.py
