#!/bin/bash

# ===================================================================
# --- Set default command here ---
DEFAULT_COMMAND="python numerical_experiments.py"
#DEFAULT_COMMAND="python paper_2_test.py"
# ===================================================================

# Free GPU threshold (MiB).
THRESHOLD=100

# Find first 'free' GPU.
FREE_GPU_ID=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v thres="$THRESHOLD" '$2 < thres {print $1; exit}')
if [ -z "$FREE_GPU_ID" ]; then
    echo "Error: No free GPU available."
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$FREE_GPU_ID
echo "Free GPU found, CUDA_VISIBLE_DEVICES=$FREE_GPU_ID"

ARGUMENT=$1
if [ "$#" -eq 0 ] || [ "$ARGUMENT" -eq 0 ]; then
    eval $DEFAULT_COMMAND
elif [[ "$ARGUMENT" =~ ^[1-9][0-9]*$ ]]; then
    COMMAND_TO_RUN="python paper_${ARGUMENT}_test.py"
    eval $COMMAND_TO_RUN
else
    "$@"
fi