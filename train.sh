#!/bin/bash

GPU_IDS="0"
NUM_GPUS=1

# Training config file
TRAIN_CONFIG="examples/config/train_config_emilia.json"

# Data config file
data_config="examples/config/data_config_emilia_.json"

# Output directory for checkpoints
OUTPUT_DIR="exp/omnivoice_emilia"

uv run accelerate launch \
 --gpu_ids "${GPU_IDS}" \
        --num_processes ${NUM_GPUS} \
        -m omnivoice.cli.train \
        --train_config ${TRAIN_CONFIG} \
        --data_config ${data_config} \
        --output_dir ${OUTPUT_DIR}