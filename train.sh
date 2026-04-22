#!/bin/bash

GPU_IDS="0"
NUM_GPUS=1
RUN_NUMBER=1

# Training config file
TRAIN_CONFIG="output_baseline/run${RUN_NUMBER}/train.json"

# Data config file
data_config="output_baseline/run${RUN_NUMBER}/data.json"

# Output directory for checkpoints
OUTPUT_DIR="output_baseline/run${RUN_NUMBER}"

uv run accelerate launch \
       --gpu_ids "${GPU_IDS}" \
       --num_processes ${NUM_GPUS} \
       -m omnivoice.cli.train \
       --train_config ${TRAIN_CONFIG} \
       --data_config ${data_config} \
       --output_dir ${OUTPUT_DIR}
