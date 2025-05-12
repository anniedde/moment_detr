#!/bin/bash

# Add current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

VIDEO_DIR="/playpen-nas-ssd4/awang/scannet_scenes/videos"
QUERY_FILE="/playpen-nas-ssd4/awang/UniVTG/results/vlp-vlp/omni_mini_aio_unified__epo3_f10_b10g1_s0.1_0.1-clip-clip-2023_05_31_06/original_queries.jsonl"
OUTPUT_FILE="predictions_original_queries.json"

# Run the moment detection
python run_on_video/batch_run.py \
    --video_dir=${VIDEO_DIR} \
    --query_file=${QUERY_FILE} \
    --output_file=${OUTPUT_FILE} \
    --device="cuda:3" 