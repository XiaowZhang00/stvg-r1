#!/bin/bash

MODEL_PATH="/data/Qwen2.5-VL-7B-Instruct"
DATASET="hcstvgv2"
TRAIN_DATA="./hcstvgv2/anno_v2/train_marker_v2.json"
EVAL_DATA="./hcstvgv2/anno_v2/val_marker_v2.json"
VIDEO_FOLDER="./vipdata/hcstvgv2video"
MAX_PIX=2048
MIN_PIX=16
NUM_WORKERS=16
OUTPUT_DIR=./hcstvgv2_preprocessed_data_maxpix_2048

python preprocess_dataset.py \
  --model_name $MODEL_PATH \
  --dataset $DATASET \
  --train_data_path $TRAIN_DATA \
  --eval_data_path $EVAL_DATA \
  --video_folder $VIDEO_FOLDER \
  --max_pix_size $MAX_PIX \
  --min_pix_size $MIN_PIX \
  --num_workers $NUM_WORKERS \
  --output_dir $OUTPUT_DIR
