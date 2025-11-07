
export CUDA_VISIBLE_DEVICES=0

# MODEL_BASE=mllm/Qwen2.5-VL-7B-Instruct
MODEL_BASE=./outputs_video

python evaluate.py \
     --model_base $MODEL_BASE \
     --dataset hcstvgv2 \
     --bbox_data_file ./hcstvgv2/anno_v2/train_v2.json \
     --mask_data_dir ./vipdata/hcstvgv2 \
     --checkpoint_dir ./outputs_video