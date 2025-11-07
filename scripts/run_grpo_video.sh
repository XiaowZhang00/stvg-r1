
export WANDB_PROJECT=Video-GRPO
export WANDB_NAME=Qwen2.5_7b_TG

export PYTHONPATH=".:$PYTHONPATH"
OUTDIR=outputs_video

export DEBUG_MODE="true"
export LOG_PATH="./qwen2.5_7b_vl_stvg_video.txt"

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12361" \
    src/open_r1/grpo_video.py \
    --deepspeed scripts/zero3_offload.json \
    --output_dir $OUTDIR \
    --model_name_or_path /data/Qwen2.5-VL-7B-Instruct \
    --preprocessed_data_path ./hcstvgv2_preprocessed_data_maxpix_2048 \
    --train_data_path ./hcstvgv2/anno_v2/train_marker_v2.json \
    --eval_data_path ./hcstvgv2/anno_v2/train_marker_v2.json \
    --video_folder ./vipdata/hcstvgv2video \
    --dataset_name charades \
    --max_prompt_length 8192 \
    --max_completion_length 1024 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name $WANDB_NAME \
    --report_to wandb \
    --save_steps 50 \
    --save_only_model true
