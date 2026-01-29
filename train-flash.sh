#!/bin/bash

# RedPajama: from huggingface

function run(){
FFT_RATIO=${1:-0.5}
MODEL=LLaMA2_hf_7B
MAX_LEN=8192
RUN_NAME=cache4096-sink4-recent0-fft_ratio$FFT_RATIO-iterate-flash
OUTPUT_DIR=experiments/$MODEL/$MAX_LEN/$RUN_NAME

mkdir -p $OUTPUT_DIR
cp train-flash.sh $OUTPUT_DIR/train.sh


NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 WANDB_PROJECT="fourier" torchrun --nproc_per_node=8 --master_port=50357 fine-tune.py  \
        --model_name_or_path /path/to/model \
        --bf16 True \
        --output_dir $OUTPUT_DIR       \
        --cache_dir /path/to/.cache/huggingface \
        --model_max_length $MAX_LEN \
        --report_to wandb \
        --run_name $MODEL-$RUN_NAME-$MAX_LEN \
        --use_flash_attn True \
        --low_rank_training True \
        --is_iterate True \
        --num_train_epochs 1  \
        --per_device_train_batch_size 1     \
        --per_device_eval_batch_size 1     \
        --gradient_accumulation_steps 8     \
        --evaluation_strategy "no"     \
        --save_strategy "steps"     \
        --save_steps 100     \
        --save_total_limit 2     \
        --learning_rate 2e-5     \
        --weight_decay 0.0     \
        --warmup_steps 20     \
        --lr_scheduler_type "constant_with_warmup"     \
        --logging_steps 1     \
        --deepspeed "ds_configs/stage2.json" \
        --tf32 True \
        --max_steps 1000 \
        --sink_size 4 \
        --recent_size 0 \
        --fft_ratio $FFT_RATIO \
| tee -a $OUTPUT_DIR/run.log
}

run 0.5
