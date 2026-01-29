#!/bin/bash

MODEL=LLaMA2_hf_chat_7B
MAX_LEN=8192
RUN_NAME=cache4096-sink4-fft_ratio0.5-bs64-8k
OUTPUT_DIR=experiments/$MODEL/LongAlpaca-16k-length/$MAX_LEN/$RUN_NAME

mkdir -p $OUTPUT_DIR
cp sft.sh $OUTPUT_DIR/train.sh


NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 WANDB_PROJECT="fourier" torchrun --nproc_per_node=4 --master_port=49688 supervised-fine-tune.py  \
        --model_name_or_path /path/to/model \
        --bf16 True \
        --output_dir $OUTPUT_DIR       \
        --cache_dir /path/to/.cache/huggingface \
        --model_max_length $MAX_LEN \
        --report_to wandb \
        --run_name $MODEL-$RUN_NAME-$MAX_LEN \
        --use_flash_attn True \
        --data_path data/LongAlpaca-16k-length/LongAlpaca-16k-length.json \
        --low_rank_training True \
        --num_train_epochs 5  \
        --per_device_train_batch_size 1     \
        --per_device_eval_batch_size 1     \
        --gradient_accumulation_steps 16     \
        --evaluation_strategy "no"     \
        --save_strategy "epoch"     \
        --save_total_limit 1     \
        --learning_rate 2e-5     \
        --weight_decay 0.0     \
        --warmup_steps 20     \
        --lr_scheduler_type "constant_with_warmup"     \
        --logging_steps 1     \
        --deepspeed "ds_configs/stage2.json" \
        --tf32 True \
        --is_iterate True \
        --sink_size 4 \
        --fft_ratio 0.5 \
        --cache_size 4096 \
| tee -a $OUTPUT_DIR/run.log
