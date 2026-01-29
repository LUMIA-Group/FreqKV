#!/bin/bash

MODEL=LLaMA2_hf_7B
MAX_LEN=8192
RUN_NAME=cache4096-sink4-recent0-fft_ratio0.5-iterate-flash
OUTPUT_DIR=experiments/$MODEL/$MAX_LEN/$RUN_NAME/checkpoint-1000

cd $OUTPUT_DIR

python zero_to_fp32.py . pytorch_model.bin

cd /path/to/repo/

python3 get_trainable_weights.py \
    --checkpoint_path $OUTPUT_DIR \
    --trainable_params "embed,norm" \
| tee -a $OUTPUT_DIR/weights.log

python3 merge_lora_weights_and_save_hf_model.py \
        --base_model /path/to/$MODEL \
        --peft_model $OUTPUT_DIR \
        --context_size $MAX_LEN \
        --save_path $OUTPUT_DIR/merged \
| tee -a $OUTPUT_DIR/merge.log
