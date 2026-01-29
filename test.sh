#!/bin/bash

function run(){
SEQ_LEN=${1:-8192}
MODEL=LLaMA2_hf_7B
MAX_LEN=8192
RUN_NAME=cache4096-sink4-recent0-fft_ratio0.5-iterate-flash
EVAL_DATA=pg19/test
PEFT_MODEL=experiments/$MODEL/$MAX_LEN/$RUN_NAME/checkpoint-1000
OUTPUT_DIR=$PEFT_MODEL/$EVAL_DATA

mkdir -p $OUTPUT_DIR
cp test.sh $OUTPUT_DIR/test.sh

python3 eval.py \
    --seq_len $SEQ_LEN \
    --context_size $MAX_LEN \
    --batch_size 1 \
    --data_path data/$EVAL_DATA.bin \
    --base_model /path/to/$MODEL \
    --peft_model $PEFT_MODEL \
    --flash_attn True \
    --is_iterate True \
    --sink_size 4 \
    --recent_size 0 \
    --fft_ratio 0.5 \
    --cache_size 4096 \
| tee -a $OUTPUT_DIR/test$SEQ_LEN.log
}

run 32768
run 16384
run 8192
run 4096
run 2048
