#!/bin/bash
#######################################################################################################################
#
# This will generate the initial model weights for the ~1.0B parameter RWKV-7 configuration on The Pile
#
# Requires the EleutherAI Pile shards (.jsonl.zst) under /mnt/TrainingData/Training/EleutherAI_ThePile_v1/pile/train
# Adjust DATA_ROOT if your dataset lives elsewhere.
#
#######################################################################################################################
#
MODEL_TYPE="x070" # x070 => rwkv-7.0
#
N_LAYER="22"
N_EMBD="1536"
#
CTX_LEN="4096" # keep the same context length as the base recipe
PROJ_DIR="out/L"$N_LAYER"-D"$N_EMBD"-"$MODEL_TYPE # set output folder
#
#######################################################################################################################
#
DATA_ROOT="/mnt/TrainingData/Training/EleutherAI_ThePile_v1/pile/train"
TOKENIZER_PATH="$(realpath ../rwkv_vocab_v20230424.txt 2>/dev/null || true)"
if [ -z "$TOKENIZER_PATH" ]; then
  TOKENIZER_PATH="../rwkv_vocab_v20230424.txt"
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "Expected training data under $DATA_ROOT but directory was not found" >&2
  exit 1
fi

if [ ! -f "$TOKENIZER_PATH" ]; then
  echo "Expected tokenizer vocab at $TOKENIZER_PATH" >&2
  exit 1
fi

python train.py --wandb "" --proj_dir $PROJ_DIR \
 --data_file "$DATA_ROOT" --data_type "stream_jsonl" --stream_pattern "*.jsonl.zst" --tokenizer_path "$TOKENIZER_PATH" \
 --stream_text_key "text" --stream_separator "\n\n" --stream_shuffle_buffer 2048 --vocab_size 65536 --my_testing $MODEL_TYPE \
 --ctx_len $CTX_LEN --train_stage 1 --epoch_count 1 --epoch_begin 0 \
 --epoch_save 1 --weight_decay 0 --head_size 64 \
 --num_nodes 1 --micro_bsz 1 --n_layer $N_LAYER --n_embd $N_EMBD --my_exit_tokens 332115325534 --magic_prime 81082817 \
 --lr_init 2e-4 --lr_final 2e-4 --warmup_steps 100 --beta1 0.9 --beta2 0.997 --adam_eps 1e-8 \
 --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 1
