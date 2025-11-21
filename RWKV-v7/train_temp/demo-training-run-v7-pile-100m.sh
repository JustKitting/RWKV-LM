#!/bin/bash
#######################################################################################################################
#
# Baseline RWKV-7 Pile training targeting ~100M parameters (no flat norm)
#
MODEL_TYPE="x070"
N_LAYER="12"
N_EMBD="768"
CTX_LEN="2048"
PROJ_DIR="out/L"$N_LAYER"-D"$N_EMBD"-"$MODEL_TYPE"-baseline"
WANDB_PROJECT="$(basename "$PROJ_DIR")"
#######################################################################################################################
#
M_BSZ="64"
LR_INIT="3e-4"
LR_FINAL="3e-5"
EPOCH_COUNT="15"
WARMUP_STEPS="500"
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
#######################################################################################################################
#
W_DECAY="0.1"
BETA_2="0.997"
ADAM_EPS="1e-8"
GRAD_CP=1
EPOCH_SAVE=3
N_NODE=1
GPU_PER_NODE=1
DS_BUCKET_MB=200
#######################################################################################################################
python train.py --load_model "0" --wandb "$WANDB_PROJECT" --proj_dir $PROJ_DIR --my_testing $MODEL_TYPE \
 --ctx_len $CTX_LEN --train_stage 3 --epoch_count $EPOCH_COUNT --epoch_begin 0 --epoch_steps 1000 \
 --data_file "$DATA_ROOT" --data_type "stream_jsonl" --stream_pattern "*.jsonl.zst" --tokenizer_path "$TOKENIZER_PATH" \
 --stream_text_key "text" --stream_separator "\n\n" --stream_shuffle_buffer 2048 \
 --my_exit_tokens 332115325534 --magic_prime 81082817 \
 --num_nodes $N_NODE --micro_bsz $M_BSZ --n_layer $N_LAYER --n_embd $N_EMBD \
 --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps $WARMUP_STEPS --beta1 0.9 --beta2 $BETA_2 --adam_eps $ADAM_EPS --vocab_size 65536 \
 --weight_decay $W_DECAY --epoch_save $EPOCH_SAVE --head_size 64 \
 --accelerator gpu --devices $GPU_PER_NODE --precision bf16 --strategy deepspeed_stage_2 --grad_cp $GRAD_CP --enable_progress_bar True --ds_bucket_mb $DS_BUCKET_MB
