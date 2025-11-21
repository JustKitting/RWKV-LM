#!/bin/bash
#######################################################################################################################
#
# RWKV-7 SYNTH training targeting ~100M parameters (mirrors baseline settings but swaps in SYNTH parquet data)
#
MODEL_TYPE="x070"
N_LAYER="12"
N_EMBD="768"
CTX_LEN="2048"
PROJ_DIR="out/L"$N_LAYER"-D"$N_EMBD"-"$MODEL_TYPE"-synth"
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
DATA_ROOT="/mnt/TrainingData/Training/SYNTH"
STREAM_PATTERN="synth_[0-3][0-9][0-9].parquet"
PARQUET_FIELDS="query,synthetic_reasoning,synthetic_answer"
TOKENIZER_PATH="$(realpath ../rwkv_vocab_v20230424.txt 2>/dev/null || true)"
if [ -z "$TOKENIZER_PATH" ]; then
  TOKENIZER_PATH="../rwkv_vocab_v20230424.txt"
fi
if [ ! -d "$DATA_ROOT" ]; then
  echo "Expected SYNTH parquet shards under $DATA_ROOT but directory was not found" >&2
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
 --data_file "$DATA_ROOT" --data_type "stream_parquet" --stream_pattern "$STREAM_PATTERN" --tokenizer_path "$TOKENIZER_PATH" \
 --stream_parquet_text_fields "$PARQUET_FIELDS" --stream_separator "\n\n" --stream_shuffle_buffer 2048 \
 --my_exit_tokens 332115325534 --magic_prime 81082817 \
 --num_nodes $N_NODE --micro_bsz $M_BSZ --n_layer $N_LAYER --n_embd $N_EMBD \
 --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps $WARMUP_STEPS --beta1 0.9 --beta2 $BETA_2 --adam_eps $ADAM_EPS --vocab_size 65536 \
 --weight_decay $W_DECAY --epoch_save $EPOCH_SAVE --head_size 64 \
 --accelerator gpu --devices $GPU_PER_NODE --precision bf16 --strategy deepspeed_stage_2 --grad_cp $GRAD_CP --enable_progress_bar true --ds_bucket_mb $DS_BUCKET_MB \
 --kl_eval_data_file "$DATA_ROOT" --kl_eval_data_type stream_parquet --kl_eval_stream_pattern "synth_[4-5][0-9][0-9].parquet" \
 --kl_eval_parquet_text_fields "$PARQUET_FIELDS" --kl_eval_stream_separator "\n\n" --kl_eval_stride 2048 --kl_eval_batch_size 2 --kl_eval_max_windows 256 \
 --mmlu_eval true --mmlu_eval_dataset_root .. --mmlu_eval_max_samples 256 --mmlu_eval_shuffle false --mmlu_eval_seed 42
