## HOW TO TRAIN RWKV-7 on MiniPile (1.5G tokens) ##

For reference, use python 3.10+, torch 2.5+, cuda 12.5+, latest deepspeed, but **keep pytorch-lightning==1.9.5**

### Note: seems deepspeed 0.17.x is buggy (worse loss or divergence). Use 0.16.8 for reference (maybe --layerwise_lr 0 can fix it)

The default config only requires 1 GPU with 10G VRAM (you can reduce bsz if you have less VRAM), so it's easy to test.

**Train RWKV-7:**
```
# you can use latest torch + latest cuda (not limited to cu121)
pip install torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning==1.9.5 deepspeed wandb ninja --upgrade

# train RWKV-7
cd RWKV-v7/train_temp/ 

# download minipile .bin .idx to train_temp/data first (download link in demo-training-prepare.sh)

# this will generate the initial weight rwkv-init.pth in out/....../
sh ./demo-training-prepare.sh

# this will load rwkv-init.pth and train the model. you may want to log in to wandb first
sh ./demo-training-run.sh

your out/....../train_log.txt should have losses similar to (must be within +-0.01 or something is wrong):
0 4.875856 131.0863 0.00059975 2025-04-24 02:23:42.481256 0
1 4.028621 56.1834 0.00059899 2025-04-24 02:28:16.674463 1
2 3.801625 44.7739 0.00059773 2025-04-24 02:32:51.059568 2
3 3.663070 38.9808 0.00059597 2025-04-24 02:37:25.409892 3
4 3.578974 35.8368 0.00059371 2025-04-24 02:41:59.711315 4
5 3.510906 33.4786 0.00059096 2025-04-24 02:46:33.990839 5
6 3.462345 31.8917 0.00058771 2025-04-24 02:51:08.378331 6
7 3.412196 30.3318 0.00058399 2025-04-24 02:55:42.927474 7
8 3.376724 29.2747 0.00057978 2025-04-24 03:00:17.504665 8
9 3.336911 28.1321 0.00057511 2025-04-24 03:04:52.006063 9
10 3.313411 27.4787 0.00056999 2025-04-24 03:09:27.563336 10
11 3.295895 27.0016 0.00056441 2025-04-24 03:14:01.786079 11
```

RWKV-7 is the whole model with carefully set stuffs, including different init / wd / lr for each parameter, so it's readily scalable and very stable (spike-free).

But the price to pay is there is no good simple "RWKV-7 layer" because a pytorch layer can't make sure itself is using correct init and hyperparameters.

So if you need to use RWKV-7 for another task, please study train_temp code (only several hundred lines) and change it to suit you.

### Training on The Pile (streaming JSONL)

If you already have the EleutherAI Pile shards as `.jsonl.zst` files (for example under `/mnt/TrainingData/Training/EleutherAI_ThePile_v1/pile/train`), you can launch the RWKV-7 Pile recipes directly without converting to binidx. The helper scripts `demo-training-prepare-v7-pile.sh` and `demo-training-run-v7-pile.sh` now default to that location and use the streaming dataloader. Adjust the `DATA_ROOT` variable in each script if your copy of the dataset lives elsewhere. The scripts will also check that the tokenizer vocabulary (`../rwkv_vocab_v20230424.txt`) is available before starting.

#### 0.36B training recipe (Chinchilla aligned)

- Scripts now target a ~0.36 B parameter RWKV-7 (18 layers × 1024 width, `ctx_len=4096`). The output directory defaults to `out/L18-D1024-x070` so the larger experiments remain untouched.
- Stage 1 (`demo-training-prepare-v7-pile.sh`) streams The Pile, builds an initial checkpoint with `micro_bsz=1`, bf16 precision, and gradient checkpointing enabled to keep memory low during the one-off init pass.
- Stage 2 (`demo-training-run-v7-pile.sh`) runs from scratch with `micro_bsz=24`, `ctx_len=4096`, and `epoch_steps=1000`. With 80 mini-epochs this covers ≈7.9 B tokens (`4096 × 24 × 1000 × 80`), matching the Chinchilla guideline of ~20 tokens per parameter.
- Warmup spans 1500 steps and the optimizer uses `(β₁, β₂) = (0.9, 0.997)`. Checkpoints drop every 5 mini-epochs (~0.49 B tokens). DeepSpeed bucket sizes stay at 200 MB to keep communication efficient.
- The tokenizer remains `rwkv_vocab_v20230424.txt` (≈65 k entries); consequently both scripts set `--vocab_size 65536`.
- After training you can sample using `tools/generate.py`; it now defaults to the 18×1024, `ctx_len=4096` configuration and pads sequences to keep the CUDA kernel happy.

RWKV-7 weight example for 1.5B (L24-D2048, vocab 65536):
| name                | shape         | comment      | initialization  |
|---------------------|---------------|--------------|-----------------|
| emb.weight          | [65536, 2048] | wdecay       | see code        |
| blocks.0.ln0.weight | [2048]        | for layer 0  | 1               |
| blocks.0.ln0.bias   | [2048]        | for layer 0  | 0               |
|                     |               |              |                 |
| blocks.*.ln1.weight | [2048]        |              | 1               |
| blocks.*.ln1.bias   | [2048]        |              | 0               |
| blocks.*.att.x_r    | [1, 1, 2048]  |              | see code        |
| blocks.*.att.x_w    | [1, 1, 2048]  |              | see code        |
| blocks.*.att.x_k    | [1, 1, 2048]  |              | see code        |
| blocks.*.att.x_v    | [1, 1, 2048]  |              | see code        |
| blocks.*.att.x_a    | [1, 1, 2048]  |              | see code        |
| blocks.*.att.x_g    | [1, 1, 2048]  |              | see code        |
| blocks.*.att.w0     | [1, 1, 2048]  | lr 2x        | see code        |
| blocks.*.att.w1     | [2048, 96]    |              | 0               |
| blocks.*.att.w2     | [96, 2048]    |              | see code        |
| blocks.*.att.a0     | [1, 1, 2048]  |              | 0               |
| blocks.*.att.a1     | [2048, 96]    |              | 0               |
| blocks.*.att.a2     | [96, 2048]    |              | see code        |
| blocks.*.att.v0     | [1, 1, 2048]  | for layer 1+ | 1               |
| blocks.*.att.v1                | [2048, 64]   | for layer 1+ | 0         |
| blocks.*.att.v2                | [64, 2048]   | for layer 1+ | see code  |
| blocks.*.att.g1                | [2048, 256]  |              | 0         |
| blocks.*.att.g2                | [256, 2048]  |              | see code  |
| blocks.*.att.k_k               | [1, 1, 2048] |              | 1         |
| blocks.*.att.k_a               | [1, 1, 2048] |              | 1         |
| blocks.*.att.r_k               | [32, 64]     |              | 0         |
| blocks.*.att.receptance.weight | [2048, 2048] | wdecay       | see code  |
| blocks.*.att.key.weight        | [2048, 2048] | wdecay       | see code  |
| blocks.*.att.value.weight      | [2048, 2048] | wdecay       | see code  |
| blocks.*.att.output.weight     | [2048, 2048] | wdecay       | 0         |
| blocks.*.att.ln_x.weight       | [2048]       |              | see code  |
| blocks.*.att.ln_x.bias         | [2048]       |              | 0         |
|                                |              |              |           |
| blocks.*.ln2.weight            | [2048]       |              | 1         |
| blocks.*.ln2.bias              | [2048]       |              | 0         |
| blocks.*.ffn.x_k               | [1, 1, 2048] |              | see code  |
| blocks.*.ffn.key.weight        | [8192, 2048] | wdecay       | see code  |
| blocks.*.ffn.value.weight      | [2048, 8192] | wdecay       | 0         |
|                                |              |              |           |
| ln_out.weight | [2048]        |        | 1         |
| ln_out.bias   | [2048]        |        | 0         |
| head.weight   | [65536, 2048] | wdecay | see code  |
