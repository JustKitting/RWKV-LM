#!/usr/bin/env python
"""Compute perplexity of an RWKV checkpoint on a text file."""

import argparse
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("RWKV_JIT_ON", "0")
os.environ.setdefault("RWKV_MY_TESTING", "x070")
os.environ.setdefault("RWKV_HEAD_SIZE", "64")

from src.model import RWKV
from src.tokenizer import RWKVTokenizer


class TextDataset(Dataset):
    def __init__(self, text_file: Path, tokenizer: RWKVTokenizer, ctx_len: int, stride: int | None = None):
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        self.tokens = tokenizer.encode(text)
        self.ctx_len = ctx_len
        self.stride = stride or ctx_len
        if self.stride <= 0:
            raise ValueError("Stride must be positive")

        required = self.ctx_len + 1
        total = len(self.tokens)
        if total < required:
            self.starts = []
        else:
            max_start = total - required
            starts = list(range(0, max_start + 1, self.stride))
            if starts[-1] != max_start:
                starts.append(max_start)
            self.starts = starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        end = start + self.ctx_len + 1
        chunk = self.tokens[start:end]
        return torch.tensor(chunk, dtype=torch.long)


def pad_to_multiple(tensor: torch.Tensor, chunk: int = 16):
    seq_len = tensor.size(1)
    pad_len = (-seq_len) % chunk
    if pad_len:
        pad_token = tensor[:, -1:].repeat(1, pad_len)
        tensor = torch.cat([tensor, pad_token], dim=1)
    return tensor


def evaluate(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = RWKVTokenizer(args.tokenizer)

    model_args = argparse.Namespace(
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        dim_att=args.dim_att or args.n_embd,
        dim_ffn=args.dim_ffn,
        ctx_len=args.ctx_len,
        vocab_size=args.vocab_size,
        head_size=args.head_size,
        weight_decay=0,
        grad_cp=0,
        betas=(0.9, 0.99),
        lr_init=0.0,
        lr_final=0.0,
        adam_eps=1e-8,
        accelerator='gpu' if device == 'cuda' else 'cpu',
        devices=1,
        precision='bf16' if device == 'cuda' else 'fp32',
        strategy='none',
        my_testing=os.environ['RWKV_MY_TESTING'],
        use_flat_norm=args.use_flat_norm,
        use_flat_norm_full=args.use_flat_norm_full,
    )

    model = RWKV(model_args)
    state = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state)
    model.to(device).eval()

    dataset = TextDataset(
        Path(args.text),
        tokenizer,
        args.ctx_len,
        stride=args.stride if args.stride > 0 else None,
    )
    dataset_len = len(dataset)
    if dataset_len == 0:
        print("Dataset is empty after tokenization; nothing to evaluate.")
        return

    print(
        f"Evaluating {dataset_len:,} windows (stride={dataset.stride}) "
        f"covering {len(dataset.tokens):,} tokens.",
        flush=True,
    )

    loader = DataLoader(dataset, batch_size=args.batch_size)
    num_batches = len(loader)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, 1):
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            inputs = pad_to_multiple(inputs)
            logits = model(inputs)
            logits = logits[:, :targets.size(1), :]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1), reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += targets.numel()

            if args.log_interval and (batch_idx % args.log_interval == 0 or batch_idx == num_batches):
                progress = (batch_idx / num_batches) * 100
                avg_loss = total_loss / total_tokens if total_tokens else float('inf')
                ppl_so_far = math.exp(avg_loss) if total_tokens else float('inf')
                print(
                    f"[{batch_idx}/{num_batches} batches | {progress:5.1f}%] "
                    f"tokens={total_tokens:,} ppl={ppl_so_far:.3f}",
                    flush=True,
                )

    ppl = math.exp(total_loss / total_tokens)
    print(f"Perplexity: {ppl:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RWKV perplexity on text")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("text", type=str, help="Text file for evaluation")
    parser.add_argument("--tokenizer", default="../rwkv_vocab_v20230424.txt")
    parser.add_argument("--n_layer", type=int, required=True)
    parser.add_argument("--n_embd", type=int, required=True)
    parser.add_argument("--dim_att", type=int, default=0)
    parser.add_argument("--dim_ffn", type=int, default=0)
    parser.add_argument("--ctx_len", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=65536)
    parser.add_argument("--head_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--stride", type=int, default=0,
                        help="Stride between evaluation windows (0 defaults to ctx_len)")
    parser.add_argument("--use_flat_norm", action="store_true")
    parser.add_argument("--use_flat_norm_full", action="store_true")
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Print progress every N batches (0 to disable)")
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
