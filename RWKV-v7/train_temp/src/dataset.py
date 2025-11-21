########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import math
import os
from typing import Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info

from .binidx import MMapIndexedDataset


def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


class _BaseDataset:
    """Common helpers shared by both dataset backends."""

    def __init__(self, args):
        self.args = args
        self.global_rank = 0
        self.world_size = 1
        self.real_epoch = 0
        self.datasets = self  # lightning callback expects this attribute

    def set_distributed(self, rank: int, world_size: int, epoch: int):
        self.global_rank = rank
        self.world_size = world_size
        self.real_epoch = epoch

    def __len__(self) -> int:  # number of samples per epoch for each rank
        return self.args.epoch_steps * self.args.micro_bsz


class BinidxDataset(_BaseDataset, Dataset):
    def __init__(self, args):
        super().__init__(args)

        self.vocab_size = args.vocab_size
        rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")

        self.data = MMapIndexedDataset(args.data_file)
        self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size
        rank_zero_info(f"Data has {self.data_size} tokens.")

        self.samples_per_epoch = args.epoch_steps * args.real_bsz
        assert self.samples_per_epoch == 40320
        rank_zero_info(f"########## train stage {args.train_stage} ##########")
        dataset_slot = self.data_size // args.ctx_len

        # HERE recompute magic_prime after you change ctx_len or swap datasets
        assert is_prime(args.magic_prime)
        assert args.magic_prime % 3 == 2
        assert args.magic_prime / dataset_slot > 0.9 and args.magic_prime / dataset_slot <= 1

    def __getitem__(self, idx):
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size

        ctx_len = args.ctx_len
        req_len = ctx_len + 1
        magic_prime = args.magic_prime

        ii = 1 + epoch * self.samples_per_epoch + (idx * world_size) + rank

        factor = (math.sqrt(5) - 1) / 2
        factor = int(magic_prime * factor)
        i = ((factor * ii * ii * ii) % magic_prime) * ctx_len

        dix = self.data.get(idx=0, offset=i, length=req_len).astype(int)

        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)

        return x, y


class StreamingJsonlDataset(_BaseDataset, Dataset):
    def __init__(self, args):
        super().__init__(args)

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("Install `datasets` to enable streaming data (--data_type stream_jsonl)") from exc

        from .tokenizer import RWKVTokenizer

        data_root = args.data_file
        if not data_root:
            raise ValueError("--data_file must point to a directory or file pattern for streaming")

        if os.path.isdir(data_root):
            data_files = os.path.join(data_root, args.stream_pattern)
        else:
            data_files = data_root

        rank_zero_info(f"Loading streaming dataset from {data_files}")
        self.ds = load_dataset(
            "json",
            data_files={"train": data_files},
            streaming=True,
            split="train",
        )

        tokenizer_path = args.tokenizer_path or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "rwkv_vocab_v20230424.txt")
        )
        self.tokenizer = RWKVTokenizer(tokenizer_path)
        self.vocab_size = args.vocab_size or self.tokenizer.vocab_size
        rank_zero_info(f"Streaming vocab size = {self.vocab_size}")

        self.separator_tokens = self.tokenizer.encode(args.stream_separator)
        if len(self.separator_tokens) == 0:
            raise ValueError("Separator produced no tokens; adjust --stream_separator")

        self.stream_text_key = args.stream_text_key
        self.shuffle_buffer = max(0, int(args.stream_shuffle_buffer))

        self._buffer: List[int] = []
        self._generator: Optional[Iterator] = None
        self._needs_reset = True

    def set_distributed(self, rank: int, world_size: int, epoch: int):
        super().set_distributed(rank, world_size, epoch)
        self._needs_reset = True
        self._generator = None

    def _make_iterator(self) -> Iterator[List[int]]:
        ds = self.ds
        if self.world_size > 1:
            ds = ds.shard(self.world_size, self.global_rank, contiguous=True)

        if self.shuffle_buffer > 0:
            seed = self.args.random_seed if self.args.random_seed >= 0 else None
            if seed is None:
                seed = (self.global_rank + 1) * (self.real_epoch + 1)
            ds = ds.shuffle(seed=seed + self.real_epoch, buffer_size=self.shuffle_buffer)

        ctx_len = self.args.ctx_len

        for sample in ds:
            text = sample.get(self.stream_text_key, "")
            if not isinstance(text, str):
                continue
            tokens = self.tokenizer.encode(text)
            if not tokens:
                continue
            tokens.extend(self.separator_tokens)
            self._buffer.extend(tokens)

            while len(self._buffer) >= ctx_len + 1:
                chunk = self._buffer[: ctx_len + 1]
                del self._buffer[: ctx_len]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y

        # Exhausted this pass; mark for reset so next __getitem__ rebuilds iterator
        self._needs_reset = True
        raise StopIteration

    def __getitem__(self, idx):
        if self._needs_reset or self._generator is None:
            self._generator = self._make_iterator()
            self._needs_reset = False
        try:
            return next(self._generator)
        except StopIteration:
            self._generator = self._make_iterator()
            return next(self._generator)


class StreamingParquetDataset(_BaseDataset, Dataset):
    def __init__(self, args):
        super().__init__(args)

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("Install `datasets` to enable parquet streaming (--data_type stream_parquet)") from exc

        from .tokenizer import RWKVTokenizer

        data_root = args.data_file
        if not data_root:
            raise ValueError("--data_file must point to a directory or file pattern for streaming")

        if os.path.isdir(data_root):
            data_files = os.path.join(data_root, args.stream_pattern)
        else:
            data_files = data_root

        rank_zero_info(f"Loading streaming parquet dataset from {data_files}")
        self.ds = load_dataset(
            "parquet",
            data_files={"train": data_files},
            streaming=True,
            split="train",
        )

        tokenizer_path = args.tokenizer_path or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "rwkv_vocab_v20230424.txt")
        )
        self.tokenizer = RWKVTokenizer(tokenizer_path)
        self.vocab_size = args.vocab_size or self.tokenizer.vocab_size
        rank_zero_info(f"Streaming vocab size = {self.vocab_size}")

        self.separator_tokens = self.tokenizer.encode(args.stream_separator)
        if len(self.separator_tokens) == 0:
            raise ValueError("Separator produced no tokens; adjust --stream_separator")

        raw_fields = getattr(args, "stream_parquet_text_fields", "synthetic_answer")
        fields = [f.strip() for f in str(raw_fields).split(",") if f.strip()]
        if not fields:
            raise ValueError("Provide at least one column via --stream_parquet_text_fields")
        self.text_fields = fields

        self.shuffle_buffer = max(0, int(args.stream_shuffle_buffer))

        self._buffer: List[int] = []
        self._generator: Optional[Iterator] = None
        self._needs_reset = True

    def set_distributed(self, rank: int, world_size: int, epoch: int):
        super().set_distributed(rank, world_size, epoch)
        self._needs_reset = True
        self._generator = None

    def _make_iterator(self) -> Iterator[List[int]]:
        ds = self.ds
        if self.world_size > 1:
            ds = ds.shard(self.world_size, self.global_rank, contiguous=True)

        if self.shuffle_buffer > 0:
            seed = self.args.random_seed if self.args.random_seed >= 0 else None
            if seed is None:
                seed = (self.global_rank + 1) * (self.real_epoch + 1)
            ds = ds.shuffle(seed=seed + self.real_epoch, buffer_size=self.shuffle_buffer)

        ctx_len = self.args.ctx_len
        separator_str = self.args.stream_separator

        for sample in ds:
            parts: List[str] = []
            for field in self.text_fields:
                value = sample.get(field)
                if value is None:
                    continue
                if not isinstance(value, str):
                    value = str(value)
                value = value.strip()
                if value:
                    parts.append(value)

            if not parts:
                continue

            text = separator_str.join(parts)
            tokens = self.tokenizer.encode(text)
            if not tokens:
                continue
            tokens.extend(self.separator_tokens)
            self._buffer.extend(tokens)

            while len(self._buffer) >= ctx_len + 1:
                chunk = self._buffer[: ctx_len + 1]
                del self._buffer[: ctx_len]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y

        self._needs_reset = True
        raise StopIteration

    def __getitem__(self, idx):
        if self._needs_reset or self._generator is None:
            self._generator = self._make_iterator()
            self._needs_reset = False
        try:
            return next(self._generator)
        except StopIteration:
            self._generator = self._make_iterator()
            return next(self._generator)


class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args

        if args.data_type == "binidx":
            self.impl = BinidxDataset(args)
        elif args.data_type == "stream_jsonl":
            self.impl = StreamingJsonlDataset(args)
        elif args.data_type == "stream_parquet":
            self.impl = StreamingParquetDataset(args)
        else:
            raise ValueError(f"Unsupported data_type {args.data_type}")

        self.datasets = self.impl
        self.vocab_size = self.impl.vocab_size

    def set_distributed(self, rank: int, world_size: int, epoch: int):
        if hasattr(self.impl, "set_distributed"):
            self.impl.set_distributed(rank, world_size, epoch)

    def __len__(self):
        return len(self.impl)

    def __getitem__(self, idx):
        return self.impl[idx]
