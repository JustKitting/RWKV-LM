"""Evaluation helpers for RWKV training."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F

from .tokenizer import RWKVTokenizer

_PAD_MULTIPLE = 16
_CHOICES = [" A", " B", " C", " D"]
_TEMPLATE = (
    "User: You are a very talented expert in <SUBJECT>. Answer this question:\n"
    "<Q>\n"
    "A. <|A|>\n"
    "B. <|B|>\n"
    "C. <|C|>\n"
    "D. <|D|>\n\n"
    "Assistant: The answer is"
)


def _default_tokenizer_path() -> Path:
    return Path(__file__).resolve().parents[2] / "rwkv_vocab_v20230424.txt"


def _pad_to_multiple(tensor: torch.Tensor, multiple: int = _PAD_MULTIPLE) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape {tuple(tensor.shape)}")
    seq_len = tensor.size(1)
    pad_len = (-seq_len) % multiple
    if pad_len:
        pad_token = tensor[:, -1:].repeat(1, pad_len)
        tensor = torch.cat([tensor, pad_token], dim=1)
    return tensor


class _EvalWindowDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, ctx_len: int, stride: int):
        if tokens.ndim != 1:
            raise ValueError("tokens must be 1D")
        if stride <= 0:
            stride = ctx_len
        self.tokens = tokens
        self.ctx_len = ctx_len
        self.stride = stride
        required = ctx_len + 1
        total = tokens.size(0)
        if total < required:
            self.starts: List[int] = []
        else:
            max_start = total - required
            starts = list(range(0, max_start + 1, stride))
            if starts[-1] != max_start:
                starts.append(max_start)
            self.starts = starts

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = self.starts[idx]
        end = start + self.ctx_len + 1
        return self.tokens[start:end]


@dataclass
class _EvalResult:
    scalars: Dict[str, float]
    details: Dict[str, int | float]
    label: str


class EvaluationManager:
    """Runs KL divergence and MMLU evaluations during training."""

    def __init__(self, args):
        self.args = args
        self.eval_log_path = Path(args.proj_dir) / "eval_metrics.jsonl"
        self.eval_log_path.parent.mkdir(parents=True, exist_ok=True)

        tokenizer_path = getattr(args, "tokenizer_path", "") or _default_tokenizer_path()
        self.tokenizer = RWKVTokenizer(str(tokenizer_path))

        self._kl_dataset: Optional[_EvalWindowDataset] = None
        self._kl_loader: Optional[DataLoader] = None
        self._kl_mode: Optional[str] = None
        self._kl_stream_config: Optional[dict] = None

        text_path = getattr(args, "kl_eval_text", "")
        if text_path:
            self._kl_mode = "text"
            self._prepare_kl_text(Path(text_path))
        else:
            stream_file = getattr(args, "kl_eval_data_file", "")
            if stream_file:
                self._kl_mode = "stream"
                self._prepare_kl_stream(stream_file)

        self._kl_enabled = self._kl_mode is not None

        self._mmlu_evaluator: Optional[_MMLUEvaluator] = None
        if getattr(args, "mmlu_eval", False):
            self._mmlu_evaluator = _MMLUEvaluator(args, self.tokenizer)

    # -------------------------------------------------------------------------------------
    # KL evaluation
    # -------------------------------------------------------------------------------------
    def _prepare_kl_text(self, text_path: Path) -> None:
        if not text_path.is_file():
            raise FileNotFoundError(f"KL eval text file not found: {text_path}")
        with text_path.open("r", encoding="utf-8") as f:
            text = f.read()
        tokens = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        dataset = _EvalWindowDataset(
            tokens,
            ctx_len=int(self.args.ctx_len),
            stride=int(getattr(self.args, "kl_eval_stride", 0) or self.args.ctx_len),
        )
        self._kl_dataset = dataset
        if len(dataset) == 0:
            raise ValueError(
                "KL eval dataset has no usable windows. "
                "Increase file size or adjust --kl_eval_stride"
            )
        batch_size = int(getattr(self.args, "kl_eval_batch_size", 1))
        self._kl_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def _prepare_kl_stream(self, data_file: str) -> None:
        data_type = getattr(self.args, "kl_eval_data_type", "") or str(self.args.data_type)
        data_file = str(data_file)
        if data_type not in {"stream_jsonl", "stream_parquet"}:
            raise ValueError("--kl_eval_data_type must be 'stream_jsonl' or 'stream_parquet'")

        stream_pattern = getattr(self.args, "kl_eval_stream_pattern", "")
        if os.path.isdir(data_file):
            if not stream_pattern:
                raise ValueError("Provide --kl_eval_stream_pattern when kl_eval_data_file is a directory")
            data_files = os.path.join(data_file, stream_pattern)
        else:
            data_files = data_file

        separator = getattr(self.args, "kl_eval_stream_separator", "") or self.args.stream_separator
        if not isinstance(separator, str) or len(separator) == 0:
            raise ValueError("KL eval separator must be a non-empty string")

        if data_type == "stream_jsonl":
            text_key = getattr(self.args, "kl_eval_stream_text_key", "") or self.args.stream_text_key
            if not text_key:
                raise ValueError("Provide --kl_eval_stream_text_key for JSONL KL evaluation")
            text_fields = [text_key]
        else:
            raw_fields = getattr(self.args, "kl_eval_parquet_text_fields", "") or self.args.stream_parquet_text_fields
            fields = [f.strip() for f in str(raw_fields).split(",") if f.strip()]
            if not fields:
                raise ValueError("Provide --kl_eval_parquet_text_fields for parquet KL evaluation")
            text_fields = fields

        self._kl_stream_config = {
            "data_type": data_type,
            "data_files": data_files,
            "separator": separator,
            "text_fields": text_fields,
        }

    def _run_kl_eval(self, pl_module: torch.nn.Module) -> _EvalResult:
        if self._kl_mode == "text":
            if not self._kl_dataset or not self._kl_loader:
                raise RuntimeError("KL text dataset not prepared")
            return self._run_kl_eval_text(pl_module)
        if self._kl_mode == "stream":
            if not self._kl_stream_config:
                raise RuntimeError("KL streaming dataset not prepared")
            return self._run_kl_eval_stream(pl_module)
        raise RuntimeError("KL evaluation is disabled")

    def _run_kl_eval_text(self, pl_module: torch.nn.Module) -> _EvalResult:
        assert self._kl_dataset is not None and self._kl_loader is not None

        max_windows = int(getattr(self.args, "kl_eval_max_windows", 0))
        if max_windows <= 0:
            max_windows = len(self._kl_dataset)

        device = next(pl_module.parameters()).device
        was_training = pl_module.training
        pl_module.eval()

        total_loss = 0.0
        total_tokens = 0
        total_windows = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self._kl_loader):
                if total_windows >= max_windows:
                    break
                remaining = max_windows - total_windows
                batch = batch[:remaining]
                if batch.size(0) == 0:
                    continue
                batch = batch.to(device)
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                inputs = _pad_to_multiple(inputs)
                logits = pl_module(inputs)
                logits = logits[:, : targets.size(1), :]
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="sum",
                )
                total_loss += loss.item()
                total_tokens += targets.numel()
                total_windows += batch.size(0)

        if was_training:
            pl_module.train()

        if total_tokens == 0:
            raise RuntimeError("KL evaluation produced zero tokens")

        avg_kl = total_loss / total_tokens
        ppl = math.exp(avg_kl)
        return _EvalResult(
            scalars={
                "eval/kl_div_nats": avg_kl,
                "eval/perplexity": ppl,
            },
            details={
                "tokens": total_tokens,
                "windows": total_windows,
            },
            label="kl_eval",
        )

    def _run_kl_eval_stream(self, pl_module: torch.nn.Module) -> _EvalResult:
        cfg = self._kl_stream_config or {}
        data_type = cfg.get("data_type")
        data_files = cfg.get("data_files")
        separator = cfg.get("separator", "\n\n")
        separator_tokens = self.tokenizer.encode(separator)
        if len(separator_tokens) == 0:
            raise ValueError("KL eval separator produced no tokens")
        text_fields = cfg.get("text_fields", [])

        if data_type not in {"stream_jsonl", "stream_parquet"}:
            raise ValueError("Unsupported KL streaming data type")

        max_windows = int(getattr(self.args, "kl_eval_max_windows", 0))
        if max_windows <= 0:
            max_windows = int(getattr(self.args, "kl_eval_batch_size", 1)) * 256
        max_windows = max(1, max_windows)

        ctx_len = int(self.args.ctx_len)
        stride = int(getattr(self.args, "kl_eval_stride", 0) or ctx_len)
        batch_size = int(getattr(self.args, "kl_eval_batch_size", 1))

        import datasets  # lazy import

        if data_type == "stream_jsonl":
            ds = datasets.load_dataset(
                "json",
                data_files={"eval": data_files},
                streaming=True,
                split="eval",
            )
        else:
            ds = datasets.load_dataset(
                "parquet",
                data_files={"eval": data_files},
                streaming=True,
                split="eval",
            )

        device = next(pl_module.parameters()).device
        was_training = pl_module.training
        pl_module.eval()

        total_loss = 0.0
        total_tokens = 0
        total_windows = 0

        buffer: List[int] = []

        def flush_buffer(force=False):
            nonlocal buffer
            nonlocal total_windows
            out_batches: List[torch.Tensor] = []
            while len(buffer) >= ctx_len + 1 and total_windows < max_windows:
                chunk = buffer[: ctx_len + 1]
                if stride <= 0:
                    del buffer[: ctx_len]
                else:
                    del buffer[: stride]
                out_batches.append(torch.tensor(chunk, dtype=torch.long))
                total_windows += 1
            if force:
                buffer = []
            return out_batches

        with torch.no_grad():
            for sample in ds:
                parts: List[str] = []
                for field in text_fields:
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
                text = separator.join(parts)
                tokens = self.tokenizer.encode(text)
                if not tokens:
                    continue
                tokens.extend(separator_tokens)
                buffer.extend(tokens)

                batches = flush_buffer()
                if not batches:
                    continue
                stacked = torch.stack(batches).to(device)
                inputs = stacked[:, :-1]
                targets = stacked[:, 1:]
                inputs = _pad_to_multiple(inputs)
                logits = pl_module(inputs)
                logits = logits[:, : targets.size(1), :]
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="sum",
                )
                total_loss += loss.item()
                total_tokens += targets.numel()

                if total_windows >= max_windows:
                    break

        if was_training:
            pl_module.train()

        if total_tokens == 0:
            raise RuntimeError("KL evaluation produced zero tokens from streaming dataset")

        avg_kl = total_loss / total_tokens
        ppl = math.exp(avg_kl)
        return _EvalResult(
            scalars={
                "eval/kl_div_nats": avg_kl,
                "eval/perplexity": ppl,
            },
            details={
                "tokens": total_tokens,
                "windows": total_windows,
            },
            label="kl_eval",
        )

    # -------------------------------------------------------------------------------------
    # MMLU evaluation
    # -------------------------------------------------------------------------------------
    def _run_mmlu_eval(self, pl_module: torch.nn.Module, should_checkpoint: bool) -> Optional[_EvalResult]:
        if not self._mmlu_evaluator or not should_checkpoint:
            return None
        try:
            return self._mmlu_evaluator.evaluate(pl_module)
        except Exception as exc:
            return _EvalResult(
                scalars={},
                details={"error": str(exc)},
                label="mmlu_error",
            )

    # -------------------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------------------
    def run_epoch_end(
        self,
        pl_module: torch.nn.Module,
        *,
        real_epoch: int,
        global_step: int,
        should_checkpoint: bool,
    ) -> Tuple[Dict[str, float], List[_EvalResult]]:
        results: List[_EvalResult] = []

        if self._kl_enabled:
            try:
                results.append(self._run_kl_eval(pl_module))
            except Exception as exc:
                results.append(
                    _EvalResult(
                        scalars={},
                        details={"error": str(exc)},
                        label="kl_eval_error",
                    )
                )

        mmlu_res = self._run_mmlu_eval(pl_module, should_checkpoint)
        if mmlu_res is not None:
            results.append(mmlu_res)

        scalars: Dict[str, float] = {}
        serialized: List[Dict[str, object]] = []
        for item in results:
            if item.scalars:
                scalars.update(item.scalars)
            serialized.append(
                {
                    "label": item.label,
                    "scalars": item.scalars,
                    "details": item.details,
                }
            )

        if serialized:
            record = {
                "epoch": real_epoch,
                "global_step": global_step,
                "timestamp": datetime.utcnow().isoformat(),
                "entries": serialized,
            }
            with self.eval_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        return scalars, results


class _MMLUEvaluator:
    def __init__(self, args, tokenizer: RWKVTokenizer):
        self.args = args
        self.tokenizer = tokenizer
        dataset_root = getattr(args, "mmlu_eval_dataset_root", "")
        if dataset_root:
            root = Path(dataset_root)
        else:
            root = Path(__file__).resolve().parents[2]
        self.splits: Dict[str, "datasets.Dataset"] = {}
        self._load_split(root, "test", "mmlu_test_dataset")
        if getattr(args, "mmlu_eval_use_dev", False):
            self._load_split(root, "dev", "mmlu_dev_dataset")

        if not self.splits:
            raise FileNotFoundError(
                "No MMLU dataset found. Provide --mmlu_eval_dataset_root pointing to folders"
            )

        self.choice_token_ids = [self._encode_choice(choice) for choice in _CHOICES]
        if not all(len(tok) == 1 for tok in self.choice_token_ids):
            raise ValueError("MMLU choice tokens are not single-token in tokenizer")
        self.choice_token_ids = [tok[0] for tok in self.choice_token_ids]

    def _load_split(self, root: Path, split_name: str, folder: str) -> None:
        dataset_path = root / folder
        if not dataset_path.exists():
            return
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise ImportError("Install `datasets` to enable MMLU evaluation") from exc
        self.splits[split_name] = load_from_disk(str(dataset_path))

    def _encode_choice(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def _prepare_indices(self, split: str, total: int) -> List[int]:
        indices = list(range(total))
        if getattr(self.args, "mmlu_eval_shuffle", False):
            seed = int(getattr(self.args, "mmlu_eval_seed", 42))
            random.Random(seed).shuffle(indices)
        limit = int(getattr(self.args, "mmlu_eval_max_samples", 0))
        if limit > 0:
            indices = indices[:limit]
        return indices

    def _evaluate_split(
        self,
        pl_module: torch.nn.Module,
        split: str,
        dataset,
    ) -> Tuple[int, int]:
        indices = self._prepare_indices(split, len(dataset))
        if not indices:
            raise ValueError(f"MMLU evaluation selected zero samples for split '{split}'")

        device = next(pl_module.parameters()).device
        correct = 0
        total = 0
        for idx in indices:
            sample = dataset[int(idx)]
            question = sample["question"]
            choices = list(sample["choices"])
            subject = sample["subject"]
            answer = int(sample["answer"])

            prompt = (
                _TEMPLATE
                .replace("<Q>", question)
                .replace("<|A|>", choices[0])
                .replace("<|B|>", choices[1])
                .replace("<|C|>", choices[2])
                .replace("<|D|>", choices[3])
                .replace("<SUBJECT>", subject.replace("_", " "))
            )
            prefix_ids = [0] + self.tokenizer.encode(prompt.replace("\r\n", "\n").strip())
            max_ctx = int(self.args.ctx_len)
            if len(prefix_ids) > max_ctx:
                prefix_ids = prefix_ids[-max_ctx:]

            inputs = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)
            inputs = _pad_to_multiple(inputs)
            logits = pl_module(inputs)
            logits = logits[:, : inputs.size(1), :]
            last_logits = logits[0, inputs.size(1) - 1]
            log_probs = F.log_softmax(last_logits, dim=-1)
            choice_scores = log_probs[self.choice_token_ids]
            pred = int(torch.argmax(choice_scores).item())
            if pred == answer:
                correct += 1
            total += 1

        return correct, total

    def evaluate(self, pl_module: torch.nn.Module) -> _EvalResult:
        was_training = pl_module.training
        pl_module.eval()

        scalars: Dict[str, float] = {}
        details: Dict[str, Dict[str, int]] = {}
        with torch.no_grad():
            for split, dataset in self.splits.items():
                correct, total = self._evaluate_split(pl_module, split, dataset)
                accuracy = correct / total if total else 0.0
                scalars[f"mmlu/{split}_accuracy"] = accuracy
                details[split] = {"correct": correct, "total": total}

        if was_training:
            pl_module.train()

        return _EvalResult(
            scalars=scalars,
            details=details,
            label="mmlu",
        )
