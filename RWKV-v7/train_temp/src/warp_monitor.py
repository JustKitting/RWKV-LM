"""Utilities for tracking floating-point curvature metrics during training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

import torch


def _tensor_summary(x: torch.Tensor, eps: float = 1e-12, max_sample: int = 4096) -> dict:
    x_detached = x.detach()
    dtype = str(x_detached.dtype)
    device = str(x_detached.device)
    numel = x_detached.numel()
    if numel == 0:
        return {
            "dtype": dtype,
            "device": device,
            "numel": 0,
            "mean_log2_abs": None,
            "std_log2_abs": None,
            "mean_abs": None,
            "max_abs": None,
            "ulp_mean": None,
        }

    if numel > max_sample:
        idx = torch.randperm(numel, device=x_detached.device)[:max_sample]
        x_detached = x_detached.reshape(-1)[idx]

    x_fp32 = x_detached.to(torch.float32)
    abs_x = x_fp32.abs().clamp_min(eps)
    log_abs = torch.log2(abs_x)
    mean_log = log_abs.mean().item()
    std_log = log_abs.std(unbiased=False).item()
    mean_abs = abs_x.mean().item()
    max_abs = abs_x.max().item()

    finfo = torch.finfo(x_detached.dtype)
    ulp_mean = finfo.eps * mean_abs

    return {
        "dtype": dtype,
        "device": device,
        "numel": int(numel),
        "mean_log2_abs": mean_log,
        "std_log2_abs": std_log,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "ulp_mean": ulp_mean,
    }


class WarpMonitor:
    """Collects per-operation curvature metrics and writes them to disk."""

    def __init__(
        self,
        output_file: Path,
        flush_every: int = 1024,
    ) -> None:
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self._records: List[dict] = []
        self._pending: dict[int, dict[str, float]] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._counter = 0
        self._step = 0

    def set_step(self, step: int) -> None:
        self._step = int(step)

    def reset(self) -> None:
        self._counter = 0
        self._records.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.flush()

    def flush(self) -> None:
        if not self._records:
            return
        with self.output_file.open("a", encoding="utf-8") as f:
            for rec in self._records:
                f.write(json.dumps(rec) + "\n")
        self._records.clear()

    def _record_tensor(self, name: str, tensor: torch.Tensor, phase: str) -> None:
        if not torch.is_floating_point(tensor):
            return
        summary = _tensor_summary(tensor)
        summary.update({
            "op_index": self._counter,
            "step": self._step,
            "name": name,
            "phase": phase,
        })
        self._counter += 1
        self._records.append(summary)
        metrics = {}
        for metric in ("mean_log2_abs", "std_log2_abs", "mean_abs", "max_abs", "ulp_mean"):
            value = summary.get(metric)
            if value is not None:
                metrics[f"warp/{name}/{phase}/{metric}"] = value
        if metrics:
            step = summary["step"]
            if step not in self._pending:
                self._pending[step] = {}
            self._pending[step].update(metrics)
        if len(self._records) >= self.flush_every:
            self.flush()

    def attach_module(self, module: torch.nn.Module, name: str) -> None:
        def hook(_, __, output):
            if torch.is_tensor(output):
                self._record_tensor(name, output, phase="hook")
            elif isinstance(output, (tuple, list)):
                for idx, item in enumerate(output):
                    if torch.is_tensor(item):
                        self._record_tensor(f"{name}[{idx}]", item, phase="hook")
        handle = module.register_forward_hook(hook)
        self._handles.append(handle)

    def install(self, model: torch.nn.Module, module_names: Optional[Iterable[str]] = None) -> None:
        if module_names is None:
            module_names = [
                "emb",
                "ln_out",
            ]
            for idx, block in enumerate(getattr(model, "blocks", [])):
                module_names.extend([
                    f"blocks.{idx}.ln0" if hasattr(block, "ln0") else None,
                    f"blocks.{idx}.ln1",
                    f"blocks.{idx}.ln2",
                    f"blocks.{idx}.att",
                    f"blocks.{idx}.ffn",
                ])
            module_names = [n for n in module_names if n is not None]

        for name in module_names:
            module = None
            if name:
                if not hasattr(self, "_module_cache"):
                    self._module_cache = dict(model.named_modules())
                module = self._module_cache.get(name)
            if module is None:
                continue
            self.attach_module(module, name)

    def record_pre(self, name: str, tensor: torch.Tensor) -> None:
        self._record_tensor(name, tensor, phase="pre")

    def record_post(self, name: str, tensor: torch.Tensor) -> None:
        self._record_tensor(name, tensor, phase="post")

    def pop_metrics(self, step: int) -> dict[str, float]:
        return dict(sorted(self._pending.pop(step, {}).items()))
