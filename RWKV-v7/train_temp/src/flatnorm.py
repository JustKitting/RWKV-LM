########################################################################################################
# Flat-space normalization utilities for RWKV
########################################################################################################

from numbers import Integral
from typing import Dict, Tuple

import torch
import torch.nn as nn


_LUT_CACHE: Dict[Tuple[str, str, int], torch.Tensor] = {}


@torch.jit.unused
def _build_lut(dtype_key, device: torch.device) -> torch.Tensor:
    """Precompute sign(x) * log1p(|x|) for every representable 16-bit float."""

    clamp = 87.5
    dtype_name: str

    # Accept torch.dtype, string aliases, or scalar-type integers (e.g. 15 for bfloat16).
    if isinstance(dtype_key, torch.dtype):
        dtype = dtype_key
        dtype_name = str(dtype)
    elif isinstance(dtype_key, str):
        key_lower = dtype_key.lower()
        if "bfloat16" in key_lower or key_lower.endswith("bf16"):
            dtype = torch.bfloat16
            dtype_name = "torch.bfloat16"
        elif "float16" in key_lower or key_lower.endswith("fp16") or "half" in key_lower:
            dtype = torch.float16
            dtype_name = "torch.float16"
        else:
            raise ValueError(f"Unsupported dtype for LUT projection: {dtype_key}")
    elif isinstance(dtype_key, Integral):
        if dtype_key == 15:  # c10::ScalarType::BFloat16
            dtype = torch.bfloat16
            dtype_name = "torch.bfloat16"
        elif dtype_key == 5:  # c10::ScalarType::Half
            dtype = torch.float16
            dtype_name = "torch.float16"
        else:
            raise ValueError(f"Unsupported dtype for LUT projection: {dtype_key}")
    else:
        raise TypeError(f"Unsupported dtype key type: {type(dtype_key)!r}")

    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Unsupported dtype for LUT projection: {dtype_key}")

    key = (dtype_name, device.type, device.index if device.type == "cuda" else -1)
    if key in _LUT_CACHE:
        return _LUT_CACHE[key]

    # torch.arange does not support uint16 on CUDA, so build on CPU first.
    values = torch.arange(0, 1 << 16, dtype=torch.int32, device='cpu')
    values = values.to(torch.uint16)
    floats = values.view(dtype).to(torch.float32)

    abs_f = floats.abs()
    mapped = torch.log1p(abs_f)
    if clamp is not None:
        mapped = torch.minimum(mapped, torch.full_like(mapped, clamp))
    mapped = torch.sign(floats) * mapped

    mapped[torch.isnan(floats) | torch.isinf(floats)] = 0.0
    mapped = mapped.to(device)
    _LUT_CACHE[key] = mapped
    return mapped


def _canonical_dtype_key(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "torch.float16"
    if dtype == torch.bfloat16:
        return "torch.bfloat16"
    return str(dtype)


def _should_use_lut(x: torch.Tensor) -> bool:
    """Check whether LUT projection is safe to use in the current execution mode."""

    if x.dtype not in (torch.float16, torch.bfloat16):
        return False

    # TorchScript has trouble with dtype arguments; fall back to analytic path.
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return False

    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "is_compiling"):
        if bool(dynamo.is_compiling()):  # type: ignore[attr-defined]
            return False

    return True


class FlatSpaceGroupNorm(torch.nn.Module):
    """GroupNorm operating in log-space with running geometric centers."""

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-6,
        momentum: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.register_buffer("running_log_mean", torch.zeros(num_channels))

    def _update_running_mean(self, batch_mean: torch.Tensor) -> None:
        if not self.training:
            return
        momentum = self.momentum
        self.running_log_mean.mul_(1 - momentum).add_(momentum * batch_mean)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        C = x.shape[-1]
        assert C == self.num_channels

        x_float = x.to(torch.float32)
        sign = torch.sign(x_float)
        log_abs = torch.log2(x_float.abs().clamp_min(self.eps))

        reduce_dims = list(range(log_abs.dim() - 1))
        batch_mean = log_abs.mean(dim=reduce_dims)
        self._update_running_mean(batch_mean.detach())
        center = batch_mean if self.training else self.running_log_mean
        center_view = center
        expand_steps = log_abs.dim() - center.dim()
        for _ in range(expand_steps):
            center_view = center_view.unsqueeze(0)
        center_view = center_view.expand_as(log_abs)

        log_rel = log_abs - center_view
        group_shape = (-1, self.num_groups, C // self.num_groups)
        log_rel_group = log_rel.reshape(group_shape)

        mean = log_rel_group.mean(dim=-1, keepdim=True)
        var = log_rel_group.var(dim=-1, keepdim=True, unbiased=False)
        norm = (log_rel_group - mean) / torch.sqrt(var + self.eps)
        norm = norm.reshape_as(log_rel)

        weight = self.weight.float()
        bias = self.bias.float()
        weight_view = weight
        bias_view = bias
        expand_steps_w = log_rel.dim() - weight.dim()
        for _ in range(expand_steps_w):
            weight_view = weight_view.unsqueeze(0)
            bias_view = bias_view.unsqueeze(0)
        weight_view = weight_view.expand_as(norm)
        bias_view = bias_view.expand_as(norm)
        out = (norm * weight_view + bias_view) * sign
        return out.to(orig_dtype)


class _LUTProject(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
        # Store original tensor for backward.
        ctx.save_for_backward(x)
        ctx.dtype = x.dtype
        ctx.is_lut = x.dtype in (torch.float16, torch.bfloat16)

        if not ctx.is_lut:
            return torch.sign(x) * torch.log1p(x.abs())

        x_uint = x.contiguous().view(torch.uint16)
        out = lut[x_uint.long()]
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        grad = grad_output / (1.0 + x.abs().to(grad_output.dtype))
        return grad, None


class FlatSpaceNorm(nn.Module):
    """LayerNorm-style normalization performed in log space around a running center."""

    def __init__(self, dim: int, eps: float = 1e-6, momentum: float = 0.01):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer("running_log_mean", torch.zeros(dim))

    def _update_running_mean(self, batch_mean: torch.Tensor) -> None:
        if not self.training:
            return
        momentum = self.momentum
        self.running_log_mean.mul_(1 - momentum).add_(momentum * batch_mean)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_float = x.to(torch.float32)
        sign = torch.sign(x_float)
        log_abs = torch.log2(x_float.abs().clamp_min(self.eps))

        reduce_dims = list(range(log_abs.dim() - 1))
        batch_mean = log_abs.mean(dim=reduce_dims)
        self._update_running_mean(batch_mean.detach())
        center = batch_mean if self.training else self.running_log_mean
        center_view = center
        expand_steps = log_abs.dim() - center.dim()
        for _ in range(expand_steps):
            center_view = center_view.unsqueeze(0)
        center_view = center_view.expand_as(log_abs)

        log_rel = log_abs - center_view
        mean = log_rel.mean(dim=-1, keepdim=True)
        var = log_rel.var(dim=-1, keepdim=True, unbiased=False)
        norm = (log_rel - mean) / torch.sqrt(var + self.eps)

        weight = self.weight.float()
        bias = self.bias.float()
        weight_view = weight
        bias_view = bias
        expand_steps_w = norm.dim() - weight.dim()
        for _ in range(expand_steps_w):
            weight_view = weight_view.unsqueeze(0)
            bias_view = bias_view.unsqueeze(0)
        weight_view = weight_view.expand_as(norm)
        bias_view = bias_view.expand_as(norm)
        out = (norm * weight_view + bias_view) * sign
        return out.to(orig_dtype)
