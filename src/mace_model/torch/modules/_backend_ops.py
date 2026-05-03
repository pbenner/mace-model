"""Shared low-level Torch operations used by module backends and radial wrappers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from ..tools.scatter import scatter_sum as torch_scatter_sum


def default_dtype():
    return torch.get_default_dtype()


def int_dtype():
    return torch.int


def asarray(value: Any, dtype: Any = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value if dtype is None else value.to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def register_constant(module: torch.nn.Module, name: str, value: Any) -> None:
    module.register_buffer(name, value)


def register_parameter(module: torch.nn.Module, name: str, value: Any) -> None:
    setattr(module, name, torch.nn.Parameter(value))


def value(value: Any, dtype: Any):
    return value if dtype is None else value.to(dtype=dtype)


def to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(()))
    return float(value)


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def argmax_node_attrs(value: torch.Tensor) -> torch.Tensor:
    return torch.argmax(value, dim=1)


def as_index(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.int64)


def concatenate(values: Sequence[Any], axis: int):
    return torch.cat(tuple(values), dim=axis)


def cat(values: Sequence[Any], dim: int = -1):
    return torch.cat(tuple(values), dim=dim)


def stack(values: Sequence[Any], dim: int = 0):
    return torch.stack(tuple(values), dim=dim)


def sum_values(value: Any, dim: int):
    return torch.sum(value, dim=dim)


def squeeze(value: Any, *, axis: int):
    return torch.squeeze(value, dim=axis)


def finfo_eps(dtype: Any):
    return torch.finfo(dtype).eps


def make_linear_layer(*, in_channels: int, out_channels: int, rngs=None):
    del rngs
    return torch.nn.Linear(in_channels, out_channels, bias=True)


def make_norm_layer(*, out_channels: int, rngs=None):
    del rngs
    return torch.nn.LayerNorm(out_channels)


def scatter_sum(src, index, dim=-1, dim_size=None, indices_are_sorted=False):
    del indices_are_sorted
    return torch_scatter_sum(src=src, index=index, dim=dim, dim_size=dim_size)


def make_parameter(module, *, name, value, requires_grad=True):
    tensor = asarray(value, dtype=default_dtype()).clone().detach()
    module.register_parameter(
        name,
        torch.nn.Parameter(tensor, requires_grad=requires_grad),
    )
    return getattr(module, name)


def make_scale_shift(module, *, name, value):
    register_constant(module, name, asarray(value, dtype=default_dtype()))
    return getattr(module, name)


def get_scale_shift(value):
    return value


def make_atomic_energies(module, atomic_energies):
    register_constant(
        module,
        'atomic_energies',
        asarray(atomic_energies, dtype=default_dtype()),
    )
    return module.atomic_energies


def get_atomic_energies(atomic_energies):
    return atomic_energies


def make_ones(*, node_feats, width):
    return torch.ones(
        (node_feats.shape[0], int(width)),
        dtype=node_feats.dtype,
        device=node_feats.device,
    )


def make_index_attrs(*, node_attrs, node_attrs_index):
    del node_attrs_index
    return torch.nonzero(node_attrs)[:, 1].int()


sin = torch.sin
exp = torch.exp
tanh = torch.tanh
abs_fn = torch.abs
where = torch.where
broadcast_to = torch.broadcast_to
ones_like = torch.ones_like
silu = torch.nn.functional.silu
sigmoid = torch.sigmoid
make_zeros = torch.zeros
make_activation_layer = torch.nn.SiLU


def radial_backend_kwargs() -> dict[str, Any]:
    return {
        'default_dtype': default_dtype,
        'int_dtype': int_dtype,
        'asarray': asarray,
        'register_constant': register_constant,
        'register_parameter': register_parameter,
        'value': value,
        'to_float': to_float,
        'argmax_node_attrs': argmax_node_attrs,
        'as_index': as_index,
        'sin': sin,
        'exp': exp,
        'tanh': tanh,
        'abs': abs_fn,
        'where': where,
        'broadcast_to': broadcast_to,
        'finfo_eps': finfo_eps,
        'ones_like': ones_like,
        'concatenate': concatenate,
        'scatter_sum': scatter_sum,
        'squeeze': squeeze,
        'make_linear_layer': make_linear_layer,
        'make_norm_layer': make_norm_layer,
        'make_activation_layer': make_activation_layer,
    }
