"""Shared low-level JAX operations used by module backends and radial wrappers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..tools.dtype import default_dtype as _default_dtype
from ..tools.scatter import scatter_sum as jax_scatter_sum


def default_dtype():
    return _default_dtype()


def asarray(value: Any, dtype: Any = None) -> jnp.ndarray:
    return jnp.asarray(value, dtype=dtype)


def int_dtype():
    return jnp.int32


def register_constant(module: Any, name: str, value: Any) -> None:
    setattr(module, name, value)


def register_parameter(module: Any, name: str, value: Any) -> None:
    setattr(module, name, nnx.Param(value))


def value(value: Any, dtype: Any) -> jnp.ndarray:
    return jnp.asarray(value, dtype=dtype)


def to_float(value: Any) -> float:
    if hasattr(value, 'get_value'):
        value = value.get_value()
    return float(jax.device_get(jnp.asarray(value)))


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, 'get_value'):
        value = value.get_value()
    return np.asarray(jax.device_get(value))


def argmax_node_attrs(value: Any) -> jnp.ndarray:
    return jnp.argmax(value, axis=1)


def as_index(value: Any) -> jnp.ndarray:
    return jnp.asarray(value, dtype=jnp.int32)


def concatenate(values: Sequence[Any], axis: int):
    return jnp.concatenate(tuple(values), axis=axis)


def cat(values: Sequence[Any], dim: int = -1):
    return concatenate(values, axis=dim)


def stack(values: Sequence[Any], dim: int = 0):
    return jnp.stack(tuple(values), axis=dim)


def sum_values(value: Any, dim: int):
    return jnp.sum(value, axis=dim)


def squeeze(value: Any, *, axis: int):
    return jnp.squeeze(value, axis=axis)


def finfo_eps(dtype: Any):
    return jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)


def make_layer_container():
    return nnx.Dict()


def make_linear_layer(*, in_channels: int, out_channels: int, rngs: nnx.Rngs):
    return nnx.Linear(in_channels, out_channels, use_bias=True, rngs=rngs)


def make_norm_layer(*, out_channels: int, rngs: nnx.Rngs):
    return nnx.LayerNorm(
        num_features=out_channels,
        use_bias=True,
        use_scale=True,
        reduction_axes=-1,
        feature_axes=-1,
        epsilon=1e-5,
        rngs=rngs,
    )


def scatter_sum(src, index, dim=-1, dim_size=None, indices_are_sorted=False):
    return jax_scatter_sum(
        src=src,
        index=index,
        dim=dim,
        dim_size=dim_size,
        indices_are_sorted=indices_are_sorted,
    )


def make_parameter(module, *, name, value, requires_grad=True):
    del requires_grad
    param = nnx.Param(jnp.asarray(value, dtype=default_dtype()))
    setattr(module, name, param)
    return param


def make_scale_shift(module, *, name, value):
    del module, name
    return nnx.Param(asarray(value, dtype=default_dtype()))


def get_scale_shift(value):
    return jax.lax.stop_gradient(value)


def make_atomic_energies(module, atomic_energies):
    del module
    return nnx.Param(asarray(atomic_energies, dtype=default_dtype()))


def get_atomic_energies(atomic_energies):
    return jax.lax.stop_gradient(atomic_energies)


def make_ones(*, node_feats, width):
    return jnp.ones((node_feats.shape[0], int(width)), dtype=node_feats.dtype)


def make_index_attrs(*, node_attrs, node_attrs_index):
    if node_attrs_index is None:
        return jnp.argmax(node_attrs, axis=1).astype(jnp.int32)
    index_attrs = jnp.asarray(node_attrs_index, dtype=jnp.int32)
    if index_attrs.ndim != 1:
        return jnp.argmax(node_attrs, axis=1).astype(jnp.int32)
    return index_attrs.reshape(-1)


sin = jnp.sin
exp = jnp.exp
tanh = jnp.tanh
abs_fn = jnp.abs
where = jnp.where
broadcast_to = jnp.broadcast_to
ones_like = jnp.ones_like
silu = jnn.silu
sigmoid = jnn.sigmoid
make_zeros = jnp.zeros


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
        'silu': silu,
        'make_layer_container': make_layer_container,
        'make_linear_layer': make_linear_layer,
        'make_norm_layer': make_norm_layer,
    }
