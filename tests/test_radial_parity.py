from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx

try:
    import cuequivariance_jax  # noqa: F401
except Exception as exc:  # pragma: no cover - environment dependent
    pytest.skip(
        f'cuequivariance_jax is unavailable in this environment: {exc}',
        allow_module_level=True,
    )

from mace_model.torch.modules.radial import AgnesiTransform as TorchAgnesiTransform
from mace_model.torch.modules.radial import BesselBasis as TorchBesselBasis
from mace_model.torch.modules.radial import ChebychevBasis as TorchChebychevBasis
from mace_model.torch.modules.radial import GaussianBasis as TorchGaussianBasis
from mace_model.torch.modules.radial import PolynomialCutoff as TorchPolynomialCutoff
from mace_model.torch.modules.radial import RadialMLP as TorchRadialMLP
from mace_model.torch.modules.radial import SoftTransform as TorchSoftTransform
from mace_model.torch.modules.radial import ZBLBasis as TorchZBLBasis

_LOCAL_JAX_RADIAL = (
    Path(__file__).resolve().parents[1]
    / 'src'
    / 'mace_model'
    / 'jax'
    / 'modules'
    / 'radial.py'
)
_LOCAL_JAX_ADAPTER = (
    Path(__file__).resolve().parents[1]
    / 'src'
    / 'mace_model'
    / 'jax'
    / 'adapters'
    / 'nnx'
    / 'torch.py'
)
_LOCAL_JAX_ROOT = _LOCAL_JAX_RADIAL.parent.parent
_LOCAL_JAX_MODULES = _LOCAL_JAX_RADIAL.parent
_ALIAS_ROOT = 'mace_local_jax_radial'
_ALIAS_MODULES = f'{_ALIAS_ROOT}.modules'


def _load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Failed to load local module {name} from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if _ALIAS_ROOT not in sys.modules:
    root_pkg = types.ModuleType(_ALIAS_ROOT)
    root_pkg.__path__ = [str(_LOCAL_JAX_ROOT)]  # type: ignore[attr-defined]
    sys.modules[_ALIAS_ROOT] = root_pkg

if _ALIAS_MODULES not in sys.modules:
    modules_pkg = types.ModuleType(_ALIAS_MODULES)
    modules_pkg.__path__ = [str(_LOCAL_JAX_MODULES)]  # type: ignore[attr-defined]
    sys.modules[_ALIAS_MODULES] = modules_pkg

_LOCAL_JAX_MODULE = _load_local_module(f'{_ALIAS_MODULES}.radial', _LOCAL_JAX_RADIAL)
_LOCAL_ADAPTER_MODULE = _load_local_module(
    f'{_ALIAS_ROOT}.nnx_torch_adapter',
    _LOCAL_JAX_ADAPTER,
)
init_from_torch = _LOCAL_ADAPTER_MODULE.init_from_torch

JaxBesselBasis = _LOCAL_JAX_MODULE.BesselBasis
JaxChebychevBasis = _LOCAL_JAX_MODULE.ChebychevBasis
JaxGaussianBasis = _LOCAL_JAX_MODULE.GaussianBasis
JaxAgnesiTransform = _LOCAL_JAX_MODULE.AgnesiTransform
JaxPolynomialCutoff = _LOCAL_JAX_MODULE.PolynomialCutoff
JaxRadialMLP = _LOCAL_JAX_MODULE.RadialMLP
JaxSoftTransform = _LOCAL_JAX_MODULE.SoftTransform
JaxZBLBasis = _LOCAL_JAX_MODULE.ZBLBasis


def _to_numpy(value):
    return np.asarray(value.array if hasattr(value, 'array') else value)


def test_torch_and_jax_bessel_match_after_weight_transfer():
    rng = np.random.default_rng(2)
    torch_model = TorchBesselBasis(r_max=5.0, num_basis=6, trainable=True).float()
    jax_model = JaxBesselBasis(
        r_max=5.0,
        num_basis=6,
        trainable=True,
        rngs=nnx.Rngs(0),
    )
    jax_model, _ = init_from_torch(jax_model, torch_model)

    x_np = rng.uniform(0.1, 4.9, size=(8, 1)).astype(np.float32)
    x_np[0, 0] = 0.0
    out_torch = torch_model(torch.tensor(x_np))
    graphdef, state = nnx.split(jax_model)
    out_jax, _ = graphdef.apply(state)(jnp.asarray(x_np))
    np.testing.assert_allclose(
        _to_numpy(out_jax),
        out_torch.detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_torch_and_jax_radial_scalar_formulas_match():
    x_np = np.linspace(0.2, 2.6, 8, dtype=np.float32)[:, None]
    node_attrs_np = np.eye(3, dtype=np.float32)[np.array([0, 1, 2, 0, 1])]
    edge_index_np = np.array(
        [[0, 1, 2, 3, 4, 0, 1, 2], [1, 2, 3, 4, 0, 2, 3, 4]],
        dtype=np.int32,
    )
    atomic_numbers_np = np.array([1, 6, 8], dtype=np.int32)
    node_attrs_index_np = np.argmax(node_attrs_np, axis=1).astype(np.int32)

    torch_x = torch.tensor(x_np)
    jax_x = jnp.asarray(x_np)

    formula_cases = [
        (
            TorchChebychevBasis(r_max=3.0, num_basis=6).float(),
            JaxChebychevBasis(r_max=3.0, num_basis=6),
            (torch_x,),
            (jax_x,),
        ),
        (
            TorchGaussianBasis(r_max=3.0, num_basis=6).float(),
            JaxGaussianBasis(r_max=3.0, num_basis=6),
            (torch_x,),
            (jax_x,),
        ),
        (
            TorchPolynomialCutoff(r_max=3.0, p=5).float(),
            JaxPolynomialCutoff(r_max=3.0, p=5),
            (torch_x,),
            (jax_x,),
        ),
        (
            TorchAgnesiTransform().float(),
            JaxAgnesiTransform(),
            (
                torch_x,
                torch.tensor(node_attrs_np),
                torch.tensor(edge_index_np, dtype=torch.int64),
                torch.tensor(atomic_numbers_np, dtype=torch.int64),
            ),
            (
                jax_x,
                jnp.asarray(node_attrs_np),
                jnp.asarray(edge_index_np, dtype=jnp.int32),
                jnp.asarray(atomic_numbers_np, dtype=jnp.int32),
                jnp.asarray(node_attrs_index_np, dtype=jnp.int32),
            ),
        ),
        (
            TorchSoftTransform().float(),
            JaxSoftTransform(),
            (
                torch_x,
                torch.tensor(node_attrs_np),
                torch.tensor(edge_index_np, dtype=torch.int64),
                torch.tensor(atomic_numbers_np, dtype=torch.int64),
            ),
            (
                jax_x,
                jnp.asarray(node_attrs_np),
                jnp.asarray(edge_index_np, dtype=jnp.int32),
                jnp.asarray(atomic_numbers_np, dtype=jnp.int32),
                jnp.asarray(node_attrs_index_np, dtype=jnp.int32),
            ),
        ),
    ]

    for torch_model, jax_model, torch_args, jax_args in formula_cases:
        out_torch = torch_model(*torch_args)
        graphdef, state = nnx.split(jax_model)
        out_jax, _ = graphdef.apply(state)(*jax_args)
        np.testing.assert_allclose(
            _to_numpy(out_jax),
            out_torch.detach().cpu().numpy(),
            rtol=1e-6,
            atol=1e-6,
        )


def test_radial_core_validation_is_shared_by_backend_wrappers():
    with pytest.raises(ValueError, match='BesselBasis requires num_basis >= 1'):
        TorchBesselBasis(r_max=5.0, num_basis=0)
    with pytest.raises(ValueError, match='ChebychevBasis requires num_basis >= 1'):
        JaxChebychevBasis(r_max=5.0, num_basis=0)
    with pytest.raises(ValueError, match='GaussianBasis requires num_basis >= 2'):
        TorchGaussianBasis(r_max=5.0, num_basis=1)
    with pytest.raises(ValueError, match='channels must have length >= 2'):
        TorchRadialMLP([4])


def test_torch_and_jax_radial_mlp_match_after_weight_transfer():
    rng = np.random.default_rng(5)
    channels = [4, 8, 16]
    torch_model = TorchRadialMLP(channels).float()
    jax_model = JaxRadialMLP(channels, rngs=nnx.Rngs(0))
    jax_model, _ = init_from_torch(jax_model, torch_model)

    x_np = rng.normal(size=(7, channels[0])).astype(np.float32)
    out_torch = torch_model(torch.tensor(x_np))
    graphdef, state = nnx.split(jax_model)
    out_jax, _ = graphdef.apply(state)(jnp.asarray(x_np))
    np.testing.assert_allclose(
        _to_numpy(out_jax),
        out_torch.detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_torch_and_jax_zbl_match_after_weight_transfer():
    x_np = np.linspace(0.4, 2.0, 8, dtype=np.float32)[:, None]
    node_attrs_np = np.eye(3, dtype=np.float32)[np.array([0, 1, 2, 0, 1])]
    edge_index_np = np.array(
        [[0, 1, 2, 3, 4, 0, 1, 2], [1, 2, 3, 4, 0, 2, 3, 4]],
        dtype=np.int32,
    )
    atomic_numbers_np = np.array([1, 6, 8], dtype=np.int32)

    torch_model = TorchZBLBasis(p=6).float()
    jax_model = JaxZBLBasis(p=6, rngs=nnx.Rngs(0))
    jax_model, _ = init_from_torch(jax_model, torch_model)

    out_torch = torch_model(
        torch.tensor(x_np),
        torch.tensor(node_attrs_np),
        torch.tensor(edge_index_np),
        torch.tensor(atomic_numbers_np),
    )
    graphdef, state = nnx.split(jax_model)
    out_jax, _ = graphdef.apply(state)(
        jnp.asarray(x_np),
        jnp.asarray(node_attrs_np),
        jnp.asarray(edge_index_np),
        jnp.asarray(atomic_numbers_np),
    )
    np.testing.assert_allclose(
        _to_numpy(out_jax),
        out_torch.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
