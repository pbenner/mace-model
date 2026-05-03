from __future__ import annotations

from collections.abc import Sequence

from flax import nnx

from mace_model.core.modules import radial as radial_core

from ..adapters.nnx.torch import nxx_auto_import_from_torch
from . import _backend_ops as backend_ops


def _make_radial_mlp_net(channels, *, rngs: nnx.Rngs):
    return _RadialSequential(channels, rngs=rngs)


JAX_RADIAL_BACKEND = radial_core.RadialBackend(
    name='jax',
    **backend_ops.radial_backend_kwargs(),
    make_radial_mlp_net=_make_radial_mlp_net,
    requires_rng_for_trainable=True,
)


@nxx_auto_import_from_torch(allow_missing_mapper=True)
@radial_core.use_radial_backend(JAX_RADIAL_BACKEND)
class BesselBasis(radial_core.BesselBasis, nnx.Module):
    """Flax implementation of the Bessel basis from MACE."""

    r_max: float
    num_basis: int = 8
    trainable: bool = False

    def __init__(
        self,
        r_max: float,
        num_basis: int = 8,
        trainable: bool = False,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        self.init(r_max=r_max, num_basis=num_basis, trainable=trainable, rngs=rngs)

    __call__ = radial_core.BesselBasis.forward
    __repr__ = radial_core.BesselBasis.__repr__


@nxx_auto_import_from_torch(allow_missing_mapper=True)
@radial_core.use_radial_backend(JAX_RADIAL_BACKEND)
class ChebychevBasis(radial_core.ChebychevBasis, nnx.Module):
    """Flax implementation of the Chebyshev polynomial basis."""

    r_max: float
    num_basis: int = 8

    def __init__(self, r_max: float, num_basis: int = 8) -> None:
        self.init(r_max=r_max, num_basis=num_basis)

    __call__ = radial_core.ChebychevBasis.forward
    __repr__ = radial_core.ChebychevBasis.__repr__


@nxx_auto_import_from_torch(allow_missing_mapper=True)
@radial_core.use_radial_backend(JAX_RADIAL_BACKEND)
class GaussianBasis(radial_core.GaussianBasis, nnx.Module):
    """Gaussian radial basis functions."""

    r_max: float
    num_basis: int = 128
    trainable: bool = False

    def __init__(
        self,
        r_max: float,
        num_basis: int = 128,
        trainable: bool = False,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        self.init(r_max=r_max, num_basis=num_basis, trainable=trainable, rngs=rngs)

    __call__ = radial_core.GaussianBasis.forward


@radial_core.use_radial_backend(JAX_RADIAL_BACKEND)
class PolynomialCutoff(radial_core.PolynomialCutoff, nnx.Module):
    """Polynomial cutoff function that goes from 1 to 0 as r approaches r_max."""

    r_max: float
    p: int = 6

    def __init__(self, r_max: float, p: int = 6) -> None:
        self.init(r_max=r_max, p=p)

    __call__ = radial_core.PolynomialCutoff.forward
    calculate_envelope = staticmethod(radial_core.PolynomialCutoff.calculate_envelope)
    __repr__ = radial_core.PolynomialCutoff.__repr__


@nxx_auto_import_from_torch(allow_missing_mapper=True)
@radial_core.use_radial_backend(JAX_RADIAL_BACKEND)
class ZBLBasis(radial_core.ZBLBasis, nnx.Module):
    """Ziegler-Biersack-Littmark (ZBL) potential with polynomial cutoff."""

    p: int = 6
    trainable: bool = False
    r_max: float | None = None

    def __init__(
        self,
        p: int = 6,
        trainable: bool = False,
        r_max: float | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        self.init(p=p, trainable=trainable, r_max=r_max, rngs=rngs)

    __call__ = radial_core.ZBLBasis.forward
    __repr__ = radial_core.ZBLBasis.__repr__


@nxx_auto_import_from_torch(allow_missing_mapper=True)
@radial_core.use_radial_backend(JAX_RADIAL_BACKEND)
class AgnesiTransform(radial_core.AgnesiTransform, nnx.Module):
    """Agnesi transform used for radial scaling."""

    q: float = radial_core.AGNESI_Q
    p: float = radial_core.AGNESI_P
    a: float = radial_core.AGNESI_A
    trainable: bool = False

    def __init__(
        self,
        q: float = radial_core.AGNESI_Q,
        p: float = radial_core.AGNESI_P,
        a: float = radial_core.AGNESI_A,
        trainable: bool = False,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        self.init(q=q, p=p, a=a, trainable=trainable, rngs=rngs)

    __call__ = radial_core.AgnesiTransform.forward
    __repr__ = radial_core.AgnesiTransform.__repr__


@nxx_auto_import_from_torch(allow_missing_mapper=True)
@radial_core.use_radial_backend(JAX_RADIAL_BACKEND)
class SoftTransform(radial_core.SoftTransform, nnx.Module):
    """Soft transform with a learnable alpha parameter."""

    alpha: float = radial_core.SOFT_ALPHA
    trainable: bool = False

    def __init__(
        self,
        alpha: float = radial_core.SOFT_ALPHA,
        trainable: bool = False,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        self.init(alpha=alpha, trainable=trainable, rngs=rngs)

    compute_r_0 = radial_core.SoftTransform.compute_r_0
    __call__ = radial_core.SoftTransform.forward
    __repr__ = radial_core.SoftTransform.__repr__


@nxx_auto_import_from_torch(allow_missing_mapper=True)
@radial_core.use_radial_backend(JAX_RADIAL_BACKEND)
class _RadialSequential(radial_core.RadialSequential, nnx.Module):
    channels: Sequence[int]

    def __init__(self, channels: Sequence[int], *, rngs: nnx.Rngs) -> None:
        self.init(channels, rngs=rngs)

    __call__ = radial_core.RadialSequential.forward


@nxx_auto_import_from_torch(allow_missing_mapper=True)
@radial_core.use_radial_backend(JAX_RADIAL_BACKEND)
class RadialMLP(radial_core.RadialMLP, nnx.Module):
    """Wrapper that aligns with the Torch RadialMLP parameter layout."""

    channels: Sequence[int]

    def __init__(self, channels: Sequence[int], *, rngs: nnx.Rngs) -> None:
        self.init(channels, rngs=rngs)

    __call__ = radial_core.RadialMLP.forward


__all__ = [
    'AgnesiTransform',
    'BesselBasis',
    'ChebychevBasis',
    'GaussianBasis',
    'PolynomialCutoff',
    'RadialMLP',
    'SoftTransform',
    'ZBLBasis',
]
