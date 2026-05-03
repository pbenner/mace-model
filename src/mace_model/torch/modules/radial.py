from __future__ import annotations

import torch

from mace_model.core.modules import radial as radial_core

from . import _backend_ops as backend_ops


def _make_radial_mlp_net(channels, *, rngs=None):
    del rngs
    return torch.nn.Sequential(
        *radial_core.make_radial_mlp_modules(TORCH_RADIAL_BACKEND, channels)
    )


TORCH_RADIAL_BACKEND = radial_core.RadialBackend(
    name='torch',
    **backend_ops.radial_backend_kwargs(),
    make_radial_mlp_net=_make_radial_mlp_net,
)


@radial_core.use_radial_backend(TORCH_RADIAL_BACKEND)
class BesselBasis(radial_core.BesselBasis, torch.nn.Module):
    """Equation (7)."""

    def __init__(self, r_max: float, num_basis: int = 8, trainable: bool = False):
        torch.nn.Module.__init__(self)
        self.init(r_max=r_max, num_basis=num_basis, trainable=trainable)

    forward = radial_core.BesselBasis.forward
    __repr__ = radial_core.BesselBasis.__repr__


@radial_core.use_radial_backend(TORCH_RADIAL_BACKEND)
class ChebychevBasis(radial_core.ChebychevBasis, torch.nn.Module):
    """Equation (7)."""

    def __init__(self, r_max: float, num_basis: int = 8):
        torch.nn.Module.__init__(self)
        self.init(r_max=r_max, num_basis=num_basis)

    forward = radial_core.ChebychevBasis.forward
    __repr__ = radial_core.ChebychevBasis.__repr__


@radial_core.use_radial_backend(TORCH_RADIAL_BACKEND)
class GaussianBasis(radial_core.GaussianBasis, torch.nn.Module):
    """Gaussian basis functions."""

    def __init__(self, r_max: float, num_basis: int = 128, trainable: bool = False):
        torch.nn.Module.__init__(self)
        self.init(r_max=r_max, num_basis=num_basis, trainable=trainable)

    forward = radial_core.GaussianBasis.forward


@radial_core.use_radial_backend(TORCH_RADIAL_BACKEND)
class PolynomialCutoff(radial_core.PolynomialCutoff, torch.nn.Module):
    """Polynomial cutoff function that goes from 1 to 0 as x approaches r_max."""

    p: torch.Tensor
    r_max: torch.Tensor

    def __init__(self, r_max: float, p: int = 6):
        torch.nn.Module.__init__(self)
        self.init(r_max=r_max, p=p)

    forward = radial_core.PolynomialCutoff.forward
    calculate_envelope = staticmethod(radial_core.PolynomialCutoff.calculate_envelope)
    __repr__ = radial_core.PolynomialCutoff.__repr__


@radial_core.use_radial_backend(TORCH_RADIAL_BACKEND)
class ZBLBasis(radial_core.ZBLBasis, torch.nn.Module):
    """Ziegler-Biersack-Littmark potential with a polynomial cutoff envelope."""

    p: torch.Tensor

    def __init__(self, p: int = 6, trainable: bool = False, **kwargs):
        torch.nn.Module.__init__(self)
        self.init(p=p, trainable=trainable, **kwargs)

    forward = radial_core.ZBLBasis.forward
    __repr__ = radial_core.ZBLBasis.__repr__


@radial_core.use_radial_backend(TORCH_RADIAL_BACKEND)
class AgnesiTransform(radial_core.AgnesiTransform, torch.nn.Module):
    """Agnesi transform used for radial scaling."""

    def __init__(
        self,
        q: float = radial_core.AGNESI_Q,
        p: float = radial_core.AGNESI_P,
        a: float = radial_core.AGNESI_A,
        trainable: bool = False,
    ):
        torch.nn.Module.__init__(self)
        self.init(q=q, p=p, a=a, trainable=trainable)

    forward = radial_core.AgnesiTransform.forward
    __repr__ = radial_core.AgnesiTransform.__repr__


@radial_core.use_radial_backend(TORCH_RADIAL_BACKEND)
class SoftTransform(radial_core.SoftTransform, torch.nn.Module):
    """Tanh-based smooth radial transformation."""

    def __init__(self, alpha: float = radial_core.SOFT_ALPHA, trainable: bool = False):
        torch.nn.Module.__init__(self)
        self.init(alpha=alpha, trainable=trainable)

    compute_r_0 = radial_core.SoftTransform.compute_r_0
    forward = radial_core.SoftTransform.forward
    __repr__ = radial_core.SoftTransform.__repr__


@radial_core.use_radial_backend(TORCH_RADIAL_BACKEND)
class RadialMLP(radial_core.RadialMLP, torch.nn.Module):
    """Construct a radial MLP (Linear -> LayerNorm -> SiLU) stack."""

    def __init__(self, channels_list) -> None:
        torch.nn.Module.__init__(self)
        self.init(channels_list)

    forward = radial_core.RadialMLP.forward


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
