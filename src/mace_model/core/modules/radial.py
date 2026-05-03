from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import ase
import numpy as np

ZBL_COEFFICIENTS = (0.1818, 0.5099, 0.2802, 0.02817)
ZBL_A_EXP = 0.300
ZBL_A_PREFACTOR = 0.4543
ZBL_BOHR_RADIUS = 0.529
ZBL_ENERGY_FACTOR = 14.3996

AGNESI_Q = 0.9183
AGNESI_P = 4.5791
AGNESI_A = 1.0805

SOFT_ALPHA = 4.0


@dataclass(frozen=True)
class RadialMLPLayerSpec:
    kind: Literal['linear', 'norm', 'act']
    index: int
    in_channels: int | None = None
    out_channels: int | None = None


def validate_edge_aligned_inputs(x: Any, edge_index: Any, *, module_name: str) -> None:
    num_edges = int(edge_index.shape[1])
    num_distances = int(x.shape[0])
    if num_distances != num_edges:
        raise ValueError(
            f'{module_name} expects one distance per edge; received '
            f'{num_distances} distances for {num_edges} edges.'
        )


def validate_num_basis(num_basis: int, *, module_name: str) -> int:
    value = int(num_basis)
    if value < 1:
        raise ValueError(f'{module_name} requires num_basis >= 1, got {value}.')
    return value


def bessel_weight_values(r_max: float, num_basis: int) -> np.ndarray:
    num_basis = validate_num_basis(num_basis, module_name='BesselBasis')
    return (
        np.pi
        / float(r_max)
        * np.linspace(
            1.0,
            float(num_basis),
            num_basis,
        )
    )


def gaussian_weight_values(r_max: float, num_basis: int) -> np.ndarray:
    num_basis = validate_num_basis(num_basis, module_name='GaussianBasis')
    return np.linspace(0.0, float(r_max), num_basis)


def gaussian_coefficient(r_max: float, num_basis: int) -> float:
    num_basis = validate_num_basis(num_basis, module_name='GaussianBasis')
    if num_basis < 2:
        raise ValueError('GaussianBasis requires num_basis >= 2.')
    return -0.5 / (float(r_max) / (num_basis - 1)) ** 2


def covalent_radii_values() -> np.ndarray:
    return np.asarray(ase.data.covalent_radii)


def bessel_basis(
    x: Any,
    bessel_weights: Any,
    prefactor: Any,
    *,
    sin: Callable[[Any], Any],
    abs_fn: Callable[[Any], Any],
    broadcast_to: Callable[[Any, Any], Any],
    finfo_eps: Callable[[Any], Any],
    where: Callable[[Any, Any, Any], Any],
) -> Any:
    numerator = sin(bessel_weights * x)
    near_zero = abs_fn(x) < finfo_eps(x.dtype)
    safe_denominator = where(near_zero, 1.0, x)
    safe_denominator = broadcast_to(safe_denominator, numerator.shape)
    ratio = numerator / safe_denominator
    near_zero = broadcast_to(near_zero, ratio.shape)
    weights = broadcast_to(bessel_weights, ratio.shape)
    return prefactor * where(near_zero, weights, ratio)


def gaussian_basis(
    x: Any,
    gaussian_weights: Any,
    coeff: Any,
    *,
    exp: Callable[[Any], Any],
) -> Any:
    shifted = x - gaussian_weights
    return exp(coeff * (shifted * shifted))


def polynomial_cutoff_envelope(x: Any, r_max: Any, p: Any) -> Any:
    r_over_r_max = x / r_max
    envelope = (
        1.0
        - ((p + 1.0) * (p + 2.0) / 2.0) * (r_over_r_max**p)
        + p * (p + 2.0) * (r_over_r_max ** (p + 1.0))
        - (p * (p + 1.0) / 2.0) * (r_over_r_max ** (p + 2.0))
    )
    return envelope * (x < r_max)


def edge_atomic_numbers(
    node_attrs: Any,
    edge_index: Any,
    atomic_numbers: Any,
    *,
    argmax_node_attrs: Callable[[Any], Any],
    as_index: Callable[[Any], Any],
    node_attrs_index: Any = None,
) -> tuple[Any, Any, Any, Any]:
    sender = edge_index[0]
    receiver = edge_index[1]
    if node_attrs_index is None:
        node_attrs_index = argmax_node_attrs(node_attrs)
    node_atomic_numbers = atomic_numbers[
        as_index(node_attrs_index).reshape(-1)
    ].reshape(-1, 1)
    z_u = as_index(node_atomic_numbers[sender])
    z_v = as_index(node_atomic_numbers[receiver])
    return sender, receiver, z_u, z_v


def pair_covalent_radii(covalent_radii: Any, z_u: Any, z_v: Any) -> Any:
    return covalent_radii[z_u] + covalent_radii[z_v]


def zbl_pair_energy(
    x: Any,
    z_u: Any,
    z_v: Any,
    *,
    coefficients: Any,
    a_exp: Any,
    a_prefactor: Any,
    exp: Callable[[Any], Any],
) -> Any:
    a = a_prefactor * ZBL_BOHR_RADIUS / ((z_u**a_exp) + (z_v**a_exp))
    r_over_a = x / a
    phi = (
        coefficients[0] * exp(-3.2 * r_over_a)
        + coefficients[1] * exp(-0.9423 * r_over_a)
        + coefficients[2] * exp(-0.4028 * r_over_a)
        + coefficients[3] * exp(-0.2016 * r_over_a)
    )
    return (ZBL_ENERGY_FACTOR * z_u * z_v) / x * phi


def agnesi_transform(x: Any, r_0: Any, *, a: Any, q: Any, p: Any) -> Any:
    r_over_r_0 = x / r_0
    numerator = a * (r_over_r_0**q)
    denominator = 1.0 + (r_over_r_0 ** (q - p))
    return 1.0 / (1.0 + numerator / denominator)


def soft_transform(x: Any, r_0: Any, *, alpha: Any, tanh: Callable[[Any], Any]) -> Any:
    p_0 = (3.0 / 4.0) * r_0
    p_1 = (4.0 / 3.0) * r_0
    m = 0.5 * (p_0 + p_1)
    scaled_alpha = alpha / (p_1 - p_0)
    s_x = 0.5 * (1.0 + tanh(scaled_alpha * (x - m)))
    return p_0 + (x - p_0) * s_x


def radial_mlp_layer_plan(channels: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in channels)
    if len(normalized) < 2:
        raise ValueError('channels must have length >= 2 for RadialMLP')
    return normalized


def iter_radial_mlp_layers(channels: Sequence[int]) -> tuple[RadialMLPLayerSpec, ...]:
    normalized = radial_mlp_layer_plan(channels)
    layers: list[RadialMLPLayerSpec] = []
    last_idx = len(normalized) - 1
    layer_idx = 0
    for idx, out_channels in enumerate(normalized[1:], start=1):
        layers.append(
            RadialMLPLayerSpec(
                kind='linear',
                index=layer_idx,
                in_channels=normalized[idx - 1],
                out_channels=out_channels,
            )
        )
        layer_idx += 1
        if idx != last_idx:
            layers.append(
                RadialMLPLayerSpec(
                    kind='norm',
                    index=layer_idx,
                    out_channels=out_channels,
                )
            )
            layer_idx += 1
            layers.append(RadialMLPLayerSpec(kind='act', index=layer_idx))
            layer_idx += 1
    return tuple(layers)


def chebyshev_basis_values(
    x: Any,
    num_basis: int,
    *,
    ones_like: Callable[[Any], Any],
    concatenate: Callable[[Sequence[Any], int], Any],
) -> Any:
    num_basis = validate_num_basis(num_basis, module_name='ChebychevBasis')
    t0 = ones_like(x)
    t1 = x
    terms = [t1]
    for _ in range(2, num_basis + 1):
        t2 = 2 * x * t1 - t0
        terms.append(t2)
        t0, t1 = t1, t2
    return concatenate(terms, -1)


@dataclass(frozen=True)
class RadialBackend:
    name: str
    default_dtype: Callable[[], Any]
    int_dtype: Callable[[], Any]
    asarray: Callable[..., Any]
    register_constant: Callable[[Any, str, Any], None]
    register_parameter: Callable[[Any, str, Any], None]
    value: Callable[[Any, Any], Any]
    to_float: Callable[[Any], float]
    argmax_node_attrs: Callable[[Any], Any]
    as_index: Callable[[Any], Any]
    sin: Callable[[Any], Any]
    exp: Callable[[Any], Any]
    tanh: Callable[[Any], Any]
    abs: Callable[[Any], Any]
    where: Callable[[Any, Any, Any], Any]
    broadcast_to: Callable[[Any, Any], Any]
    finfo_eps: Callable[[Any], Any]
    ones_like: Callable[[Any], Any]
    concatenate: Callable[[Sequence[Any], int], Any]
    scatter_sum: Callable[..., Any] | None = None
    squeeze: Callable[..., Any] | None = None
    silu: Callable[[Any], Any] | None = None
    make_layer_container: Callable[[], Any] | None = None
    make_linear_layer: Callable[..., Any] | None = None
    make_norm_layer: Callable[..., Any] | None = None
    make_activation_layer: Callable[[], Any] | None = None
    make_radial_mlp_net: Callable[..., Any] | None = None
    requires_rng_for_trainable: bool = False

    def require(self, field_name: str) -> Callable[..., Any]:
        fn = getattr(self, field_name, None)
        if fn is None:
            raise NotImplementedError(
                f"RadialBackend '{self.name}' is missing required operation "
                f"'{field_name}'."
            )
        return fn


def use_radial_backend(backend: RadialBackend):
    def decorator(cls):
        cls.BACKEND = backend
        return cls

    return decorator


def _require_backend(instance: Any, class_name: str) -> RadialBackend:
    backend = getattr(instance, 'BACKEND', None)
    if backend is None:
        raise RuntimeError(f'{class_name} requires a class-level BACKEND.')
    return backend


def _register_constant(
    instance: Any,
    name: str,
    value: Any,
    *,
    dtype: Any = None,
) -> None:
    backend = _require_backend(instance, instance.__class__.__name__)
    if dtype is None:
        dtype = backend.default_dtype()
    backend.register_constant(instance, name, backend.asarray(value, dtype=dtype))


def _register_trainable(
    instance: Any,
    name: str,
    value: Any,
    trainable: bool,
    *,
    rngs: Any = None,
    module_name: str,
) -> None:
    backend = _require_backend(instance, module_name)
    if trainable and backend.requires_rng_for_trainable and rngs is None:
        raise ValueError(f'rngs is required for trainable {module_name}')
    array = backend.asarray(value, dtype=backend.default_dtype())
    if trainable:
        backend.register_parameter(instance, name, array)
    else:
        backend.register_constant(instance, name, array)


class BesselBasis:
    BACKEND: RadialBackend

    def init(
        self,
        r_max: float,
        num_basis: int = 8,
        trainable: bool = False,
        *,
        rngs: Any = None,
    ) -> None:
        self.num_basis = validate_num_basis(num_basis, module_name='BesselBasis')
        self.trainable = bool(trainable)
        _register_trainable(
            self,
            'bessel_weights',
            bessel_weight_values(r_max, self.num_basis),
            self.trainable,
            rngs=rngs,
            module_name='BesselBasis',
        )
        _register_constant(self, 'r_max', float(r_max))
        _register_constant(self, 'prefactor', np.sqrt(2.0 / float(r_max)))

    def forward(self, x: Any) -> Any:
        backend = _require_backend(self, 'BesselBasis')
        x = backend.asarray(x)
        dtype = x.dtype
        return bessel_basis(
            x,
            backend.value(self.bessel_weights, dtype),
            backend.value(self.prefactor, dtype),
            sin=backend.sin,
            abs_fn=backend.abs,
            broadcast_to=backend.broadcast_to,
            finfo_eps=backend.finfo_eps,
            where=backend.where,
        )

    def __repr__(self):
        backend = _require_backend(self, 'BesselBasis')
        return (
            f'{self.__class__.__name__}(r_max={backend.to_float(self.r_max)}, '
            f'num_basis={self.num_basis}, trainable={self.trainable})'
        )


class ChebychevBasis:
    BACKEND: RadialBackend

    def init(self, r_max: float, num_basis: int = 8) -> None:
        self.r_max = float(r_max)
        self.num_basis = validate_num_basis(num_basis, module_name='ChebychevBasis')

    def forward(self, x: Any) -> Any:
        backend = _require_backend(self, 'ChebychevBasis')
        return chebyshev_basis_values(
            backend.asarray(x),
            self.num_basis,
            ones_like=backend.ones_like,
            concatenate=backend.concatenate,
        )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(r_max={self.r_max}, num_basis={self.num_basis})'
        )


class GaussianBasis:
    BACKEND: RadialBackend

    def init(
        self,
        r_max: float,
        num_basis: int = 128,
        trainable: bool = False,
        *,
        rngs: Any = None,
    ) -> None:
        self.r_max = float(r_max)
        self.num_basis = validate_num_basis(num_basis, module_name='GaussianBasis')
        self.trainable = bool(trainable)
        self.coeff = gaussian_coefficient(self.r_max, self.num_basis)
        _register_trainable(
            self,
            'gaussian_weights',
            gaussian_weight_values(self.r_max, self.num_basis),
            self.trainable,
            rngs=rngs,
            module_name='GaussianBasis',
        )

    def forward(self, x: Any) -> Any:
        backend = _require_backend(self, 'GaussianBasis')
        x = backend.asarray(x)
        dtype = x.dtype
        return gaussian_basis(
            x,
            backend.value(self.gaussian_weights, dtype),
            backend.asarray(self.coeff, dtype=dtype),
            exp=backend.exp,
        )


class PolynomialCutoff:
    BACKEND: RadialBackend

    def init(self, r_max: float, p: int = 6) -> None:
        backend = _require_backend(self, 'PolynomialCutoff')
        _register_constant(self, 'r_max', float(r_max))
        _register_constant(self, 'p', int(p), dtype=backend.int_dtype())

    def forward(self, x: Any) -> Any:
        backend = _require_backend(self, 'PolynomialCutoff')
        x = backend.asarray(x)
        dtype = x.dtype
        return self.calculate_envelope(
            x,
            backend.value(self.r_max, dtype),
            backend.value(self.p, dtype),
        )

    @staticmethod
    def calculate_envelope(x: Any, r_max: Any, p: Any) -> Any:
        return polynomial_cutoff_envelope(x, r_max, p)

    def __repr__(self):
        backend = _require_backend(self, 'PolynomialCutoff')
        return (
            f'{self.__class__.__name__}(p={backend.to_float(self.p):.0f}, '
            f'r_max={backend.to_float(self.r_max)})'
        )


class ZBLBasis:
    BACKEND: RadialBackend

    def init(
        self,
        p: int = 6,
        trainable: bool = False,
        r_max: float | None = None,
        *,
        rngs: Any = None,
        **kwargs,
    ) -> None:
        if r_max is None and 'r_max' in kwargs:
            r_max = kwargs['r_max']
        self.p_value = int(p)
        self.trainable = bool(trainable)
        self.r_max = r_max
        if self.r_max is not None:
            logging.warning(
                'r_max is deprecated. r_max is determined from the covalent radii.'
            )
        backend = _require_backend(self, 'ZBLBasis')
        _register_constant(self, 'c', ZBL_COEFFICIENTS)
        _register_constant(self, 'p', self.p_value, dtype=backend.int_dtype())
        _register_constant(self, 'covalent_radii', covalent_radii_values())
        _register_trainable(
            self,
            'a_exp',
            ZBL_A_EXP,
            self.trainable,
            rngs=rngs,
            module_name='ZBLBasis',
        )
        _register_trainable(
            self,
            'a_prefactor',
            ZBL_A_PREFACTOR,
            self.trainable,
            rngs=rngs,
            module_name='ZBLBasis',
        )

    def forward(
        self,
        x: Any,
        node_attrs: Any,
        edge_index: Any,
        atomic_numbers: Any,
        node_attrs_index: Any = None,
    ) -> Any:
        backend = _require_backend(self, 'ZBLBasis')
        validate_edge_aligned_inputs(x, edge_index, module_name='ZBLBasis')
        _, receiver, z_u, z_v = edge_atomic_numbers(
            node_attrs,
            edge_index,
            atomic_numbers,
            argmax_node_attrs=backend.argmax_node_attrs,
            as_index=backend.as_index,
            node_attrs_index=node_attrs_index,
        )
        x = backend.asarray(x)
        dtype = x.dtype
        v_edges = zbl_pair_energy(
            x,
            z_u,
            z_v,
            coefficients=backend.value(self.c, dtype),
            a_exp=backend.value(self.a_exp, dtype),
            a_prefactor=backend.value(self.a_prefactor, dtype),
            exp=backend.exp,
        )
        covalent_radii = backend.value(self.covalent_radii, dtype)
        r_max = pair_covalent_radii(covalent_radii, z_u, z_v)
        envelope = polynomial_cutoff_envelope(
            x,
            r_max,
            backend.value(self.p, dtype),
        )
        scatter_sum = backend.require('scatter_sum')
        squeeze = backend.require('squeeze')
        v_zbl = scatter_sum(
            0.5 * v_edges * envelope,
            receiver,
            dim=0,
            dim_size=node_attrs.shape[0],
        )
        return squeeze(v_zbl, axis=-1)

    def __repr__(self):
        return f'{self.__class__.__name__}(c={self.c})'


class AgnesiTransform:
    BACKEND: RadialBackend

    def init(
        self,
        q: float = AGNESI_Q,
        p: float = AGNESI_P,
        a: float = AGNESI_A,
        trainable: bool = False,
        *,
        rngs: Any = None,
    ) -> None:
        self.trainable = bool(trainable)
        _register_trainable(
            self, 'q', q, self.trainable, rngs=rngs, module_name='AgnesiTransform'
        )
        _register_trainable(
            self, 'p', p, self.trainable, rngs=rngs, module_name='AgnesiTransform'
        )
        _register_trainable(
            self, 'a', a, self.trainable, rngs=rngs, module_name='AgnesiTransform'
        )
        _register_constant(self, 'covalent_radii', covalent_radii_values())

    def forward(
        self,
        x: Any,
        node_attrs: Any,
        edge_index: Any,
        atomic_numbers: Any,
        node_attrs_index: Any = None,
    ) -> Any:
        backend = _require_backend(self, 'AgnesiTransform')
        validate_edge_aligned_inputs(x, edge_index, module_name='AgnesiTransform')
        _, _, z_u, z_v = edge_atomic_numbers(
            node_attrs,
            edge_index,
            atomic_numbers,
            argmax_node_attrs=backend.argmax_node_attrs,
            as_index=backend.as_index,
            node_attrs_index=node_attrs_index,
        )
        x = backend.asarray(x)
        dtype = x.dtype
        r_0 = 0.5 * pair_covalent_radii(
            backend.value(self.covalent_radii, dtype), z_u, z_v
        )
        return agnesi_transform(
            x,
            r_0,
            a=backend.value(self.a, dtype),
            q=backend.value(self.q, dtype),
            p=backend.value(self.p, dtype),
        )

    def __repr__(self):
        backend = _require_backend(self, 'AgnesiTransform')
        return (
            f'{self.__class__.__name__}(a={backend.to_float(self.a):.4f}, '
            f'q={backend.to_float(self.q):.4f}, '
            f'p={backend.to_float(self.p):.4f})'
        )


class SoftTransform:
    BACKEND: RadialBackend

    def init(
        self,
        alpha: float = SOFT_ALPHA,
        trainable: bool = False,
        *,
        rngs: Any = None,
    ) -> None:
        self.trainable = bool(trainable)
        _register_trainable(
            self, 'alpha', alpha, self.trainable, rngs=rngs, module_name='SoftTransform'
        )
        _register_constant(self, 'covalent_radii', covalent_radii_values())

    def compute_r_0(
        self,
        node_attrs: Any,
        edge_index: Any,
        atomic_numbers: Any,
        node_attrs_index: Any = None,
    ) -> Any:
        backend = _require_backend(self, 'SoftTransform')
        _, _, z_u, z_v = edge_atomic_numbers(
            node_attrs,
            edge_index,
            atomic_numbers,
            argmax_node_attrs=backend.argmax_node_attrs,
            as_index=backend.as_index,
            node_attrs_index=node_attrs_index,
        )
        return pair_covalent_radii(self.covalent_radii, z_u, z_v)

    def forward(
        self,
        x: Any,
        node_attrs: Any,
        edge_index: Any,
        atomic_numbers: Any,
        node_attrs_index: Any = None,
    ) -> Any:
        backend = _require_backend(self, 'SoftTransform')
        validate_edge_aligned_inputs(x, edge_index, module_name='SoftTransform')
        x = backend.asarray(x)
        dtype = x.dtype
        r_0 = backend.value(
            self.compute_r_0(
                node_attrs,
                edge_index,
                atomic_numbers,
                node_attrs_index=node_attrs_index,
            ),
            dtype,
        )
        return soft_transform(
            x,
            r_0,
            alpha=backend.value(self.alpha, dtype),
            tanh=backend.tanh,
        )

    def __repr__(self):
        backend = _require_backend(self, 'SoftTransform')
        return (
            f'{self.__class__.__name__}(alpha={backend.to_float(self.alpha):.4f}, '
            f'trainable={self.trainable})'
        )


def make_radial_mlp_modules(
    backend: RadialBackend,
    channels: Sequence[int],
    *,
    rngs: Any = None,
) -> list[Any]:
    modules = []
    for spec in iter_radial_mlp_layers(channels):
        if spec.kind == 'linear':
            make_linear = backend.require('make_linear_layer')
            modules.append(
                make_linear(
                    in_channels=spec.in_channels,
                    out_channels=spec.out_channels,
                    rngs=rngs,
                )
            )
        elif spec.kind == 'norm':
            make_norm = backend.require('make_norm_layer')
            modules.append(make_norm(out_channels=spec.out_channels, rngs=rngs))
        else:
            make_activation = backend.require('make_activation_layer')
            modules.append(make_activation())
    return modules


class RadialSequential:
    BACKEND: RadialBackend

    def init(self, channels: Sequence[int], *, rngs: Any = None) -> None:
        backend = _require_backend(self, 'RadialSequential')
        self.channels = radial_mlp_layer_plan(channels)
        make_container = backend.require('make_layer_container')
        self.layers = make_container()
        self._layer_order: list[tuple[str, str | None]] = []
        for spec in iter_radial_mlp_layers(self.channels):
            if spec.kind == 'linear':
                make_linear = backend.require('make_linear_layer')
                key = str(spec.index)
                self.layers[key] = make_linear(
                    in_channels=spec.in_channels,
                    out_channels=spec.out_channels,
                    rngs=rngs,
                )
                self._layer_order.append(('linear', key))
            elif spec.kind == 'norm':
                make_norm = backend.require('make_norm_layer')
                key = str(spec.index)
                self.layers[key] = make_norm(
                    out_channels=spec.out_channels,
                    rngs=rngs,
                )
                self._layer_order.append(('norm', key))
            else:
                self._layer_order.append(('act', None))

    def forward(self, inputs: Any) -> Any:
        backend = _require_backend(self, 'RadialSequential')
        silu = backend.require('silu')
        x = inputs
        for kind, key in self._layer_order:
            if kind == 'act':
                x = silu(x)
            else:
                if key is None:
                    raise ValueError('Missing layer key for radial sequential block')
                x = self.layers[key](x)
        return x


class RadialMLP:
    BACKEND: RadialBackend

    def init(self, channels: Sequence[int], *, rngs: Any = None) -> None:
        backend = _require_backend(self, 'RadialMLP')
        self.channels = radial_mlp_layer_plan(channels)
        self.hs = list(self.channels)
        make_net = backend.require('make_radial_mlp_net')
        self.net = make_net(self.channels, rngs=rngs)

    def forward(self, inputs: Any) -> Any:
        return self.net(inputs)


__all__ = [
    'AGNESI_A',
    'AGNESI_P',
    'AGNESI_Q',
    'AgnesiTransform',
    'BesselBasis',
    'ChebychevBasis',
    'GaussianBasis',
    'PolynomialCutoff',
    'RadialBackend',
    'RadialMLPLayerSpec',
    'RadialMLP',
    'RadialSequential',
    'SOFT_ALPHA',
    'SoftTransform',
    'ZBL_A_EXP',
    'ZBL_A_PREFACTOR',
    'ZBL_COEFFICIENTS',
    'ZBLBasis',
    'agnesi_transform',
    'bessel_basis',
    'bessel_weight_values',
    'chebyshev_basis_values',
    'covalent_radii_values',
    'edge_atomic_numbers',
    'gaussian_basis',
    'gaussian_coefficient',
    'gaussian_weight_values',
    'iter_radial_mlp_layers',
    'make_radial_mlp_modules',
    'pair_covalent_radii',
    'polynomial_cutoff_envelope',
    'radial_mlp_layer_plan',
    'soft_transform',
    'use_radial_backend',
    'validate_edge_aligned_inputs',
    'validate_num_basis',
    'zbl_pair_energy',
]
