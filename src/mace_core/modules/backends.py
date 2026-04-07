from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ModelBackend:
    """
    Unified backend operation bundle for model modules.

    The readout mixin uses a subset of these operations today. Additional module
    mixins can reuse the same backend object as they are introduced.
    """

    name: str

    # Core operations already used by the readout block.
    make_irreps: Callable[[Any], Any] | None = None
    make_linear: Callable[..., Any] | None = None
    make_activation: Callable[..., Any] | None = None
    mask_head: Callable[[Any, Any, int], Any] | None = None

    # Additional module operations to support broader MACE blocks.
    make_tensor_product: Callable[..., Any] | None = None
    make_fully_connected_tensor_product: Callable[..., Any] | None = None
    make_symmetric_contraction: Callable[..., Any] | None = None
    tp_out_irreps_with_instructions: Callable[..., Any] | None = None
    reshape_irreps: Callable[..., Any] | None = None
    scatter_sum: Callable[..., Any] | None = None

    def require(self, field_name: str) -> Callable[..., Any]:
        fn = getattr(self, field_name, None)
        if fn is None:
            raise NotImplementedError(
                f"ModelBackend '{self.name}' is missing required operation "
                f"'{field_name}'."
            )
        return fn


_MODEL_BACKEND_FIELDS = (
    "make_irreps",
    "make_linear",
    "make_activation",
    "mask_head",
    "make_tensor_product",
    "make_fully_connected_tensor_product",
    "make_symmetric_contraction",
    "tp_out_irreps_with_instructions",
    "reshape_irreps",
    "scatter_sum",
)


def define_backend(*, name: str):
    """
    Decorator that turns a class of backend operations into a ModelBackend.
    """

    def decorator(ops_cls):
        kwargs = {}
        for field_name in _MODEL_BACKEND_FIELDS:
            fn = getattr(ops_cls, field_name, None)
            if fn is not None:
                kwargs[field_name] = fn
        return ModelBackend(name=name, **kwargs)

    return decorator


def use_backend(backend: ModelBackend):
    """
    Decorator to bind a specific backend implementation to a block class.
    """

    def decorator(cls):
        cls.BACKEND = backend
        return cls

    return decorator
