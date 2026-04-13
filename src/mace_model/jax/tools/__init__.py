from __future__ import annotations

from importlib import import_module

from .dtype import default_dtype
from .scatter import scatter_sum

__all__ = [
    "build_model",
    "bundle",
    "coerce_irreps",
    "convert_torch_to_jax",
    "default_dtype",
    "load_model_bundle",
    "model_builder",
    "normalize_atomic_config",
    "prepare_template_data",
    "resolve_model_paths",
    "scatter_sum",
]


def __getattr__(name: str):
    if name in {"bundle", "model_builder"}:
        return import_module(f".{name}", __name__)
    if name == "convert_torch_to_jax":
        module = import_module(".torch_import", __name__)
        return getattr(module, name)
    if name in {
        "build_model",
        "coerce_irreps",
        "normalize_atomic_config",
        "prepare_template_data",
    }:
        module = import_module(".model_builder", __name__)
        return getattr(module, name)
    if name in {"load_model_bundle", "resolve_model_paths"}:
        module = import_module(".bundle", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
