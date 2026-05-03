"""Compatibility helpers for cuequivariance Torch imports."""

from __future__ import annotations

import importlib
import warnings


def _import_cuequivariance_torch():
    original_warn = warnings.warn

    def warn(message, *args, **kwargs):
        if isinstance(message, str) and message.startswith(
            '`torch.jit.script` is deprecated'
        ):
            return None
        return original_warn(message, *args, **kwargs)

    with warnings.catch_warnings():
        warnings.warn = warn
        try:
            return importlib.import_module('cuequivariance_torch')
        finally:
            warnings.warn = original_warn


cuequivariance_torch = _import_cuequivariance_torch()


__all__ = ['cuequivariance_torch']
