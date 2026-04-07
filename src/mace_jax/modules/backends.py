from __future__ import annotations

from e3nn_jax import Irreps
from mace_core.modules.backends import define_backend

from mace_jax.adapters.e3nn import nn
from mace_jax.modules.irreps_tools import mask_head
from mace_jax.modules.wrapper_ops import Linear


@define_backend(name="jax")
class _JaxBackendSpec:
    @staticmethod
    def make_irreps(value):
        return Irreps(value)

    @staticmethod
    def make_linear(*, irreps_in, irreps_out, cueq_config, rngs):
        return Linear(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            cueq_config=cueq_config,
            rngs=rngs,
        )

    @staticmethod
    def make_activation(*, hidden_irreps, gate, cueq_config):
        return nn.Activation(
            irreps_in=hidden_irreps,
            acts=[gate],
            layout_str=getattr(cueq_config, "layout_str", "mul_ir"),
        )

    mask_head = staticmethod(mask_head)


JAX_BACKEND = _JaxBackendSpec
