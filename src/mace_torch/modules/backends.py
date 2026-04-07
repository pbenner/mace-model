from __future__ import annotations

from e3nn import nn

from mace.modules.irreps_tools import mask_head
from mace.modules.wrapper_ops import Linear
from mace.tools.compile import simplify_if_compile

from mace_core.modules.backends import define_backend


@define_backend(name="torch")
class _TorchBackendSpec:
    @staticmethod
    def make_irreps(value):
        return value

    @staticmethod
    def make_linear(*, irreps_in, irreps_out, cueq_config, rngs):
        del rngs
        return Linear(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            cueq_config=cueq_config,
        )

    @staticmethod
    def make_activation(*, hidden_irreps, gate, cueq_config):
        del cueq_config
        return simplify_if_compile(nn.Activation)(
            irreps_in=hidden_irreps,
            acts=[gate],
        )

    mask_head = staticmethod(mask_head)


TORCH_BACKEND = _TorchBackendSpec
