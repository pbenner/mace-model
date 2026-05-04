from __future__ import annotations

import math
from abc import abstractmethod
from typing import Any

import cuequivariance as cue
import torch

from mace_model.core.modules.irreps_utils import (
    tp_out_irreps_with_instructions as _core_tp_out_irreps_with_instructions,
)
from mace_model.core.modules.polar import layout_from_cueq_config
from mace_model.torch.adapters.cuequivariance import (
    CuEquivarianceConfig,
    Linear,
    OEQConfig,
    TransposeIrrepsLayoutWrapper,
)
from mace_model.torch.adapters.e3nn import nn, o3

from .radial import RadialMLP


def _tp_out_irreps_with_instructions(
    irreps1: o3.Irreps,
    irreps2: o3.Irreps,
    target_irreps: o3.Irreps,
):
    return _core_tp_out_irreps_with_instructions(
        make_irreps=o3.Irreps,
        irreps1=irreps1,
        irreps2=irreps2,
        target_irreps=target_irreps,
    )


class MultiLayerFeatureMixer(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        num_interactions: int,
        cueq_config: CuEquivarianceConfig | None = None,
    ) -> None:
        super().__init__()
        self.linears = torch.nn.ModuleList(
            [
                Linear(node_feats_irreps, node_feats_irreps, cueq_config=cueq_config)
                for _ in range(int(num_interactions))
            ]
        )

    def forward(self, all_node_feats: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(all_node_feats[0])
        for layer_index, linear in enumerate(self.linears):
            out = out + linear(all_node_feats[layer_index])
        return out


class EnvironmentDependentSpinSourceBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        max_l: int,
        zero_charges: bool = False,
        cueq_config: CuEquivarianceConfig | None = None,
    ) -> None:
        super().__init__()
        self.zero_charges = bool(zero_charges)
        self.irreps_out = 2 * o3.Irreps.spherical_harmonics(int(max_l))
        self.linear = Linear(irreps_in, self.irreps_out, cueq_config=cueq_config)

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
        multipoles = self.linear(node_feats)
        if self.zero_charges:
            masked = torch.zeros_like(multipoles)
            masked[:, 1:] = multipoles[:, 1:]
            multipoles = masked
        return multipoles.unsqueeze(-2)


class PotentialEmbeddingBlock(torch.nn.Module):
    def __init__(
        self,
        potential_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        node_attrs_irreps: o3.Irreps,
        cueq_config: CuEquivarianceConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.potential_irreps = o3.Irreps(potential_irreps)
        self.node_feats_irreps = o3.Irreps(node_feats_irreps)
        self.node_attrs_irreps = o3.Irreps(node_attrs_irreps)
        self.cueq_config = cueq_config
        self._setup(**kwargs)

    @abstractmethod
    def _setup(self, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        potential_feats: torch.Tensor,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        *args,
    ) -> torch.Tensor:
        raise NotImplementedError


class AgnosticChargeBiasedLinearPotentialEmbedding(PotentialEmbeddingBlock):
    def _setup(self, charges_irreps: o3.Irreps) -> None:
        self.potential_linear = o3.Linear(
            self.potential_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        self.node_feats_linear = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        self.charges_irreps = o3.Irreps(charges_irreps)
        self.charge_embedding = o3.Linear(
            self.charges_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        layout_str = layout_from_cueq_config(self.cueq_config)
        self._potential_to_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.potential_irreps,
            source=layout_str,
            target='mul_ir',
            cueq_config=self.cueq_config,
        )
        self._node_feats_to_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.node_feats_irreps,
            source=layout_str,
            target='mul_ir',
            cueq_config=self.cueq_config,
        )
        self._charges_to_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.charges_irreps,
            source=layout_str,
            target='mul_ir',
            cueq_config=self.cueq_config,
        )
        self._node_feats_from_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.node_feats_irreps,
            source='mul_ir',
            target=layout_str,
            cueq_config=self.cueq_config,
        )

    def forward(
        self,
        potential_feats: torch.Tensor,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        local_charges: torch.Tensor,
    ) -> torch.Tensor:
        del node_attrs
        potential_in = self._potential_to_mul_ir(potential_feats)
        node_feats_in = self._node_feats_to_mul_ir(node_feats)
        charges_in = self._charges_to_mul_ir(local_charges)

        potential_emb = self.potential_linear(potential_in)
        node_feats_emb = self.node_feats_linear(node_feats_in)
        charges_emb = self.charge_embedding(charges_in)
        return self._node_feats_from_mul_ir(
            potential_emb + node_feats_emb + charges_emb
        )


class NoNonLinearity(torch.nn.Module):
    def __init__(self, invar_irreps: o3.Irreps) -> None:
        super().__init__()
        self.irreps = invar_irreps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


class MLPNonLinearity(torch.nn.Module):
    def __init__(self, invar_irreps: o3.Irreps) -> None:
        super().__init__()
        channels = o3.Irreps(invar_irreps).count(o3.Irrep(0, 1))
        self.mlp = nn.FullyConnectedNet(
            [channels, 64, 64, channels],
            torch.nn.functional.silu,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.mlp(value)


class FieldUpdateBlock(torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        hidden_irreps: o3.Irreps,
        avg_num_neighbors: float,
        potential_irreps: o3.Irreps,
        charges_irreps: o3.Irreps,
        field_norm_factor: float,
        radial_MLP: list[int] | None = None,
        cueq_config: CuEquivarianceConfig | None = None,
        oeq_config: OEQConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.node_attrs_irreps = o3.Irreps(node_attrs_irreps)
        self.node_feats_irreps = o3.Irreps(node_feats_irreps)
        self.edge_attrs_irreps = o3.Irreps(edge_attrs_irreps)
        self.edge_feats_irreps = o3.Irreps(edge_feats_irreps)
        self.target_irreps = o3.Irreps(target_irreps)
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.avg_num_neighbors = float(avg_num_neighbors)
        self.potential_irreps = o3.Irreps(potential_irreps)
        self.charges_irreps = o3.Irreps(charges_irreps)
        self.radial_MLP = radial_MLP
        self.register_buffer(
            'field_norm_factor',
            torch.tensor(float(field_norm_factor), dtype=torch.get_default_dtype()),
        )
        self.cueq_config = cueq_config
        self.oeq_config = oeq_config
        self._setup(**kwargs)

    @abstractmethod
    def _setup(self, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        potential_features: torch.Tensor,
        local_charges: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


def instructions_for_sparse_tp(
    feat_in1: o3.Irreps, feat_in2: o3.Irreps, feat_out: o3.Irreps
):
    channels1 = o3.Irreps(feat_in1).count(o3.Irrep(0, 1))
    channels2 = o3.Irreps(feat_in2).count(o3.Irrep(0, 1))
    channels3 = o3.Irreps(feat_out).count(o3.Irrep(0, 1))
    if channels1 != channels2 or channels1 != channels3:
        raise ValueError('Sparse tensor product scalar channels must match.')
    _, instructions = _tp_out_irreps_with_instructions(feat_in1, feat_in2, feat_out)
    return [(i, j, 0, mode, trainable) for i, j, _k, mode, trainable in instructions]


class SparseUvuTensorProduct(torch.nn.Module):
    """Torch-native sparse `uvu` tensor product backed by cue Clebsch-Gordan data."""

    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: list[tuple[int, int, int, str, bool]],
        layout: str = 'mul_ir',
    ) -> None:
        super().__init__()
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)
        if layout not in {'mul_ir', 'ir_mul'}:
            raise ValueError(f'Unsupported sparse TP layout {layout!r}.')
        self.layout = layout

        normalized_instructions = [
            tuple(instruction) if len(instruction) == 6 else (*tuple(instruction), 1.0)
            for instruction in instructions
        ]
        self.instructions = normalized_instructions
        self.shared_weights = True
        self.internal_weights = True

        in1_slices = self.irreps_in1.slices()
        in2_slices = self.irreps_in2.slices()
        out_slices = self.irreps_out.slices()
        self._path_meta: list[
            tuple[
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                float,
                int,
                int,
                int,
                int,
                int,
                str,
            ]
        ] = []

        def num_elements(instruction) -> int:
            i_in1, i_in2, _i_out, mode, _has_weight, _path_weight = instruction
            if mode != 'uvu':
                raise NotImplementedError(
                    "SparseUvuTensorProduct only supports 'uvu' connection mode."
                )
            del i_in1
            mul2, _ir2 = self.irreps_in2[int(i_in2)]
            return int(mul2)

        normalization_coefficients = []
        for instruction in normalized_instructions:
            i_in1, i_in2, i_out, mode, _has_weight, path_weight = instruction
            if mode != 'uvu':
                raise NotImplementedError(
                    "SparseUvuTensorProduct only supports 'uvu' connection mode."
                )
            _mul1, ir1 = self.irreps_in1[int(i_in1)]
            _mul2, ir2 = self.irreps_in2[int(i_in2)]
            _mul_out, ir_out = self.irreps_out[int(i_out)]
            if ir1.p * ir2.p != ir_out.p:
                raise ValueError('Sparse tensor-product path has incompatible parity.')
            x = sum(
                num_elements(other)
                for other in normalized_instructions
                if int(other[2]) == int(i_out)
            )
            alpha = float(ir_out.dim)
            if x > 0:
                alpha /= float(x)
            alpha *= float(path_weight)
            normalization_coefficients.append(math.sqrt(alpha))

        output_mask_fields = []
        for out_index, (mul, ir) in enumerate(self.irreps_out):
            active = any(
                int(instruction[2]) == out_index
                and normalization_coefficients[instruction_index] != 0
                for instruction_index, instruction in enumerate(normalized_instructions)
            )
            value = 1.0 if active else 0.0
            output_mask_fields.append(
                torch.full(
                    (int(mul) * int(ir.dim),),
                    value,
                    dtype=torch.get_default_dtype(),
                )
            )
        output_mask = (
            torch.cat(output_mask_fields)
            if output_mask_fields
            else torch.ones(0, dtype=torch.get_default_dtype())
        )
        self.register_buffer('output_mask', output_mask)

        weight_offset = 0
        for path_index, (instruction, path_weight) in enumerate(
            zip(normalized_instructions, normalization_coefficients)
        ):
            i1, i2, io, _mode, _has_weight, _raw_path_weight = instruction
            i1 = int(i1)
            i2 = int(i2)
            io = int(io)
            mul1, ir1 = self.irreps_in1[i1]
            mul2, ir2 = self.irreps_in2[i2]
            mul_out, ir_out = self.irreps_out[io]
            dim1 = int(ir1.dim)
            dim2 = int(ir2.dim)
            dim_out = int(ir_out.dim)
            if mul_out != mul1:
                raise NotImplementedError(
                    'SparseUvuTensorProduct requires output multiplicity to match '
                    'the first input multiplicity.'
                )
            cg = cue.O3.clebsch_gordan(ir1, ir2, ir_out)[0]
            cg_name = f'_cg_{path_index}'
            self.register_buffer(
                cg_name,
                torch.as_tensor(cg, dtype=torch.get_default_dtype()),
                persistent=False,
            )

            weight_size = int(mul1) * int(mul2)
            self._path_meta.append(
                (
                    in1_slices[i1].start,
                    in1_slices[i1].stop,
                    in2_slices[i2].start,
                    in2_slices[i2].stop,
                    out_slices[io].start,
                    out_slices[io].stop,
                    weight_offset,
                    weight_offset + weight_size,
                    float(path_weight),
                    int(mul1),
                    int(mul2),
                    dim1,
                    dim2,
                    dim_out,
                    cg_name,
                )
            )
            weight_offset += weight_size
        self.weight_numel = weight_offset
        self.weight = torch.nn.Parameter(torch.randn(self.weight_numel))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if x1.ndim != 2 or x2.ndim != 2:
            raise ValueError(
                'SparseUvuTensorProduct expects flattened [batch, irreps.dim] tensors.'
            )

        batch = x1.shape[0]
        out = x1.new_zeros((batch, self.irreps_out.dim))

        def to_mul_ir(block: torch.Tensor, mul: int, dim: int) -> torch.Tensor:
            if self.layout == 'mul_ir':
                return block.view(batch, mul, dim)
            return block.view(batch, dim, mul).transpose(1, 2)

        def from_mul_ir(value: torch.Tensor) -> torch.Tensor:
            if self.layout == 'mul_ir':
                return value.reshape(batch, -1)
            return value.transpose(1, 2).reshape(batch, -1)

        for (
            in1_start,
            in1_stop,
            in2_start,
            in2_stop,
            out_start,
            out_stop,
            weight_start,
            weight_stop,
            path_weight,
            mul1,
            mul2,
            dim1,
            dim2,
            dim_out,
            cg_name,
        ) in self._path_meta:
            in1_block = x1[:, in1_start:in1_stop]
            in2_block = x2[:, in2_start:in2_stop]
            out_block = out[:, out_start:out_stop]
            weight = self.weight[weight_start:weight_stop].view(mul1, mul2)
            cg = getattr(self, cg_name).to(dtype=x1.dtype, device=x1.device)
            x1_view = to_mul_ir(in1_block, mul1, dim1)
            x2_view = to_mul_ir(in2_block, mul2, dim2)
            coupled = torch.einsum('bua,bvd,adc->buvc', x1_view, x2_view, cg)
            contribution = path_weight * torch.einsum(
                'buvc,uv->buc',
                coupled,
                weight,
            )
            out_block_view = to_mul_ir(out_block, mul1, dim_out)
            out[:, out_start:out_stop] = from_mul_ir(out_block_view + contribution)

        return out * self.output_mask.to(dtype=out.dtype, device=out.device)


class GeneralNonLinearBiasReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Any,
        irrep_out: o3.Irreps = o3.Irreps('0e'),
        irreps_out: o3.Irreps | None = None,
        cueq_config: CuEquivarianceConfig | None = None,
        oeq_config: OEQConfig | None = None,
    ) -> None:
        super().__init__()
        del oeq_config
        self.hidden_irreps = o3.Irreps(MLP_irreps)
        self.irreps_out = o3.Irreps(irrep_out if irreps_out is None else irreps_out)
        irreps_scalars = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.hidden_irreps
                if ir.l == 0 and ir in self.irreps_out
            ]
        )
        irreps_gated = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.hidden_irreps
                if ir.l > 0 and ir in self.irreps_out
            ]
        )
        irreps_gates = o3.Irreps([(mul, (0, 1)) for mul, _ in irreps_gated])
        activation_fn = gate if gate is not None else torch.nn.functional.silu
        self.equivariant_nonlin = nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[activation_fn for _ in irreps_scalars],
            irreps_gates=irreps_gates,
            act_gates=[torch.nn.functional.sigmoid] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()
        self.linear_1 = Linear(
            irreps_in=irreps_in,
            irreps_out=self.irreps_nonlin,
            cueq_config=cueq_config,
        )
        self.linear_mid = o3.Linear(
            irreps_in=self.hidden_irreps,
            irreps_out=self.irreps_nonlin,
            biases=True,
        )
        self.linear_2 = o3.Linear(
            irreps_in=self.hidden_irreps,
            irreps_out=self.irreps_out,
            biases=True,
        )
        layout_str = layout_from_cueq_config(cueq_config)
        self._tp_to_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.irreps_nonlin,
            source=layout_str,
            target='mul_ir',
            cueq_config=cueq_config,
        )
        self._tp_from_mul_ir_out = TransposeIrrepsLayoutWrapper(
            irreps=self.irreps_out,
            source='mul_ir',
            target=layout_str,
            cueq_config=cueq_config,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.linear_1(value)
        value = self._tp_to_mul_ir(value)
        value = self.equivariant_nonlin(value)
        value = self.linear_mid(value)
        value = self.equivariant_nonlin(value)
        value = self.linear_2(value)
        return self._tp_from_mul_ir_out(value)


class AgnosticEmbeddedOneBodyVariableUpdate(FieldUpdateBlock):
    def _setup(
        self,
        potential_embedding_cls: type[
            PotentialEmbeddingBlock
        ] = AgnosticChargeBiasedLinearPotentialEmbedding,
        nonlinearity_cls: type[torch.nn.Module] = NoNonLinearity,
        num_elements: int | None = None,
        **kwargs,
    ) -> None:
        del kwargs, nonlinearity_cls, num_elements
        invar_irreps = o3.Irreps(f'{self.node_feats_irreps.count(o3.Irrep(0, 1))}x0e')
        self.potential_embedding = potential_embedding_cls(
            potential_irreps=self.potential_irreps,
            node_feats_irreps=self.node_feats_irreps,
            node_attrs_irreps=self.node_attrs_irreps,
            charges_irreps=self.charges_irreps,
            cueq_config=self.cueq_config,
        )
        self.source_embedding = Linear(
            self.node_attrs_irreps,
            invar_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        dot_instructions = instructions_for_sparse_tp(
            self.node_feats_irreps, self.node_feats_irreps, invar_irreps
        )
        self.dot_products = SparseUvuTensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=self.node_feats_irreps,
            irreps_out=invar_irreps,
            instructions=dot_instructions,
            layout=layout_from_cueq_config(self.cueq_config),
        )
        self.nonlinearity = RadialMLP(
            [2 * invar_irreps.dim] + [64, 64, 64] + [invar_irreps.dim]
        )
        _, tp_instructions = _tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            invar_irreps,
            self.node_feats_irreps,
        )
        self.tp_out = SparseUvuTensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=invar_irreps,
            irreps_out=self.node_feats_irreps,
            instructions=tp_instructions,
            layout=layout_from_cueq_config(self.cueq_config),
        )
        mlp_irreps = (
            (32 * o3.Irreps.spherical_harmonics(self.charges_irreps.lmax))
            .sort()[0]
            .simplify()
        )
        self.readout = GeneralNonLinearBiasReadoutBlock(
            irreps_in=self.node_feats_irreps,
            MLP_irreps=mlp_irreps,
            gate=torch.nn.functional.silu,
            irreps_out=self.charges_irreps + o3.Irreps('2x0e'),
            cueq_config=None,
            oeq_config=self.oeq_config,
        )
        layout_str = layout_from_cueq_config(self.cueq_config)
        self._readout_to_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.node_feats_irreps,
            source=layout_str,
            target='mul_ir',
            cueq_config=self.cueq_config,
        )
        self._readout_from_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.charges_irreps + o3.Irreps('2x0e'),
            source='mul_ir',
            target=layout_str,
            cueq_config=self.cueq_config,
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        potential_features: torch.Tensor,
        local_charges: torch.Tensor,
    ) -> torch.Tensor:
        del edge_attrs, edge_feats, edge_index
        mixed_feats = self.potential_embedding(
            potential_features,
            node_feats,
            node_attrs,
            local_charges,
        )
        invariant_descriptors = self.dot_products(node_feats, mixed_feats)
        source_embedding = self.source_embedding(node_attrs)
        invariant_descriptors_embedded = torch.cat(
            [invariant_descriptors, source_embedding], dim=-1
        )
        nonlin_feats = self.nonlinearity(invariant_descriptors_embedded)
        new_feats = self.tp_out(node_feats, nonlin_feats)
        multipoles = self.readout(self._readout_to_mul_ir(new_feats))
        return self._readout_from_mul_ir(multipoles)


class PostScfReadout(torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        hidden_irreps: o3.Irreps,
        avg_num_neighbors: float,
        potential_irreps: o3.Irreps,
        charges_irreps: o3.Irreps,
        radial_MLP: list[int] | None = None,
        cueq_config: CuEquivarianceConfig | None = None,
        oeq_config: OEQConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.node_attrs_irreps = o3.Irreps(node_attrs_irreps)
        self.node_feats_irreps = o3.Irreps(node_feats_irreps)
        self.edge_attrs_irreps = o3.Irreps(edge_attrs_irreps)
        self.edge_feats_irreps = o3.Irreps(edge_feats_irreps)
        self.target_irreps = o3.Irreps(target_irreps)
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.avg_num_neighbors = float(avg_num_neighbors)
        self.radial_MLP = radial_MLP or [64, 64, 64]
        self.potential_irreps = o3.Irreps(potential_irreps)
        self.charges_irreps = o3.Irreps(charges_irreps)
        self.cueq_config = cueq_config
        self.oeq_config = oeq_config
        self._setup(**kwargs)

    @abstractmethod
    def _setup(self, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        field_feats: torch.Tensor,
        charges_0: torch.Tensor,
        charges_induced: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class OneBodyMLPFieldReadout(PostScfReadout):
    def _setup(self, **kwargs) -> None:
        del kwargs
        invar_irreps = o3.Irreps(f'{self.node_feats_irreps.count(o3.Irrep(0, 1))}x0e')
        self.linear_up_q = o3.Linear(
            self.charges_irreps,
            self.node_feats_irreps,
            biases=True,
        )
        self.linear_up_v = o3.Linear(
            self.potential_irreps,
            self.node_feats_irreps,
            biases=True,
        )
        layout_str = layout_from_cueq_config(self.cueq_config)
        self._q_to_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.charges_irreps,
            source=layout_str,
            target='mul_ir',
            cueq_config=self.cueq_config,
        )
        self._v_to_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.potential_irreps,
            source=layout_str,
            target='mul_ir',
            cueq_config=self.cueq_config,
        )
        self._up_from_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.node_feats_irreps,
            source='mul_ir',
            target=layout_str,
            cueq_config=self.cueq_config,
        )
        dot_instructions = instructions_for_sparse_tp(
            self.node_feats_irreps, self.node_feats_irreps, invar_irreps
        )
        self.dot_products_q = SparseUvuTensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=self.node_feats_irreps,
            irreps_out=invar_irreps,
            instructions=dot_instructions,
            layout=layout_from_cueq_config(self.cueq_config),
        )
        self.dot_products_v = SparseUvuTensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=self.node_feats_irreps,
            irreps_out=invar_irreps,
            instructions=dot_instructions,
            layout=layout_from_cueq_config(self.cueq_config),
        )
        self.mlp = RadialMLP([2 * invar_irreps.dim] + [128, 128, 128] + [1])

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        field_feats: torch.Tensor,
        charges_0: torch.Tensor,
        charges_induced: torch.Tensor,
    ) -> torch.Tensor:
        del node_attrs, edge_attrs, edge_feats, edge_index
        q_in = self._q_to_mul_ir(charges_induced + charges_0)
        q_up = self._up_from_mul_ir(self.linear_up_q(q_in))
        v_in = self._v_to_mul_ir(field_feats)
        v_up = self._up_from_mul_ir(self.linear_up_v(v_in))
        invar_feats = torch.cat(
            [
                self.dot_products_q(node_feats, q_up),
                self.dot_products_v(node_feats, v_up),
            ],
            dim=-1,
        )
        return self.mlp(invar_feats).squeeze(-1)


field_update_blocks = {
    'AgnosticEmbeddedOneBodyVariableUpdate': AgnosticEmbeddedOneBodyVariableUpdate,
}

field_readout_blocks = {
    'OneBodyMLPFieldReadout': OneBodyMLPFieldReadout,
}


__all__ = [
    'AgnosticChargeBiasedLinearPotentialEmbedding',
    'AgnosticEmbeddedOneBodyVariableUpdate',
    'EnvironmentDependentSpinSourceBlock',
    'FieldUpdateBlock',
    'GeneralNonLinearBiasReadoutBlock',
    'MLPNonLinearity',
    'MultiLayerFeatureMixer',
    'NoNonLinearity',
    'OneBodyMLPFieldReadout',
    'PostScfReadout',
    'PotentialEmbeddingBlock',
    'SparseUvuTensorProduct',
    'field_readout_blocks',
    'field_update_blocks',
    'instructions_for_sparse_tp',
]
