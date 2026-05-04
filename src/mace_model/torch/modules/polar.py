from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from typing import Any, NamedTuple

import torch

from mace_model.core.modules.models import PolarMACEModel
from mace_model.core.modules.polar import (
    expand_field_feature_norms,
    layout_from_cueq_config,
)
from mace_model.torch.adapters.cuequivariance import TransposeIrrepsLayoutWrapper
from mace_model.torch.adapters.e3nn import o3
from mace_model.torch.tools.scatter import scatter_mean, scatter_sum

from .blocks import NonLinearBiasReadoutBlock, NonLinearReadoutBlock
from .field_blocks import (
    AgnosticChargeBiasedLinearPotentialEmbedding,
    EnvironmentDependentSpinSourceBlock,
    MLPNonLinearity,
    MultiLayerFeatureMixer,
    NoNonLinearity,
    field_readout_blocks,
    field_update_blocks,
)
from .models import ScaleShiftMACE
from .utils import get_atomic_virials_stresses, get_outputs, prepare_graph

_GRAPH_LONGRANGE_ERROR = (
    "Cannot import 'graph_longrange'. Install graph_electrostatics from "
    'https://github.com/WillBaldwin0/graph_electrostatics to use PolarMACE.'
)


class _GraphLongrangeImports(NamedTuple):
    electrostatic_energy_cls: Any
    electrostatic_features_cls: Any
    external_field_block_cls: Any
    gto_basis_kspace_cutoff: Any
    compute_k_vectors_flat: Any


def _find_graph_longrange() -> bool:
    try:
        return importlib.util.find_spec('graph_longrange') is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


GRAPH_LONGRANGE_AVAILABLE = _find_graph_longrange()
_GRAPH_LONGRANGE_IMPORTS: _GraphLongrangeImports | None = None


def _load_graph_longrange() -> _GraphLongrangeImports:
    global GRAPH_LONGRANGE_AVAILABLE, _GRAPH_LONGRANGE_IMPORTS
    if _GRAPH_LONGRANGE_IMPORTS is not None:
        return _GRAPH_LONGRANGE_IMPORTS
    try:
        from torch.serialization import add_safe_globals

        add_safe_globals([slice])
        from graph_longrange.energy import GTOElectrostaticEnergy
        from graph_longrange.features import GTOElectrostaticFeatures
        from graph_longrange.gto_utils import (
            DisplacedGTOExternalFieldBlock,
            gto_basis_kspace_cutoff,
        )
        from graph_longrange.kspace import compute_k_vectors_flat
    except Exception as exc:
        GRAPH_LONGRANGE_AVAILABLE = False
        raise ImportError(_GRAPH_LONGRANGE_ERROR) from exc

    _GRAPH_LONGRANGE_IMPORTS = _GraphLongrangeImports(
        electrostatic_energy_cls=GTOElectrostaticEnergy,
        electrostatic_features_cls=GTOElectrostaticFeatures,
        external_field_block_cls=DisplacedGTOExternalFieldBlock,
        gto_basis_kspace_cutoff=gto_basis_kspace_cutoff,
        compute_k_vectors_flat=compute_k_vectors_flat,
    )
    GRAPH_LONGRANGE_AVAILABLE = True
    return _GRAPH_LONGRANGE_IMPORTS


_POTENTIAL_EMBEDDING_CLASSES = {
    'AgnosticChargeBiasedLinearPotentialEmbedding': (
        AgnosticChargeBiasedLinearPotentialEmbedding
    ),
}
_NONLINEARITY_CLASSES = {
    'MLPNonLinearity': MLPNonLinearity,
    'NoNonLinearity': NoNonLinearity,
}


def _permute_to_e3nn_convention(value: torch.Tensor) -> torch.Tensor:
    indices = torch.tensor([1, 2, 0], device=value.device)
    return value[..., indices]


def _compute_total_charge_dipole_permuted(
    density_coefficients: torch.Tensor,
    positions: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dipole = scatter_sum(
        src=positions * density_coefficients[:, :1],
        index=batch.unsqueeze(-1),
        dim=0,
        dim_size=num_graphs,
    )
    if density_coefficients.shape[1] > 1:
        dipole_p = scatter_sum(
            src=density_coefficients[..., 1:4],
            index=batch,
            dim=-2,
            dim_size=num_graphs,
        )
        dipole = dipole + dipole_p[..., [2, 0, 1]]

    total_charge = scatter_sum(
        src=density_coefficients[:, 0],
        index=batch,
        dim=-1,
        dim_size=num_graphs,
    )
    return total_charge, dipole


class PolarMACE(PolarMACEModel, ScaleShiftMACE):
    """Torch-side PolarMACE implementation.

    The shared Polar constructor and irreps bookkeeping live in
    :mod:`mace_model.core.modules.polar`; this class keeps the Torch-specific
    graph_longrange objects, module wiring, and tensor execution.
    """

    def __init__(
        self,
        kspace_cutoff_factor: float = 1.5,
        atomic_multipoles_max_l: int = 0,
        atomic_multipoles_smearing_width: float = 1.0,
        field_feature_max_l: int = 0,
        field_feature_widths: Sequence[float] = (1.0,),
        num_recursion_steps: int = 1,
        field_si: bool = False,
        include_electrostatic_self_interaction: bool = False,
        add_local_electron_energy: bool = False,
        quadrupole_feature_corrections: bool = False,
        return_electrostatic_potentials: bool = False,
        field_feature_norms: Sequence[float] | None = None,
        field_norm_factor: float | None = 0.02,
        fixedpoint_update_config: dict[str, Any] | None = None,
        field_readout_config: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        graph_longrange = _load_graph_longrange()

        constructor_args = self.require_polar_mace_kwargs(
            kwargs,
            make_irreps=o3.Irreps,
        )
        hidden_irreps = constructor_args.hidden_irreps
        mlp_irreps = constructor_args.mlp_irreps
        gate = constructor_args.gate
        avg_num_neighbors = constructor_args.avg_num_neighbors
        num_interactions = constructor_args.num_interactions
        num_elements = constructor_args.num_elements

        kwargs = self.prepare_polar_base_kwargs(
            kwargs,
            readout_cls=NonLinearReadoutBlock,
        )
        cueq_config = kwargs.get('cueq_config')
        oeq_config = kwargs.get('oeq_config')
        super().__init__(**kwargs)

        self.initialize_polar_common_attributes(
            kspace_cutoff_factor=kspace_cutoff_factor,
            atomic_multipoles_max_l=atomic_multipoles_max_l,
            atomic_multipoles_smearing_width=atomic_multipoles_smearing_width,
            field_feature_max_l=field_feature_max_l,
            field_feature_widths=field_feature_widths,
            num_recursion_steps=num_recursion_steps,
            field_si=field_si,
            include_electrostatic_self_interaction=include_electrostatic_self_interaction,
            add_local_electron_energy=add_local_electron_energy,
            quadrupole_feature_corrections=quadrupole_feature_corrections,
            return_electrostatic_potentials=return_electrostatic_potentials,
            field_feature_norms=field_feature_norms,
            field_norm_factor=field_norm_factor,
            fixedpoint_update_config=fixedpoint_update_config,
            field_readout_config=field_readout_config,
        )

        kspace_cutoff = (
            self.kspace_cutoff_factor
            * graph_longrange.gto_basis_kspace_cutoff(
                [self.atomic_multipoles_smearing_width] + self.field_feature_widths,
                max(self.atomic_multipoles_max_l, self.field_feature_max_l),
            )
        )
        self.register_buffer(
            'kspace_cutoff',
            torch.tensor(kspace_cutoff, dtype=torch.get_default_dtype()),
        )

        self.register_buffer(
            'field_feature_norms',
            torch.tensor(
                expand_field_feature_norms(
                    field_feature_norms=self._field_feature_norms,
                    field_feature_widths=self.field_feature_widths,
                    field_feature_max_l=self.field_feature_max_l,
                ),
                dtype=torch.get_default_dtype(),
            ),
        )

        self.lr_source_maps = torch.nn.ModuleList(
            EnvironmentDependentSpinSourceBlock(
                irreps_in=hidden_irreps,
                max_l=self.atomic_multipoles_max_l,
                cueq_config=cueq_config,
            )
            for _ in range(num_interactions)
        )

        polar_irreps = self.make_polar_irreps_layout(
            make_irreps=o3.Irreps,
            make_irrep=o3.Irrep,
            hidden_irreps=hidden_irreps,
            atomic_multipoles_max_l=self.atomic_multipoles_max_l,
            field_feature_max_l=self.field_feature_max_l,
            field_feature_widths=self.field_feature_widths,
            num_elements=num_elements,
            radial_embedding_out_dim=self.radial_embedding.out_dim,
        )
        self.charges_irreps = polar_irreps.charges_irreps
        self.field_irreps = polar_irreps.field_irreps
        self.potential_irreps = polar_irreps.potential_irreps
        self.from_ell_max_field_update = polar_irreps.from_ell_max_field_update
        layout_str = layout_from_cueq_config(cueq_config)
        self._charges_to_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.charges_irreps,
            source=layout_str,
            target='mul_ir',
            cueq_config=cueq_config,
        )
        self._charges_from_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.charges_irreps,
            source='mul_ir',
            target=layout_str,
            cueq_config=cueq_config,
        )

        self.electric_potential_descriptor = graph_longrange.electrostatic_features_cls(
            density_max_l=self.atomic_multipoles_max_l,
            density_smearing_width=self.atomic_multipoles_smearing_width,
            feature_max_l=self.field_feature_max_l,
            feature_smearing_widths=self.field_feature_widths,
            kspace_cutoff=kspace_cutoff,
            include_self_interaction=self.field_si,
            quadrupole_feature_corrections=self.quadrupole_feature_corrections,
            integral_normalization='receiver',
        )
        self._field_from_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.field_irreps,
            source='mul_ir',
            target=layout_str,
            cueq_config=cueq_config,
        )

        self.fukui_source_map = NonLinearBiasReadoutBlock(
            hidden_irreps,
            mlp_irreps.simplify(),
            gate,
            o3.Irreps('2x0e'),
            cueq_config=None,
            oeq_config=oeq_config,
        )
        self._fukui_to_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=hidden_irreps,
            source=layout_str,
            target='mul_ir',
            cueq_config=cueq_config,
        )

        field_update_cls, update_config = self.resolve_field_update_config(
            self._fixedpoint_update_config,
            field_update_registry=field_update_blocks,
            potential_embedding_registry=_POTENTIAL_EMBEDDING_CLASSES,
            nonlinearity_registry=_NONLINEARITY_CLASSES,
            default_potential_embedding_cls=AgnosticChargeBiasedLinearPotentialEmbedding,
        )
        self.field_dependent_charges_maps = torch.nn.ModuleList(
            field_update_cls(
                node_attrs_irreps=polar_irreps.node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=polar_irreps.field_update_sh_irreps,
                edge_feats_irreps=polar_irreps.edge_feats_irreps,
                target_irreps=polar_irreps.field_interaction_irreps,
                hidden_irreps=hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
                potential_irreps=self.potential_irreps,
                charges_irreps=self.charges_irreps,
                num_elements=num_elements,
                field_norm_factor=self.field_norm_factor,
                cueq_config=cueq_config,
                oeq_config=oeq_config,
                **update_config,
            )
            for _ in range(self.num_recursion_steps)
        )

        field_readout_cls, readout_config = self.resolve_field_readout_config(
            self._field_readout_config,
            field_readout_registry=field_readout_blocks,
        )
        self.local_electron_energy = field_readout_cls(
            node_attrs_irreps=polar_irreps.node_attr_irreps,
            node_feats_irreps=hidden_irreps,
            edge_attrs_irreps=polar_irreps.field_update_sh_irreps,
            edge_feats_irreps=polar_irreps.edge_feats_irreps,
            target_irreps=polar_irreps.field_interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            potential_irreps=self.potential_irreps,
            charges_irreps=self.charges_irreps,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
            **readout_config,
        )

        self.external_field_contribution = graph_longrange.external_field_block_cls(
            self.field_feature_max_l,
            self.field_feature_widths,
            'receiver',
        )
        self.coulomb_energy = graph_longrange.electrostatic_energy_cls(
            density_max_l=self.atomic_multipoles_max_l,
            density_smearing_width=self.atomic_multipoles_smearing_width,
            kspace_cutoff=float(kspace_cutoff),
            include_self_interaction=self.include_electrostatic_self_interaction,
        )
        self.layer_feature_mixer = MultiLayerFeatureMixer(
            node_feats_irreps=hidden_irreps,
            num_interactions=num_interactions,
            cueq_config=cueq_config,
        )

    def forward(
        self,
        data: dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
        use_pbc_evaluator: bool = False,
        compute_node_feats: bool = True,
        fermi_level: torch.Tensor | None = None,
        external_field: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        graph_longrange = _load_graph_longrange()

        ctx = prepare_graph(
            data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
        )
        is_lammps = ctx.is_lammps
        lammps_natoms = ctx.interaction_kwargs.lammps_natoms
        lammps_class = ctx.interaction_kwargs.lammps_class
        batch = data['batch']
        edge_index = data['edge_index']
        node_attrs = data['node_attrs']

        if fermi_level is None:
            fermi_level = data['fermi_level']
        if external_field is None:
            external_field = data['external_field']
        external_potential = torch.hstack(
            (torch.zeros_like(fermi_level).unsqueeze(-1), external_field)
        )

        node_e0 = self.atomic_energies_fn(node_attrs)[
            ctx.num_atoms_arange, ctx.node_heads
        ]
        e0 = scatter_sum(
            src=node_e0,
            index=batch,
            dim=0,
            dim_size=ctx.num_graphs,
        ).to(ctx.vectors.dtype)

        node_feats = self.node_embedding(node_attrs)
        edge_attrs = self.spherical_harmonics(_permute_to_e3nn_convention(ctx.vectors))
        edge_feats, cutoff = self.radial_embedding(
            ctx.lengths,
            node_attrs,
            edge_index,
            self.atomic_numbers,
        )
        if self.pair_repulsion:
            pair_node_energy = self.pair_repulsion_fn(
                ctx.lengths,
                node_attrs,
                edge_index,
                self.atomic_numbers,
            )
            if is_lammps:
                pair_node_energy = pair_node_energy[: lammps_natoms[0]]
        else:
            pair_node_energy = torch.zeros_like(node_e0)

        apply_embedding = self.make_apply_embedding(
            data=data,
            batch=batch,
            num_graphs=ctx.num_graphs,
        )
        node_feats, e0 = apply_embedding(node_feats, e0, ctx.node_heads)

        node_es_list: list[torch.Tensor] = [pair_node_energy]
        node_feats_list: list[torch.Tensor] = []
        spin_charge_density = torch.zeros(
            (batch.shape[0], self.charges_irreps.dim),
            device=batch.device,
            dtype=ctx.vectors.dtype,
        )

        for layer_index, (interaction, product, lr_source) in enumerate(
            zip(self.interactions, self.products, self.lr_source_maps)
        ):
            node_attrs_slice = node_attrs
            if is_lammps and layer_index > 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats, sc = interaction(
                node_attrs=node_attrs_slice,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
                cutoff=cutoff,
                first_layer=(layer_index == 0),
                lammps_class=lammps_class,
                lammps_natoms=lammps_natoms,
            )
            if is_lammps and layer_index == 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=node_attrs_slice,
            )
            node_feats_list.append(node_feats)

            feat_idx = (
                -1
                if len(self.readouts) == 1
                else min(layer_index, len(self.readouts) - 1)
            )
            node_es = self.readouts[feat_idx](node_feats, ctx.node_heads)[
                ctx.num_atoms_arange, ctx.node_heads
            ]
            node_es_list.append(node_es)
            spin_charge_density = spin_charge_density + lr_source(node_feats).squeeze(
                -2
            )

        node_feats_out = (
            torch.cat(node_feats_list, dim=-1) if compute_node_feats else None
        )
        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
        node_inter_es = self.scale_shift(node_inter_es, ctx.node_heads)
        inter_e = scatter_sum(
            node_inter_es,
            batch,
            dim=-1,
            dim_size=ctx.num_graphs,
        )

        k_vectors, kv_norms_squared, k_vectors_batch, k_vectors_0mask = (
            graph_longrange.compute_k_vectors_flat(
                self.kspace_cutoff,
                ctx.cell.view(-1, 3, 3),
                data['rcell'].view(-1, 3, 3),
            )
        )
        field_feature_cache = self.electric_potential_descriptor.precompute_geometry(
            k_vectors=k_vectors,
            k_norm2=kv_norms_squared,
            k_vector_batch=k_vectors_batch,
            k0_mask=k_vectors_0mask,
            node_positions=ctx.positions,
            batch=batch,
            volume=data['volume'],
            pbc=data['pbc'].view(-1, 3),
            force_pbc_evaluator=use_pbc_evaluator,
        )

        features_mixed = self.layer_feature_mixer(torch.stack(node_feats_list, dim=0))
        spin_charge_density = spin_charge_density.view(
            spin_charge_density.shape[0],
            2,
            -1,
        )
        fukui_sources = self.fukui_source_map(self._fukui_to_mul_ir(node_feats))
        fukui_norm = scatter_sum(
            src=fukui_sources.double(),
            index=batch,
            dim=0,
            dim_size=ctx.num_graphs,
        )[batch].to(ctx.vectors.dtype)
        fukui_norm = torch.where(
            fukui_norm == 0,
            torch.ones_like(fukui_norm),
            fukui_norm,
        )
        fukui_sources = fukui_sources / fukui_norm
        q_plus_spin = (data['total_charge'] + (data['total_spin'] - 1))[batch]
        q_minus_spin = (data['total_charge'] - (data['total_spin'] - 1))[batch]
        pred_total_charges_0 = scatter_sum(
            src=spin_charge_density[:, :, 0].double(),
            index=batch,
            dim=0,
            dim_size=ctx.num_graphs,
        )[batch].to(ctx.vectors.dtype)
        spin_charge_density = spin_charge_density.clone()
        spin_charge_density[:, 0, 0] = spin_charge_density[:, 0, 0] + (
            fukui_sources[:, 0] * ((q_plus_spin / 2) - pred_total_charges_0[:, 0])
        )
        spin_charge_density[:, 1, 0] = spin_charge_density[:, 1, 0] + (
            fukui_sources[:, 1] * ((q_minus_spin / 2) - pred_total_charges_0[:, 1])
        )

        potential_features = torch.zeros(
            (batch.shape[0], self.potential_irreps.dim),
            device=batch.device,
            dtype=ctx.vectors.dtype,
        )
        field_independent_spin_charge_density = spin_charge_density.clone()
        electrostatic_potentials: torch.Tensor | None = None

        for recursion_index in range(self.num_recursion_steps):
            source_feats_alpha = self._charges_to_mul_ir(
                spin_charge_density[:, 0, :].clone()
            )
            source_feats_beta = self._charges_to_mul_ir(
                spin_charge_density[:, 1, :].clone()
            )
            field_feats_alpha = self.electric_potential_descriptor.forward_dynamic(
                cache=field_feature_cache,
                source_feats=source_feats_alpha.unsqueeze(-2),
                pbc=data['pbc'].view(-1, 3),
            )
            field_feats_beta = self.electric_potential_descriptor.forward_dynamic(
                cache=field_feature_cache,
                source_feats=source_feats_beta.unsqueeze(-2),
                pbc=data['pbc'].view(-1, 3),
            )
            field_feats_alpha = self._field_from_mul_ir(field_feats_alpha)
            field_feats_beta = self._field_from_mul_ir(field_feats_beta)
            electrostatic_potentials = None

            barycenter = scatter_mean(
                src=ctx.positions.double(),
                index=batch,
                dim=0,
                dim_size=ctx.num_graphs,
            ).to(ctx.positions.dtype)
            half_external_field = 0.5 * self.external_field_contribution(
                batch,
                ctx.positions - barycenter[batch, :],
                external_potential,
            )
            field_feats_alpha = (
                field_feats_alpha + half_external_field
            ) / self.field_feature_norms
            field_feats_beta = (
                field_feats_beta + half_external_field
            ) / self.field_feature_norms
            potential_features = torch.cat(
                (field_feats_alpha, field_feats_beta),
                dim=-1,
            )
            charge_sources_out = self.field_dependent_charges_maps[recursion_index](
                node_attrs=node_attrs,
                node_feats=features_mixed,
                edge_attrs=edge_attrs[:, : self.from_ell_max_field_update],
                edge_feats=edge_feats,
                edge_index=edge_index,
                potential_features=potential_features,
                local_charges=spin_charge_density.view(
                    spin_charge_density.shape[0],
                    -1,
                ),
            )

            current_fukui_sources = charge_sources_out[:, -2:]
            charge_sources = charge_sources_out[:, :-2]
            spin_charge_density = spin_charge_density + charge_sources.view(
                spin_charge_density.shape[0],
                2,
                -1,
            )
            fukui_norm2 = scatter_sum(
                src=current_fukui_sources.double(),
                index=batch,
                dim=0,
                dim_size=ctx.num_graphs,
            )[batch].to(ctx.vectors.dtype)
            fukui_norm2 = torch.where(
                fukui_norm2 == 0,
                torch.ones_like(fukui_norm2),
                fukui_norm2,
            )
            current_fukui_sources = current_fukui_sources / fukui_norm2
            pred_total_charges = scatter_sum(
                src=spin_charge_density[:, :, 0].double(),
                index=batch,
                dim=0,
                dim_size=ctx.num_graphs,
            )[batch].to(ctx.vectors.dtype)
            spin_charge_density = spin_charge_density.clone()
            spin_charge_density[:, 0, 0] = spin_charge_density[:, 0, 0] + (
                current_fukui_sources[:, 0]
                * ((q_plus_spin / 2) - pred_total_charges[:, 0])
            )
            spin_charge_density[:, 1, 0] = spin_charge_density[:, 1, 0] + (
                current_fukui_sources[:, 1]
                * ((q_minus_spin / 2) - pred_total_charges[:, 1])
            )

        total_energy = e0 + inter_e
        local_q_e = self.local_electron_energy(
            node_attrs=node_attrs,
            node_feats=node_feats,
            edge_attrs=edge_attrs[:, : self.from_ell_max_field_update],
            edge_feats=edge_feats,
            edge_index=edge_index,
            field_feats=potential_features,
            charges_0=field_independent_spin_charge_density.view(
                field_independent_spin_charge_density.shape[0],
                -1,
            ),
            charges_induced=spin_charge_density.view(
                spin_charge_density.shape[0],
                -1,
            ),
        )
        electron_energy = scatter_sum(
            src=local_q_e,
            index=batch,
            dim=-1,
            dim_size=ctx.num_graphs,
        )
        if self.add_local_electron_energy:
            total_energy = total_energy + electron_energy
        else:
            electron_energy = torch.zeros_like(electron_energy)

        charge_density = spin_charge_density.sum(dim=1)
        spin_density = spin_charge_density[:, 0, :] - spin_charge_density[:, 1, :]
        charge_density_mul_ir = self._charges_to_mul_ir(charge_density)
        spin_density_mul_ir = self._charges_to_mul_ir(spin_density)
        spin_charge_density_mul_ir = torch.stack(
            [
                self._charges_to_mul_ir(spin_charge_density[:, 0, :]),
                self._charges_to_mul_ir(spin_charge_density[:, 1, :]),
            ],
            dim=1,
        )
        total_charge, total_dipole = _compute_total_charge_dipole_permuted(
            charge_density_mul_ir,
            ctx.positions,
            batch,
            ctx.num_graphs,
        )
        electrostatic_energy = self.coulomb_energy(
            k_vectors=k_vectors,
            k_norm2=kv_norms_squared,
            k_vector_batch=k_vectors_batch,
            k0_mask=k_vectors_0mask,
            source_feats=charge_density_mul_ir,
            node_positions=ctx.positions,
            batch=batch,
            volume=data['volume'],
            pbc=data['pbc'].view(-1, 3),
            force_pbc_evaluator=use_pbc_evaluator,
        )
        total_energy = (
            total_energy
            + electrostatic_energy
            + torch.sum(external_potential[:, 1:] * total_dipole, dim=-1)
        )

        forces, virials, stress, hessian, edge_forces = get_outputs(
            energy=total_energy,
            positions=ctx.positions,
            displacement=ctx.displacement,
            vectors=ctx.vectors,
            cell=ctx.cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces or compute_atomic_stresses,
        )
        atomic_virials: torch.Tensor | None = None
        atomic_stresses: torch.Tensor | None = None
        if compute_atomic_stresses and edge_forces is not None:
            atomic_virials, atomic_stresses = get_atomic_virials_stresses(
                edge_forces=edge_forces,
                edge_index=edge_index,
                vectors=ctx.vectors,
                num_atoms=ctx.positions.shape[0],
                batch=batch,
                cell=ctx.cell,
            )

        return {
            'energy': total_energy,
            'node_energy': node_e0.clone().double() + node_inter_es.clone().double(),
            'interaction_energy': inter_e,
            'forces': forces,
            'edge_forces': edge_forces,
            'virials': virials,
            'stress': stress,
            'atomic_virials': atomic_virials,
            'atomic_stresses': atomic_stresses,
            'hessian': hessian,
            'displacement': ctx.displacement,
            'node_feats': node_feats_out,
            'density_coefficients': charge_density_mul_ir,
            'spin_density': spin_density_mul_ir,
            'charges_history': torch.stack(
                [spin_charge_density_mul_ir.clone().detach()],
                dim=-1,
            ),
            'fermi_level': external_potential[:, 0],
            'external_field': external_potential[:, 1:],
            'charges': charge_density_mul_ir[:, 0],
            'spins': spin_density_mul_ir[:, 0],
            'dipole': total_dipole,
            'total_charge': total_charge,
            'electrostatic_energy': electrostatic_energy,
            'electron_energy': electron_energy,
            'electrostatic_potentials': electrostatic_potentials,
            'spin_charge_density': spin_charge_density_mul_ir,
        }


__all__ = ['GRAPH_LONGRANGE_AVAILABLE', 'PolarMACE']
