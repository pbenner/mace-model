from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from mace_model.torch.adapters.e3nn import o3
from mace_model.torch.modules.blocks import RealAgnosticInteractionBlock
from mace_model.torch.modules.field_blocks import (
    EnvironmentDependentSpinSourceBlock,
    MultiLayerFeatureMixer,
    SparseUvuTensorProduct,
    instructions_for_sparse_tp,
)
from mace_model.torch.modules.models import PolarMACE
from mace_model.torch.modules.polar import GRAPH_LONGRANGE_AVAILABLE

pytestmark = [
    pytest.mark.filterwarnings(
        'ignore:`torch\\.jit\\.script` is deprecated.*:DeprecationWarning'
    ),
    pytest.mark.filterwarnings(
        'ignore:__array_wrap__ must accept context and return_scalar arguments.*:DeprecationWarning'
    ),
]


def _make_polar_model(*, pair_repulsion: bool = False) -> PolarMACE:
    return PolarMACE(
        r_max=4.5,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=1,
        interaction_cls=RealAgnosticInteractionBlock,
        interaction_cls_first=RealAgnosticInteractionBlock,
        num_interactions=1,
        num_elements=2,
        hidden_irreps=o3.Irreps('4x0e + 4x1o'),
        MLP_irreps=o3.Irreps('4x0e'),
        atomic_energies=np.array([-1.25, -2.0], dtype=np.float32),
        avg_num_neighbors=6.0,
        atomic_numbers=[11, 17],
        correlation=2,
        gate=torch.nn.functional.silu,
        pair_repulsion=pair_repulsion,
        distance_transform='None',
        radial_type='bessel',
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
    )


def _make_polar_data() -> dict[str, torch.Tensor]:
    dtype = torch.get_default_dtype()
    cell = 10.0 * torch.eye(3, dtype=dtype).unsqueeze(0)
    return {
        'positions': torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype),
        'node_attrs': torch.eye(2, dtype=dtype),
        'edge_index': torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        'shifts': torch.zeros((2, 3), dtype=dtype),
        'unit_shifts': torch.zeros((2, 3), dtype=dtype),
        'cell': cell,
        'batch': torch.zeros(2, dtype=torch.long),
        'ptr': torch.tensor([0, 2], dtype=torch.long),
        'pbc': torch.tensor([[False, False, False]]),
        'rcell': torch.linalg.inv(cell),
        'volume': torch.linalg.det(cell).abs(),
        'total_charge': torch.zeros(1, dtype=dtype),
        'total_spin': torch.ones(1, dtype=dtype),
        'fermi_level': torch.zeros(1, dtype=dtype),
        'external_field': torch.zeros((1, 3), dtype=dtype),
    }


def _clone_data(data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in data.items()}


class _ConstantPairEnergy(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)

    def forward(
        self,
        lengths: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        del lengths, edge_index, atomic_numbers
        return node_attrs.new_full((node_attrs.shape[0],), self.value)


def test_polar_mace_requires_graph_longrange_dependency():
    if GRAPH_LONGRANGE_AVAILABLE:
        pytest.skip('graph_longrange is installed in this environment')

    with pytest.raises(ImportError, match='graph_longrange'):
        PolarMACE()


def test_polar_mace_constructs_when_graph_longrange_is_available():
    if not GRAPH_LONGRANGE_AVAILABLE:
        pytest.skip('graph_longrange is not installed in this environment')

    model = _make_polar_model()

    assert model.keep_last_layer_irreps is True
    assert len(model.lr_source_maps) == 1
    assert len(model.field_dependent_charges_maps) == 1
    assert model.potential_irreps.dim == 2 * model.field_irreps.dim


def test_polar_mace_forward_uses_pair_repulsion_like_scale_shift_mace():
    if not GRAPH_LONGRANGE_AVAILABLE:
        pytest.skip('graph_longrange is not installed in this environment')

    torch.manual_seed(0)
    model = _make_polar_model(pair_repulsion=True)
    data = _make_polar_data()

    model.pair_repulsion_fn = _ConstantPairEnergy(0.0)
    without_pair = model(
        _clone_data(data),
        compute_force=False,
        compute_node_feats=False,
    )

    model.pair_repulsion_fn = _ConstantPairEnergy(0.25)
    with_pair = model(
        _clone_data(data),
        compute_force=False,
        compute_node_feats=False,
    )

    torch.testing.assert_close(
        with_pair['interaction_energy'] - without_pair['interaction_energy'],
        torch.tensor([0.5], dtype=torch.get_default_dtype()),
    )
    assert with_pair['node_feats'] is None


def test_polar_field_source_blocks_have_expected_shapes():
    irreps = o3.Irreps('2x0e')
    all_node_feats = torch.randn(2, 3, irreps.dim)

    mixer = MultiLayerFeatureMixer(node_feats_irreps=irreps, num_interactions=2)
    mixed = mixer(all_node_feats)
    assert mixed.shape == (3, irreps.dim)

    source = EnvironmentDependentSpinSourceBlock(irreps_in=irreps, max_l=1)
    multipoles = source(mixed)
    expected_irreps = 2 * o3.Irreps.spherical_harmonics(1)
    assert multipoles.shape == (3, 1, expected_irreps.dim)


def test_sparse_uvu_tensor_product_uses_cue_scalar_contraction():
    irreps = o3.Irreps('2x0e + 2x1o')
    out_irreps = o3.Irreps('2x0e')
    instructions = instructions_for_sparse_tp(irreps, irreps, out_irreps)
    sparse = SparseUvuTensorProduct(
        irreps_in1=irreps,
        irreps_in2=irreps,
        irreps_out=out_irreps,
        instructions=instructions,
    )
    assert all('_cg_' not in name for name in sparse.state_dict())
    sparse.weight.data.fill_(1.0)

    x1 = torch.tensor([[0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    x2 = torch.tensor([[0.0, 0.0, 2.0, 1.0, 0.0, -1.0, 1.0, 2.0]])
    x1_vectors = x1[:, 2:].view(1, 2, 3)
    x2_vectors = x2[:, 2:].view(1, 2, 3)
    expected = 0.5 * torch.einsum('bud,bvd->bu', x1_vectors, x2_vectors)
    expected = expected / math.sqrt(3.0)

    torch.testing.assert_close(sparse(x1, x2), expected)
