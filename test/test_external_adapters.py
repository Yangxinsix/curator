from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch
from torch import nn

from curator.data import properties
from curator.model import adapters as adapter_shim
from curator.model.adapter import utils as adapter_utils
from curator.model.adapter.allegro import AllegroAdapter
from curator.model.adapter.esen import ESENAdapter
from curator.layer.feature.extractor import FeatureExtractor


def _sample_batch(include_edges: bool = True, include_cell: bool = True):
    data = {
        properties.n_atoms: torch.tensor([2], dtype=torch.long),
        properties.atomic_numbers: torch.tensor([1, 8], dtype=torch.long),
        properties.positions: torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]],
            dtype=torch.float32,
        ),
        properties.pbc: torch.tensor([[False, False, False]], dtype=torch.bool),
    }
    if include_cell:
        data[properties.cell] = torch.eye(3, dtype=torch.float32).view(1, 3, 3)
    if include_edges:
        data[properties.edge_idx] = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        data[properties.edge_diff] = torch.tensor(
            [[0.0, 0.0, 0.9], [0.0, 0.0, -0.9]],
            dtype=torch.float32,
        )
        data[properties.edge_dist] = torch.tensor([0.9, 0.9], dtype=torch.float32)
        data[properties.cell_displacements] = torch.zeros((2, 3), dtype=torch.float32)
    return data


def test_external_model_shim_dispatch(monkeypatch):
    loaders = dict(adapter_utils._ADAPTER_LOADERS)
    monkeypatch.setitem(adapter_utils._ADAPTER_LOADERS, "dummy", lambda spec, device=None: (spec.resource, device))
    assert adapter_shim.parse_external_model_spec("dummy:model?x=1").resource == "model"
    assert adapter_shim.is_external_model_spec("dummy:model")
    assert adapter_shim.load_external_model("dummy:model", device="cpu") == ("model", "cpu")
    adapter_utils._ADAPTER_LOADERS.clear()
    adapter_utils._ADAPTER_LOADERS.update(loaders)


def test_split_batch_structures_and_cell_offsets():
    batch = _sample_batch(include_edges=True, include_cell=True)
    structs = adapter_utils.split_batch_structures(batch)
    assert len(structs) == 1
    struct = structs[0]
    assert struct.edge_index is not None
    offsets = adapter_utils.infer_cell_offsets(
        struct.edge_index,
        struct.positions,
        struct.cell,
        edge_diff=struct.edge_diff,
        cell_displacements=struct.cell_displacements,
    )
    assert offsets.shape == (2, 3)
    assert torch.equal(offsets, torch.zeros_like(offsets))


def test_allegro_adapter_direct_and_fallback(monkeypatch):
    calls = {"compute_neighborlist": 0}

    fake_atomic_data_dict = SimpleNamespace(
        POSITIONS_KEY="pos",
        ATOMIC_NUMBERS_KEY="atomic_numbers",
        ATOM_TYPE_KEY="atom_types",
        PBC_KEY="pbc",
        CELL_KEY="cell",
        EDGE_INDEX_KEY="edge_index",
        EDGE_VECTORS_KEY="edge_vectors",
        EDGE_LENGTH_KEY="edge_lengths",
        batched_from_list=lambda frames: {"frames": frames},
    )

    def fake_from_dict(frame):
        return dict(frame)

    def fake_compute_neighborlist_(data, cutoff, backend="matscipy"):
        calls["compute_neighborlist"] += 1
        data["edge_index"] = torch.zeros((2, 0), dtype=torch.long)
        return data

    nequip_module = ModuleType("nequip")
    nequip_data = ModuleType("nequip.data")
    nequip_nl = ModuleType("nequip.data._nl")
    nequip_data.AtomicDataDict = fake_atomic_data_dict
    nequip_data.from_dict = fake_from_dict
    nequip_nl.compute_neighborlist_ = fake_compute_neighborlist_
    monkeypatch.setitem(sys.modules, "nequip", nequip_module)
    monkeypatch.setitem(sys.modules, "nequip.data", nequip_data)
    monkeypatch.setitem(sys.modules, "nequip.data._nl", nequip_nl)

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.edge_readout = nn.Linear(4, 1)
            self.type_names = ["H", "O"]
            self.edge_norm = SimpleNamespace(r_max=5.0)

        def forward(self, batch):
            self.last_batch = batch
            return batch

    adapter = AllegroAdapter(DummyModel())
    direct = adapter._build_batch(_sample_batch(include_edges=True))
    assert torch.equal(direct["frames"][0]["atom_types"], torch.tensor([0, 1], dtype=torch.long))
    assert direct["frames"][0]["edge_index"].shape == (2, 2)
    assert calls["compute_neighborlist"] == 0

    adapter._build_batch(_sample_batch(include_edges=False))
    assert calls["compute_neighborlist"] == 1


def test_esen_adapter_direct_and_fallback(monkeypatch):
    calls = {"from_ase": 0}

    class FakeAtomicData:
        def __init__(self, payload):
            self.payload = payload

        @classmethod
        def from_dict(cls, payload):
            return cls(payload)

        @classmethod
        def from_ase(cls, atoms, **kwargs):
            calls["from_ase"] += 1
            return cls({"atoms": atoms, **kwargs})

    fairchem_atomic = ModuleType("fairchem.core.datasets.atomic_data")
    fairchem_atomic.AtomicData = FakeAtomicData
    fairchem_atomic.atomicdata_list_to_batch = lambda frames: {"frames": frames}
    monkeypatch.setitem(sys.modules, "fairchem", ModuleType("fairchem"))
    monkeypatch.setitem(sys.modules, "fairchem.core", ModuleType("fairchem.core"))
    monkeypatch.setitem(sys.modules, "fairchem.core.datasets", ModuleType("fairchem.core.datasets"))
    monkeypatch.setitem(sys.modules, "fairchem.core.datasets.atomic_data", fairchem_atomic)

    class DummyInner(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = SimpleNamespace(cutoff=6.0)
            self.energy_block = nn.Linear(4, 1)

        def on_predict_check(self, batch):
            return None

    class DummyWrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.module = DummyInner()

        def forward(self, batch):
            return batch

    predict_unit = SimpleNamespace(
        model=DummyWrapped(),
        dataset_to_tasks={"omol": [SimpleNamespace(property="energy")]},
        lazy_model_intialized=True,
        device="cpu",
        inference_settings=SimpleNamespace(base_precision_dtype=torch.float32),
    )

    adapter = ESENAdapter(predict_unit=predict_unit, task_name="omol")
    direct = adapter._build_batch(_sample_batch(include_edges=True))
    frame = direct["frames"][0].payload
    assert frame["edge_index"].shape == (2, 2)
    assert torch.equal(frame["edge_index"], torch.tensor([[1, 0], [0, 1]], dtype=torch.long))
    assert frame["cell_offsets"].shape == (2, 3)
    assert calls["from_ase"] == 0

    adapter._build_batch(_sample_batch(include_edges=False))
    assert calls["from_ase"] == 1


def test_feature_extractor_recognizes_nequip_scalar_linear(monkeypatch):
    class FakeScalarLinearLayer(nn.Module):
        pass

    fake_nequip = ModuleType("nequip")
    fake_nequip_nn = ModuleType("nequip.nn")
    fake_nequip_mlp = ModuleType("nequip.nn.mlp")
    fake_nequip_mlp.ScalarLinearLayer = FakeScalarLinearLayer
    monkeypatch.setitem(sys.modules, "nequip", fake_nequip)
    monkeypatch.setitem(sys.modules, "nequip.nn", fake_nequip_nn)
    monkeypatch.setitem(sys.modules, "nequip.nn.mlp", fake_nequip_mlp)
    monkeypatch.setattr(
        "curator.layer.feature.extractor.util.find_spec",
        lambda name: object() if name == "nequip" else None,
    )

    linear_types = FeatureExtractor._resolve_linear_types()
    assert FakeScalarLinearLayer in linear_types
