from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from curator.data import properties
from curator.layer import FeatureProjection, ProjectedElementEmbedding
from curator.model import Painn
from curator.train.distill import prepare_teacher_model_for_offline_distillation


def test_feature_projection_writes_requested_output():
    module = FeatureProjection(4, 7)
    data = {properties.node_final_feature: torch.randn(3, 4)}

    assert module(data)[properties.node_feature_distill].shape == (3, 7)
    assert module.model_outputs == [properties.node_feature_distill]


def test_projected_element_embedding_loads_frozen_table(tmp_path):
    path = tmp_path / "teacher_embeddings.npy"
    np.save(path, np.arange(30, dtype=np.float32).reshape(10, 3))
    module = ProjectedElementEmbedding(path, out_features=5, index_offset=1)

    output = module(torch.tensor([0, 5]))
    output.sum().backward()

    assert output.shape == (2, 5)
    assert "embeddings" not in dict(module.named_parameters())
    assert module.projection.weight.grad is not None


def test_painn_export_preserves_injected_embedding():
    embedding = ProjectedElementEmbedding(torch.randn(10, 6), 8, index_offset=1)
    source = Painn(
        num_interactions=1,
        num_features=8,
        cutoff=5.0,
        num_elements=9,
        atomic_number_offset=1,
        atom_embedding=embedding,
    )

    rebuilt = Painn(**source.export_init_kwargs())

    assert isinstance(rebuilt.atom_embedding, ProjectedElementEmbedding)
    assert rebuilt.atom_embedding.index_offset == 1
    assert torch.equal(rebuilt.atom_embedding.embeddings, embedding.embeddings)


def test_node_features_round_trip_through_sqlite(tmp_path):
    apsw = pytest.importorskip("apsw")
    from curator.data.sql_database import QMDatabase, STANDARD_COLUMN_SPECS

    path = tmp_path / "features.sqlite"
    extra_columns = {
        properties.teacher_node_features: dict(
            STANDARD_COLUMN_SPECS[properties.node_final_feature]
        )
    }
    db = QMDatabase(
        str(path),
        flags=apsw.SQLITE_OPEN_READWRITE | apsw.SQLITE_OPEN_CREATE,
        extra_columns=extra_columns,
    )
    features = np.arange(15, dtype=np.float32).reshape(3, 5)
    db.add_data(
        {
            properties.atomic_numbers: np.array([1, 6, 8], dtype=np.int32),
            properties.positions: np.zeros((3, 3), dtype=np.float32),
            properties.pbc: False,
            properties.teacher_node_features: features,
        }
    )

    loaded = db[0]

    assert loaded[properties.teacher_node_features].shape == (3, 5)
    np.testing.assert_array_equal(
        loaded[properties.teacher_node_features], features
    )


def test_offline_feature_distillation_exposes_teacher_feature():
    class Teacher:
        model_outputs = [properties.energy]

    teacher = Teacher()
    prepare_teacher_model_for_offline_distillation(
        teacher,
        {properties.node_final_feature: properties.teacher_node_features},
    )

    assert properties.node_final_feature in teacher.model_outputs


def test_node_feature_distillation_config_composes():
    config_dir = str(
        (Path(__file__).parents[1] / "curator" / "configs").resolve()
    )
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(
            config_name="train",
            overrides=["task/outputs=energy_force_node_feature_distill"],
        )
        atom_config = compose(
            config_name="train",
            overrides=[
                "+model/atom_embedding@model.representation.atom_embedding=projected_teacher",
                "model.representation.atom_embedding.embeddings=/tmp/teacher.npy",
            ],
        )

    assert (
        config.model.output_modules.node_feature_projection._target_
        == "curator.layer.FeatureProjection"
    )
    assert (
        config.task.outputs.node_feature_distill.teacher_output_property
        == properties.node_final_feature
    )
    assert (
        atom_config.model.representation.atom_embedding._target_
        == "curator.layer.ProjectedElementEmbedding"
    )
