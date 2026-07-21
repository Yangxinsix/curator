from types import SimpleNamespace

import torch

from curator.model.adapter.utils import (
    ExternalModelSpec,
    format_external_model_spec,
    parse_external_model_spec,
)
from curator.model.lit_module import LitNNP


def test_external_model_spec_format_round_trip():
    spec = ExternalModelSpec("matgl", "M3GNet", {"backend": "PYG", "head": "0"})
    rendered = format_external_model_spec(spec)
    assert parse_external_model_spec(rendered) == spec


def test_external_state_dict_checkpoint_omits_model_object(monkeypatch):
    task = object.__new__(LitNNP)
    torch.nn.Module.__init__(task)
    task.config = SimpleNamespace(data={"name": "data"}, model={"name": "model"})
    task.model = torch.nn.Linear(2, 1)
    task.model._curator_external_model_spec = "matgl:M3GNet"
    task.model._curator_checkpoint_save_mode = "state_dict"
    task.outputs = torch.nn.ModuleList()
    task.optimizer = None
    task.save_entire_model = True
    monkeypatch.setattr(
        "curator.model.lit_module.get_model_wrapper_config",
        lambda model: SimpleNamespace(to_dict=lambda: {}),
    )

    checkpoint = {}
    task.on_save_checkpoint(checkpoint)

    assert checkpoint["external_model_spec"] == "matgl:M3GNet"
    assert checkpoint["model_save_mode"] == "state_dict"
    assert "model" not in checkpoint
