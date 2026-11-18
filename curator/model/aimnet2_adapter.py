"""Adapt complete AIMNet2 torch modules to Curator without calculator detaches."""
import io
import json
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch
from torch import nn
from curator.data import properties
from curator.model.base import Representation


class AIMNet2Adapter(Representation):
    """A molecular dense-mode adapter; the wrapped model owns its full energy.

    A trusted legacy TorchScript file embeds Coulomb and dispersion. New AIMNet
    artifacts that externalize these terms are deliberately rejected. No atomic
    charge target is read. total_charge is a required molecular condition.
    Use Curator GradientOutput with grad_on_positions=True for conservative forces.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        model_path: Optional[str] = None,
        *,
        complete_energy: bool = False,
        trainable: bool = False,
        trainable_prefixes: Sequence[str] = ("afv", "conv_a", "conv_q", "mlps", "outputs.energy_mlp"),
        energy_reference: Optional[Mapping[int, float]] = None,
        cutoff: Optional[float] = None,
        energy_reference_file: Optional[str] = None,
    ):
        super().__init__()
        if (model is None) == (model_path is None):
            raise ValueError("Specify exactly one of model and model_path.")
        if model_path is not None:
            # torch.jit.load explicitly: ACDB files use .pkl despite being JIT.
            model = torch.jit.load(str(Path(model_path).expanduser()), map_location="cpu")
            outputs = getattr(model, "outputs", None)
            embedded = getattr(outputs, "_modules", {})
            complete_energy = complete_energy or {"lrcoulomb", "dftd3"}.issubset(embedded.keys())
        if not isinstance(model, nn.Module):
            raise TypeError("AIMNet2Adapter requires a torch nn.Module.")
        metadata = getattr(model, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = getattr(model, "_metadata", {})
        if isinstance(metadata, dict) and (
            metadata.get("needs_coulomb", False) or metadata.get("needs_dispersion", False)
        ):
            raise ValueError("Bare AIMNet v2 model lacks external Coulomb/dispersion; supply a complete energy module.")
        if not complete_energy:
            raise ValueError("Declare complete_energy=True only for an assembled full energy model.")
        self.model = model
        self.cutoff = float(cutoff if cutoff is not None else getattr(model, "cutoff", 5.0))
        self.model_outputs = [properties.energy]
        self.trainable_prefixes = tuple(trainable_prefixes)
        if trainable and not self.trainable_prefixes:
            raise ValueError("trainable_prefixes must explicitly select neural parameters.")
        selected = 0
        for name, parameter in model.named_parameters():
            enabled = trainable and any(
                name == prefix or name.startswith(prefix + ".") for prefix in self.trainable_prefixes
            )
            parameter.requires_grad_(enabled)
            selected += int(enabled) * parameter.numel()
        if trainable and selected == 0:
            raise ValueError("No trainable neural parameters matched; a frozen/inlined export cannot be fine-tuned.")
        if energy_reference_file is not None:
            if energy_reference is not None:
                raise ValueError("Specify energy_reference or energy_reference_file, not both.")
            energy_reference = json.loads(Path(energy_reference_file).read_text())["per_element_eV"]
        reference = {int(z): float(e) for z, e in (energy_reference or {}).items()}
        if any(z < 1 or z > 118 for z in reference):
            raise ValueError("Energy-reference atomic numbers must lie in 1..118.")
        table = torch.zeros(119, dtype=torch.float64)
        known = torch.zeros(119, dtype=torch.bool)
        for z, value in reference.items():
            table[z] = value
            known[z] = True
        if not torch.isfinite(table).all():
            raise ValueError("Energy references must be finite.")
        self.register_buffer("energy_reference", table)
        self.register_buffer("reference_known", known)
        self.subtract_reference = bool(reference)

    def _apply(self, fn, recurse=True):
        reference = self.energy_reference
        def preserve_double(tensor):
            moved = fn(tensor)
            if tensor.dtype == torch.float64 and moved.dtype != torch.float64:
                return tensor.to(device=moved.device, dtype=torch.float64)
            return moved
        super()._apply(preserve_double, recurse=recurse)
        self.energy_reference = reference.to(device=self.energy_reference.device, dtype=torch.float64)
        return self

    def __getstate__(self):
        state = super().__getstate__().copy()
        if isinstance(self.model, torch.jit.ScriptModule):
            buffer = io.BytesIO()
            torch.jit.save(self.model, buffer)
            state["_aimnet_jit_bytes"] = buffer.getvalue()
            state["_modules"] = dict(state["_modules"])
            state["_modules"].pop("model")
        return state

    def __setstate__(self, state):
        payload = state.pop("_aimnet_jit_bytes", None)
        super().__setstate__(state)
        if payload is not None:
            self.model = torch.jit.load(io.BytesIO(payload), map_location=self.energy_reference.device)

    def module_groups(self):
        return OrderedDict([("representation", [self.model])])

    def forward(self, data):
        if properties.pbc in data and bool(data[properties.pbc].any()):
            raise ValueError("AIMNet2Adapter dense mode supports isolated molecules only.")
        if properties.total_charge not in data:
            raise ValueError("Explicit total_charge boundary input is required; atomic charges are never used.")
        positions = data[properties.positions]
        if not positions.requires_grad:
            positions.requires_grad_(True)
        numbers = data[properties.atomic_numbers].reshape(-1)
        counts = data[properties.n_atoms].reshape(-1).long()
        if counts.numel() == 0 or bool((counts <= 0).any()) or int(counts.sum()) != len(numbers):
            raise ValueError("n_atoms must describe a nonempty partition of the atoms.")
        charge = data[properties.total_charge].reshape(-1)
        if charge.numel() != counts.numel() or not bool(torch.isfinite(charge).all()):
            raise ValueError("One finite total charge is required per structure.")
        image = data.get(properties.image_idx)
        if image is None:
            image = torch.repeat_interleave(torch.arange(len(counts), device=counts.device), counts)
        if image.shape != numbers.shape or not torch.equal(
            torch.bincount(image.long(), minlength=len(counts)), counts
        ):
            raise ValueError("image_idx and n_atoms disagree.")
        if self.subtract_reference and (
            bool((numbers < 1).any()) or bool((numbers > 118).any())
            or not bool(self.reference_known[numbers].all())
        ):
            raise ValueError("Energy reference is missing a species present in this batch.")
        energies = []
        for graph in range(len(counts)):
            index = torch.nonzero(image == graph, as_tuple=False).reshape(-1)
            model_input = {
                "coord": positions.index_select(0, index).unsqueeze(0),
                "numbers": numbers.index_select(0, index).unsqueeze(0),
                "charge": charge[graph:graph + 1],
            }
            if "spin_multiplicity" in data:
                model_input["mult"] = data["spin_multiplicity"].reshape(-1)[graph:graph + 1]
            output = self.model(model_input)
            energy = output["energy"].reshape(-1)
            if energy.numel() != 1:
                raise ValueError("AIMNet2 energy must contain one scalar per molecule.")
            if self.subtract_reference:
                # Subtract large absolute atomic references before casting down.
                energy = energy.double() - self.energy_reference[numbers[index]].sum()
            energies.append(energy)
        data[properties.energy] = torch.cat(energies)
        return data

