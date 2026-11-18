"""DPA4C periodic RSA ablations and the fixed-cell AuMgO joint CUDA graph."""

import math

import torch
from torch import nn
from curator.data import properties as p
from curator.layer.efa.reciprocal_rsa import PeriodicRSA
from curator.layer.efa.reciprocal_rsa_channels import ChannelSpectrumRSA
from graph_joint import JointGraph, eligible
from qefa_cuda import MolecularFactors


class BatchedRSA:
    """Factorize the periodic bilinear operator over atoms, with live Q/K/gain gradients."""

    def select_training_cell(self, cells):
        # Only a fixed-cell E/F path is specialized. Generic native inference
        # below uses the actual cell each time, also for deformations/supercells.
        cells = cells.reshape(-1, 3, 3)
        if cells.requires_grad or not torch.equal(cells, cells[:1].expand_as(cells)):
            raise ValueError(
                "AuMgO graph requires one fixed cell without stress labels"
            )
        cell = cells[0]
        previous = getattr(self, "_training_cell", None)
        if previous is not None:
            if (
                previous.device != cell.device
                or previous.dtype != cell.dtype
                or not torch.equal(previous, cell)
            ):
                raise ValueError(
                    "AuMgO training cells changed; use the generic periodic path"
                )
            return
        waves, weights = self._explicit_modes(cell)
        half = len(waves) // 2
        if half == 0:
            raise ValueError("AuMgO cutoff includes no nonzero reciprocal modes")
        self._training_cell = cell.detach().clone()
        self._training_waves = waves[:half].detach()
        self._training_weights = (2 * weights[:half].repeat(2)).detach()

    def prepare_periodic_batch(self, query, key, positions, cells):
        if not (query.is_cuda and torch.cuda.is_current_stream_capturing()):
            self.select_training_cell(cells)
        batch, atoms, _ = positions.shape
        angular, channels = query.shape[-2:]
        phase = (positions - positions.mean(1, keepdim=True)) @ self._training_waves.T
        harmonics = torch.cat((phase.cos(), phase.sin()), -1)
        weighted = harmonics * (self._training_weights / math.sqrt(angular * channels))
        q = weighted[..., None, None] * query.reshape(
            batch, atoms, 1, angular, channels
        )
        if isinstance(self, ChannelSpectrumRSA):
            gain = self.log_gains(self._training_waves.norm(dim=-1)).exp().repeat(2, 1)
            gain = gain.repeat_interleave(channels // gain.shape[-1], dim=-1)
            q = q * gain[:, None, :]
        k = harmonics[..., None] * key.reshape(batch, atoms, 1, angular * channels)
        return MolecularFactors(q.flatten(2), k.flatten(2))


class SharedBatchedRSA(BatchedRSA, PeriodicRSA):
    pass


class SpectrumBatchedRSA(BatchedRSA, ChannelSpectrumRSA):
    pass


class ConstantBatchedRSA(SpectrumBatchedRSA):
    def log_gains(self, radius):
        return self.group_log_gain.expand(radius.shape[0], -1)


def install_aumgo_arm(model, arm):
    """Call on the RSA model *before* constructing the optimizer, for training and reload."""
    if arm not in {"A0", "A1", "A2", "A3", "A4"}:
        raise ValueError(f"Unknown AuMgO arm {arm}")
    head = model.output_modules[0]
    attention = head.internal_energy.attention
    if not isinstance(attention, PeriodicRSA):
        raise TypeError("Build DPA4C with periodic_backend='rsa' first")
    if arm in ("A0", "A1"):
        attention.__class__ = SharedBatchedRSA
    else:
        groups = 1 if arm == "A2" else 4
        attention = ChannelSpectrumRSA.from_rsa(attention, groups)
        if arm == "A4":
            attention.__class__ = SpectrumBatchedRSA
        else:
            del attention.band_coefficients
            attention.__class__ = ConstantBatchedRSA
            attention.group_log_gain = nn.Parameter(
                attention.frequencies.new_zeros(groups)
            )
        head.internal_energy.attention = attention
    return model


class AuMgOGraph(JointGraph):
    """Refresh image displacements on each replay for the fixed periodic slab."""

    @staticmethod
    def eligible(batch):
        return eligible(batch, periodic=True) and bool((batch[p.n_atoms] == 110).all())

    def _inputs(self, batch):
        data, inputs = super()._inputs(batch)
        cells = data[p.cell].reshape(-1, 3, 3)
        self.trainer.model.output_modules[
            0
        ].internal_energy.attention.select_training_cell(cells)
        edges = data[p.edge_idx]
        if p.cell_displacements in data:
            shifts = data[p.cell_displacements]
        else:
            pos = data[p.positions].detach()
            shifts = data[p.edge_diff].detach() - (pos[edges[:, 1]] - pos[edges[:, 0]])
        length = len(inputs[3])
        shifts = torch.nn.functional.pad(shifts, (0, 0, 0, length - len(edges)))
        return data, (*inputs, shifts)

    @staticmethod
    def _edge_vectors(pos, edges, mask, outside, shifts):
        return torch.where(mask, pos[edges[:, 1]] - pos[edges[:, 0]] + shifts, outside)
