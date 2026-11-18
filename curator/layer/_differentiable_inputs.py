"""Prepare independent geometry-level inputs for scalar-energy derivatives."""

from __future__ import annotations

import torch

from curator.data import properties


class DifferentiableInputs(torch.nn.Module):
    """Connect per-geometry coordinate, uniform-field and strain views.

    Place this module after neighbor-list construction and before the GNN. Atom
    rows must be grouped by geometry according to ``n_atoms``. The saved private
    tuples contain the exact live views used to construct subsequent model
    inputs; slicing a tensor only after computing energy would not create such
    derivative paths.

    Positions are always prepared. ``uniform_field=True`` prepares one Cartesian
    three-vector per geometry under ``external_field``; any local per-atom field
    remains a separate caller-managed input. ``strain=True`` adds a zero-valued
    Cartesian deformation tensor per geometry, transforming row-vector positions,
    cell vectors and periodic image shifts by ``(I + strain).T``. The neighbor
    topology stays fixed during these local derivatives.

    ``_observable_positions`` refers to the unstrained coordinates. Uniform fields
    are laboratory-frame inputs and are not automatically deformed. Volumes are
    computed from the original cells. No property-specific readout is selected.
    """

    def __init__(self, uniform_field: bool = False, strain: bool = False) -> None:
        super().__init__()
        self.uniform_field = bool(uniform_field)
        self.strain = bool(strain)

    def forward(self, data):
        if torch.is_inference_mode_enabled():
            raise RuntimeError("differentiable inputs require inference_mode=False")
        positions = data[properties.positions]
        if not positions.is_floating_point() or positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must be a floating-point [N,3] tensor")
        if positions.dtype not in (torch.float32, torch.float64):
            raise ValueError("differentiable coordinates require float32 or float64")
        # Geometric linalg must not be lowered to bfloat16 by GNN autocast.
        with torch.enable_grad(), torch.autocast(device_type=positions.device.type, enabled=False):
            return self._prepare(data, positions)

    def _prepare(self, data, positions):
        if "_observable_positions" in data:
            raise ValueError("DifferentiableInputs must be applied once to an unprepared batch")
        counts_tensor = data[properties.n_atoms]
        if counts_tensor.ndim != 1 or counts_tensor.dtype not in (torch.int32, torch.int64):
            raise ValueError("n_atoms must be a one-dimensional integer tensor")
        counts = counts_tensor.tolist()
        if not counts or any(count <= 0 for count in counts) or sum(counts) != positions.shape[0]:
            raise ValueError("n_atoms must contain positive counts summing to the position count")
        n_geometries = len(counts)
        images = torch.repeat_interleave(
            torch.arange(n_geometries, device=positions.device),
            counts_tensor.to(device=positions.device),
        )
        if properties.image_idx in data and not torch.equal(data[properties.image_idx], images):
            raise ValueError("atoms must be grouped by geometry consistently with n_atoms and image_idx")
        data[properties.image_idx] = images
        if properties.edge_idx not in data:
            raise ValueError("construct the neighbor list before DifferentiableInputs")
        edges = data[properties.edge_idx]
        if edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype not in (torch.int32, torch.int64):
            raise ValueError("edge_idx must have shape [n_edges,2] and integer dtype")
        if edges.numel() and (bool((edges < 0).any()) or bool((edges >= len(positions)).any())):
            raise ValueError("edge_idx contains an invalid atom index")
        edge_images = images[edges[:, 0]]
        if not torch.equal(edge_images, images[edges[:, 1]]):
            raise ValueError("neighbor edges cannot connect different geometries")

        if properties.cell_displacements in data:
            shifts = data[properties.cell_displacements].to(positions)
        elif properties.edge_diff in data:
            # The neighbor backend has already selected periodic images. Recover
            # their fixed reference translations before applying any strain.
            reference_pairs = positions.detach()[edges[:, 1]] - positions.detach()[edges[:, 0]]
            shifts = data[properties.edge_diff].detach().to(positions) - reference_pairs
        else:
            if properties.cell in data and (
                properties.pbc not in data or bool(data[properties.pbc].any())
            ):
                raise ValueError("periodic edges need edge_diff or explicit cell_displacements")
            shifts = positions.new_zeros((edges.shape[0], 3))
        if shifts.shape != (edges.shape[0], 3):
            raise ValueError("periodic image shifts must have shape [n_edges,3]")

        if not positions.requires_grad:
            positions.requires_grad_(True)
        position_parts = tuple(positions.split(counts))
        data["_observable_positions"] = position_parts

        if self.uniform_field:
            field = data.get(properties.external_field)
            if field is None:
                field = positions.new_zeros((n_geometries, 3), requires_grad=True)
            else:
                if not isinstance(field, torch.Tensor) or not field.is_floating_point():
                    raise ValueError("external_field must be a floating-point [n_geometries,3] tensor")
                if field.shape != (n_geometries, 3):
                    raise ValueError("uniform external_field must have shape [n_geometries,3]; keep per-atom fields separately")
                field = field.to(positions)
                if not field.requires_grad:
                    field.requires_grad_(True)
            field_parts = tuple(field.unbind(0))
            data["_observable_fields"] = field_parts
            # This also connects consumers that read the public batched tensor
            # instead of directly using one of the saved geometry views.
            data[properties.external_field] = torch.stack(field_parts)

        cell = None
        if properties.cell in data:
            original_cell = data[properties.cell]
            if original_cell.shape not in ((3 * n_geometries, 3), (n_geometries, 3, 3)):
                raise ValueError("cell must contain one [3,3] matrix per geometry")
            cell = original_cell.to(positions).reshape(n_geometries, 3, 3)
            data["_observable_volumes"] = torch.linalg.det(cell).abs()
        if self.strain:
            if cell is None:
                raise ValueError("strain preparation requires cells")
            if properties.strain in data:
                raise ValueError("strain is created here; do not chain another strain transform")
            volumes = data["_observable_volumes"]
            if not bool(torch.isfinite(volumes).all()) or bool((volumes <= 0).any()):
                raise ValueError("strain preparation requires finite nonzero cell volumes")
            strain = positions.new_zeros((n_geometries, 3, 3), requires_grad=True)
            strain_parts = tuple(strain.unbind(0))
            data[properties.strain] = strain
            data["_observable_strains"] = strain_parts
            identity = torch.eye(3, dtype=positions.dtype, device=positions.device)
            deformations = torch.stack([identity + part for part in strain_parts])
            data[properties.positions] = torch.cat([
                part @ deformations[index].T for index, part in enumerate(position_parts)
            ])
            data[properties.cell] = torch.matmul(cell, deformations.transpose(1, 2)).reshape(original_cell.shape)
            shifts = torch.bmm(shifts.unsqueeze(1), deformations[edge_images].transpose(1, 2)).squeeze(1)
        else:
            data[properties.positions] = torch.cat(position_parts)

        # Rebuild even without strain, so GNN distances depend on the saved
        # coordinate views. In particular, periodic self-image edges retain the
        # strain dependence of their image translation.
        data[properties.cell_displacements] = shifts
        current_positions = data[properties.positions]
        data[properties.edge_diff] = current_positions[edges[:, 1]] - current_positions[edges[:, 0]] + shifts
        data[properties.edge_dist] = torch.linalg.vector_norm(data[properties.edge_diff], dim=-1)
        return data
