import torch
from typing import Optional, List, Callable, Union, Tuple

from curator.data import properties


def get_jacobian(
    forces: torch.Tensor,
    positions: torch.Tensor,
    grad_outputs: torch.Tensor,
    create_graph: bool = False,
    vectorize: bool = True,
) -> torch.Tensor:
    def compute_grad(grad_output: torch.Tensor) -> torch.Tensor:
        grad = torch.autograd.grad(
            outputs=forces,
            inputs=positions,
            grad_outputs=grad_output,
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        if grad is None:
            raise RuntimeError("EnergyHessianOutput received `forces` that are not differentiable with respect to `positions`.")
        return grad

    if vectorize and hasattr(torch, "vmap"):
        if grad_outputs.dim() == 4:
            return torch.vmap(torch.vmap(compute_grad))(grad_outputs)
        return torch.vmap(compute_grad)(grad_outputs)

    if grad_outputs.dim() == 4:
        jacobian = torch.zeros(
            grad_outputs.shape[0],
            grad_outputs.shape[1],
            positions.shape[0],
            positions.shape[1],
            device=positions.device,
            dtype=positions.dtype,
        )
        for i in range(grad_outputs.shape[0]):
            for j in range(grad_outputs.shape[1]):
                jacobian[i, j] = compute_grad(grad_outputs[i, j])
        return jacobian

    jacobian = torch.zeros(
        grad_outputs.shape[0],
        positions.shape[0],
        positions.shape[1],
        device=positions.device,
        dtype=positions.dtype,
    )
    for i in range(grad_outputs.shape[0]):
        jacobian[i] = compute_grad(grad_outputs[i])
    return jacobian


def get_full_energy_hessian(
    forces: torch.Tensor,
    positions: torch.Tensor,
    create_graph: bool = False,
    vectorize: bool = True,
) -> torch.Tensor:
    num_atoms = positions.shape[0]
    grad_outputs = torch.zeros(
        num_atoms,
        3,
        num_atoms,
        3,
        device=positions.device,
        dtype=positions.dtype,
    )
    idx = torch.arange(num_atoms, device=positions.device)
    grad_outputs[idx, :, idx, :] = torch.eye(3, device=positions.device, dtype=positions.dtype).unsqueeze(0).expand(num_atoms, -1, -1)
    return -get_jacobian(
        forces,
        positions,
        grad_outputs,
        create_graph=create_graph,
        vectorize=vectorize,
    )


def sample_hessian_rows(
    forces: torch.Tensor,
    positions: torch.Tensor,
    n_atoms: torch.Tensor,
    num_samples: int,
    mask: Optional[torch.Tensor] = None,
    create_graph: bool = False,
    vectorize: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Sample Hessian rows for each structure in a batch.

    Returns:
        A tuple ``(samples_per_structure, rows_per_structure)`` where each list entry
        corresponds to one structure in the batch.

        - ``samples_per_structure[i]`` has shape ``[num_samples_i, 2]`` and stores
          ``(row_atom, row_component)`` pairs.
        - ``rows_per_structure[i]`` has shape ``[num_samples_i, num_kept_atoms_i, 3]``.
          If ``mask`` is not provided, this is ``[num_samples_i, num_atoms_i, 3]``.
    """
    total_num_atoms = positions.shape[0]
    if mask is None:
        mask = torch.ones(total_num_atoms, dtype=torch.bool, device=positions.device)
    atom_offsets = [0] + torch.cumsum(n_atoms, 0).tolist()
    grad_outputs = torch.zeros(
        num_samples,
        total_num_atoms,
        3,
        device=positions.device,
        dtype=positions.dtype,
    )
    samples_per_structure = sample_hessian_indices(
        n_atoms,
        num_samples=num_samples,
        mask=mask,
    )
    for i, samples in enumerate(samples_per_structure):
        offset_samples = samples.clone()
        offset_samples[:, 0] += atom_offsets[i]
        # Each structure writes into a disjoint atom block, so different structures can
        # share the same sample dimension without mixing their Hessian rows.
        grad_outputs[torch.arange(samples.shape[0], device=positions.device), offset_samples[:, 0], offset_samples[:, 1]] = 1

    jacobian = -get_jacobian(
        forces,
        positions,
        grad_outputs,
        create_graph=create_graph,
        vectorize=vectorize,
    )
    rows_per_structure: List[torch.Tensor] = []
    for i, num_atoms_i in enumerate(n_atoms.tolist()):
        structure_mask = mask[atom_offsets[i]:atom_offsets[i + 1]]
        sampled_rows = jacobian[: samples_per_structure[i].shape[0], atom_offsets[i]:atom_offsets[i + 1], :]
        rows_per_structure.append(sampled_rows[:, structure_mask, :])
    return samples_per_structure, rows_per_structure


def sample_hessian_projections(
    forces: torch.Tensor,
    positions: torch.Tensor,
    num_probes: int,
    n_atoms: Optional[torch.Tensor] = None,
    probe_vectors: Optional[torch.Tensor] = None,
    create_graph: bool = False,
    vectorize: bool = True,
    normalize_probes: bool = False,
    probe_distribution: str = "gaussian",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project the energy Hessian along dense probe vectors.

    ``forces = -dE/dR``, so ``-d(forces)/dR @ probe`` gives ``H_E @ probe``.
    The returned tensors both have shape ``[num_probes, n_atoms, 3]``.
    """
    if probe_vectors is None:
        shape = (int(num_probes), positions.shape[0], positions.shape[1])
        if str(probe_distribution).lower() in {"rademacher", "sign"}:
            probe_vectors = torch.empty(shape, device=positions.device, dtype=positions.dtype)
            probe_vectors.bernoulli_(0.5).mul_(2).sub_(1)
        else:
            probe_vectors = torch.randn(shape, device=positions.device, dtype=positions.dtype)
    else:
        probe_vectors = probe_vectors.to(device=positions.device, dtype=positions.dtype)
        if probe_vectors.dim() != 3 or probe_vectors.shape[1:] != positions.shape:
            raise ValueError(
                "Energy Hessian probe vectors must have shape "
                f"[num_probes, {positions.shape[0]}, {positions.shape[1]}], got {tuple(probe_vectors.shape)}."
            )

    if normalize_probes:
        if n_atoms is None:
            denom = probe_vectors.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1)
            probe_vectors = probe_vectors / denom
        else:
            probe_vectors = probe_vectors.clone()
            offset = 0
            for count in n_atoms.tolist():
                stop = offset + int(count)
                block = probe_vectors[:, offset:stop]
                denom = block.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1)
                probe_vectors[:, offset:stop] = block / denom
                offset = stop

    projected = -get_jacobian(
        forces,
        positions,
        probe_vectors,
        create_graph=create_graph,
        vectorize=vectorize,
    )
    return probe_vectors, projected


def project_hessian(
    energy_hessian: torch.Tensor,
    probe_vectors: torch.Tensor,
) -> torch.Tensor:
    """Apply a full, possibly block-diagonal Hessian to dense probes."""
    if energy_hessian.dim() != 4:
        raise ValueError(
            "Energy Hessian must have shape [n_atoms, 3, n_atoms, 3], "
            f"got {tuple(energy_hessian.shape)}."
        )
    if probe_vectors.dim() != 3:
        raise ValueError(
            "Energy Hessian probes must have shape [num_probes, n_atoms, 3], "
            f"got {tuple(probe_vectors.shape)}."
        )
    expected = (energy_hessian.shape[2], energy_hessian.shape[3])
    if tuple(probe_vectors.shape[1:]) != expected:
        raise ValueError(
            "Energy Hessian and probe dimensions do not match: expected probes "
            f"with trailing shape {expected}, got {tuple(probe_vectors.shape[1:])}."
        )
    return torch.einsum("aibj,kbj->kai", energy_hessian, probe_vectors)


def sample_hessian_indices(
    n_atoms: torch.Tensor,
    num_samples: int,
    mask: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    total_num_atoms = int(n_atoms.sum().item())
    if mask is None:
        mask = torch.ones(total_num_atoms, dtype=torch.bool, device=n_atoms.device)
    atom_offsets = [0] + torch.cumsum(n_atoms, 0).tolist()
    samples_per_structure: List[torch.Tensor] = []
    for i, num_atoms_i in enumerate(n_atoms.tolist()):
        structure_mask = mask[atom_offsets[i]:atom_offsets[i + 1]]
        valid_rows = torch.where(structure_mask)[0]
        if valid_rows.numel() == 0:
            raise ValueError(f"Cannot sample Hessian rows for structure {i}: mask selects no atoms.")
        valid_row_indices = valid_rows.repeat_interleave(3) * 3 + torch.arange(3, device=mask.device).repeat(valid_rows.numel())
        chosen_row_indices = valid_row_indices[
            torch.randperm(valid_row_indices.numel(), device=mask.device)[: min(num_samples, valid_row_indices.numel())]
        ]
        samples_per_structure.append(
            torch.stack((chosen_row_indices // 3, chosen_row_indices % 3), dim=1)
        )
    return samples_per_structure


def gather_hessian_rows(
    energy_hessian: torch.Tensor,
    n_atoms: torch.Tensor,
    row_indices: List[torch.Tensor],
    mask: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    total_num_atoms = energy_hessian.shape[0]
    if mask is None:
        mask = torch.ones(total_num_atoms, dtype=torch.bool, device=energy_hessian.device)
    atom_offsets = [0] + torch.cumsum(n_atoms, 0).tolist()
    rows_per_structure: List[torch.Tensor] = []
    for i, num_atoms_i in enumerate(n_atoms.tolist()):
        start = atom_offsets[i]
        stop = atom_offsets[i + 1]
        structure_mask = mask[start:stop]
        samples = row_indices[i]
        rows = energy_hessian[start + samples[:, 0], samples[:, 1], start:stop, :]
        rows_per_structure.append(rows[:, structure_mask, :])
    return rows_per_structure


class EnergyHessianSample(torch.nn.Module):
    def __init__(self, mask_key: Optional[str] = None) -> None:
        super().__init__()
        self.mask_key = mask_key

    def forward(
        self,
        data: properties.Type,
        key: str,
        indices: Optional[List[torch.Tensor]] = None,
        num_samples: Optional[int] = None,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        value = data[key]
        if isinstance(value, list):
            return value, indices
        if not torch.is_tensor(value):
            raise TypeError(
                f"EnergyHessianSample expected tensor or list input for '{key}', "
                f"got {type(value).__name__}."
            )
        if properties.n_atoms not in data:
            raise KeyError("EnergyHessianSample requires `n_atoms` in the batch.")
        mask = data[self.mask_key] if self.mask_key is not None else None
        if indices is None:
            if num_samples is None:
                return value, None
            indices = sample_hessian_indices(
                data[properties.n_atoms],
                num_samples=num_samples,
                mask=mask,
            )
        return gather_hessian_rows(
            value,
            data[properties.n_atoms],
            indices,
            mask=mask,
        ), indices


class EnergyHessianOutput(torch.nn.Module):
    def __init__(
        self,
        vectorize: bool = True,
        num_samples: Optional[int] = None,
        num_probes: Optional[int] = None,
        mask_key: Optional[str] = None,
        probe_key: str = properties.energy_hessian_probe_vectors,
        normalize_probes: bool = False,
        probe_distribution: str = "gaussian",
        only_train: bool = False,
        model_outputs: Optional[List[str]] = None,
        update_callback: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.vectorize = vectorize
        self.num_samples = num_samples
        self.num_probes = num_probes
        self.mask_key = mask_key
        self.probe_key = probe_key
        self.normalize_probes = bool(normalize_probes)
        self.probe_distribution = str(probe_distribution)
        self.only_train = bool(only_train)
        self.update_callback = update_callback
        self.model_outputs = model_outputs if model_outputs is not None else [properties.energy_hessian]

    @torch.jit.ignore
    def update_model_outputs(self, outputs: Union[List[str], str]):
        if isinstance(outputs, str):
            self.model_outputs.append(outputs)
        else:
            self.model_outputs.extend(outputs)
        if self.update_callback:
            self.update_callback()

    def forward(
        self,
        data: properties.Type,
        training: Optional[bool] = None,
    ) -> properties.Type:
        if self.only_train and not self.training:
            zero = data[properties.positions].new_zeros(())
            for key in self.model_outputs:
                data[key] = zero
            return data
        if (
            properties.energy_hessian not in self.model_outputs
            and properties.energy_hessian_sampled not in self.model_outputs
            and properties.energy_hessian_sample_indices not in self.model_outputs
            and properties.energy_hessian_projected not in self.model_outputs
            and properties.energy_hessian_probe_vectors not in self.model_outputs
        ):
            return data
        if properties.forces not in data:
            raise KeyError(
                "EnergyHessianOutput requires `forces` in the batch. Add a "
                "force-producing output before EnergyHessianOutput."
            )
        if properties.positions not in data:
            raise KeyError("EnergyHessianOutput requires `positions` in the batch.")
        forces = data[properties.forces]
        positions = data[properties.positions]
        if not forces.requires_grad:
            raise RuntimeError("EnergyHessianOutput requires differentiable `forces`.")
        if not positions.requires_grad:
            raise RuntimeError("EnergyHessianOutput requires differentiable `positions`.")
        create_graph = self.training if training is None else training
        if properties.energy_hessian in self.model_outputs:
            data[properties.energy_hessian] = get_full_energy_hessian(
                forces,
                positions,
                create_graph=create_graph,
                vectorize=self.vectorize,
            )
        if properties.energy_hessian_sampled in self.model_outputs or properties.energy_hessian_sample_indices in self.model_outputs:
            if self.num_samples is None:
                raise ValueError("EnergyHessianOutput requires `num_samples` to output sampled Hessian values.")
            mask = data[self.mask_key] if self.mask_key is not None else None
            row_indices, rows = sample_hessian_rows(
                forces,
                positions,
                data[properties.n_atoms],
                num_samples=self.num_samples,
                mask=mask,
                create_graph=create_graph,
                vectorize=self.vectorize,
            )
            if properties.energy_hessian_sample_indices in self.model_outputs:
                data[properties.energy_hessian_sample_indices] = row_indices
            if properties.energy_hessian_sampled in self.model_outputs:
                data[properties.energy_hessian_sampled] = rows
        if properties.energy_hessian_projected in self.model_outputs or properties.energy_hessian_probe_vectors in self.model_outputs:
            probe_vectors = data.get(self.probe_key)
            if probe_vectors is None and self.num_probes is None:
                raise ValueError("EnergyHessianOutput requires `num_probes` or input probe vectors to output Hessian projections.")
            probe_vectors, projected = sample_hessian_projections(
                forces,
                positions,
                num_probes=int(self.num_probes or probe_vectors.shape[0]),
                n_atoms=data[properties.n_atoms],
                probe_vectors=probe_vectors,
                create_graph=create_graph,
                vectorize=self.vectorize,
                normalize_probes=self.normalize_probes,
                probe_distribution=self.probe_distribution,
            )
            if properties.energy_hessian_probe_vectors in self.model_outputs:
                data[properties.energy_hessian_probe_vectors] = probe_vectors
            if properties.energy_hessian_projected in self.model_outputs:
                data[properties.energy_hessian_projected] = projected
        return data

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(vectorize={self.vectorize}, "
            f"num_samples={self.num_samples}, num_probes={self.num_probes}, "
            f"mask_key={self.mask_key}, probe_key={self.probe_key}, "
            f"normalize_probes={self.normalize_probes}, "
            f"probe_distribution={self.probe_distribution}, model_outputs={self.model_outputs})"
        )
