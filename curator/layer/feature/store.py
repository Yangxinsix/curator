from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import h5py
import torch

from .common import KernelName


class H5Feature:
    """Append-only HDF5 store for model features."""

    def __init__(
        self,
        path: Union[str, Path],
        num_models: int,
        kernels: Optional[Sequence[KernelName]] = None,
        dataset_size: Optional[int] = None,
        compression: Optional[str] = None,
        chunk_rows: Optional[int] = None,
    ) -> None:
        self.path = Path(path)
        self.num_models = int(num_models)
        self.kernels = list(kernels) if kernels is not None else None
        self.dataset_size = int(dataset_size) if dataset_size is not None else None
        self.compression = compression
        self.chunk_rows = chunk_rows
        if self.kernels is not None:
            self.ensure(self.kernels, dataset_size=self.dataset_size)

    def ensure(
        self,
        kernels: Optional[Sequence[KernelName]] = None,
        dataset_size: Optional[int] = None,
    ) -> List[str]:
        if kernels is None:
            kernels = self.kernels
        if kernels is None:
            raise ValueError("kernels are required.")
        kernels_list = [str(k) for k in kernels]
        with h5py.File(self.path, "a") as handle:
            existing = handle.attrs.get("kernels")
            if existing is None:
                handle.attrs["kernels"] = kernels_list
            else:
                stored = [k.decode() if isinstance(k, (bytes, bytearray)) else str(k) for k in existing]
                if stored != kernels_list:
                    raise ValueError("HDF5 kernels do not match.")
            existing_models = handle.attrs.get("num_models")
            if existing_models is None:
                handle.attrs["num_models"] = self.num_models
            elif int(existing_models) != self.num_models:
                raise ValueError("HDF5 num_models does not match.")
            if dataset_size is not None:
                existing_size = handle.attrs.get("dataset_size")
                if existing_size is None:
                    handle.attrs["dataset_size"] = int(dataset_size)
                elif int(existing_size) != int(dataset_size):
                    raise ValueError("HDF5 dataset_size does not match.")
        self.kernels = kernels_list
        if dataset_size is not None:
            self.dataset_size = int(dataset_size)
        return kernels_list

    def count(self, kernel: KernelName, model_idx: int) -> int:
        if model_idx < 0 or model_idx >= self.num_models:
            raise ValueError("model_idx is out of range.")
        with h5py.File(self.path, "a") as handle:
            group = handle.get(f"features/{kernel}")
            if group is None or "counts" not in group:
                return 0
            return int(group["counts"][model_idx])

    def append(
        self,
        kernel: KernelName,
        model_idx: int,
        feats: torch.Tensor,
        image_idx: Optional[torch.Tensor] = None,
    ) -> None:
        if model_idx < 0 or model_idx >= self.num_models:
            raise ValueError("model_idx is out of range.")
        if not torch.is_tensor(feats):
            raise TypeError("feats must be a torch.Tensor.")
        if feats.dim() != 2:
            raise ValueError("feats must be 2D (N, P).")
        if feats.numel() == 0:
            return
        if image_idx is not None:
            if not torch.is_tensor(image_idx):
                raise TypeError("image_idx must be a torch.Tensor.")
            if image_idx.dim() != 1:
                raise ValueError("image_idx must be 1D.")
            if image_idx.shape[0] != feats.shape[0]:
                raise ValueError("image_idx length must match feats rows.")

        with h5py.File(self.path, "a") as handle:
            group = handle.require_group(f"features/{kernel}")
            data = group.get("data")
            if data is None:
                chunks = True if self.chunk_rows is None else (1, self.chunk_rows, feats.shape[1])
                data = group.create_dataset(
                    "data",
                    shape=(self.num_models, 0, feats.shape[1]),
                    maxshape=(self.num_models, None, feats.shape[1]),
                    chunks=chunks,
                    compression=self.compression,
                )
            elif data.shape[0] != self.num_models or data.shape[2] != feats.shape[1]:
                raise ValueError("HDF5 data shape does not match.")

            counts = group.get("counts")
            if counts is None:
                counts = group.create_dataset("counts", data=[0] * self.num_models, dtype="i8")
            current = int(counts[model_idx])
            new_total = current + feats.shape[0]
            if new_total > data.shape[1]:
                data.resize((self.num_models, new_total, data.shape[2]))
            data[model_idx, current:new_total, :] = feats.detach().cpu().numpy()
            counts[model_idx] = new_total

            if image_idx is not None:
                idx = group.get("image_idx")
                if idx is None:
                    idx = group.create_dataset(
                        "image_idx",
                        shape=(self.num_models, 0),
                        maxshape=(self.num_models, None),
                        chunks=True,
                        dtype="i8",
                    )
                old = idx.shape[1]
                if new_total > old:
                    idx.resize((self.num_models, new_total))
                idx[model_idx, current:new_total] = image_idx.detach().cpu().numpy()

    def load(self, kernel: KernelName) -> torch.Tensor:
        with h5py.File(self.path, "r") as handle:
            group = handle.get(f"features/{kernel}")
            if group is None or "data" not in group:
                return torch.empty((self.num_models, 0, 0))
            return torch.from_numpy(group["data"][()])

    def load_with_counts(self, kernel: KernelName) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        with h5py.File(self.path, "r") as handle:
            group = handle.get(f"features/{kernel}")
            if group is None or "data" not in group:
                data = torch.empty((self.num_models, 0, 0))
                counts = torch.zeros((self.num_models,), dtype=torch.long)
                return data, counts, None
            data = torch.from_numpy(group["data"][()])
            counts = (
                torch.from_numpy(group["counts"][()])
                if "counts" in group
                else torch.full((self.num_models,), data.shape[1], dtype=torch.long)
            )
            image_idx = torch.from_numpy(group["image_idx"][()]) if "image_idx" in group else None
            return data, counts, image_idx

    def load_image_idx(self, kernel: KernelName) -> Optional[torch.Tensor]:
        with h5py.File(self.path, "r") as handle:
            group = handle.get(f"features/{kernel}")
            if group is None or "image_idx" not in group:
                return None
            return torch.from_numpy(group["image_idx"][()])
