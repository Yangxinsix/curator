import abc
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch

from . import properties


class Transform(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def forward(self):
        raise NotImplementedError


class UnitTransform(Transform):
    def __init__(
        self,
        unit_dict: Dict[str, float]
    ) -> None:
        super().__init__()

        self.unit_dict = unit_dict

    def forward(self, data: properties.Type) -> properties.Type:
        for k, v in self.unit_dict:
            data[k] *= v

        return data


class EnsureBidirectionalEdges(Transform):
    """Deduplicate directed edges and add any missing reverse edges.

    Reverse edges swap the atom indices and negate geometric displacement
    fields. Other registered edge fields are copied from the corresponding
    forward edge. Existing reverse edges retain their own field values.

    Batched inputs are handled per structure when ``n_pairs`` is present, so
    the contiguous edge grouping and per-structure pair counts are preserved.
    """

    _DEFAULT_ANTISYMMETRIC_FIELDS = {
        properties.edge_diff,
        properties.cell_displacements,
    }

    def __init__(
        self,
        antisymmetric_fields: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__()
        self.antisymmetric_fields: Set[str] = set(
            self._DEFAULT_ANTISYMMETRIC_FIELDS
            if antisymmetric_fields is None
            else antisymmetric_fields
        )
        self.antisymmetric_fields.add(properties.edge_diff)

    @staticmethod
    def _edge_key(
        atom_pair: List[int],
        displacement: List[float],
    ) -> Tuple:
        return (atom_pair[0], atom_pair[1], *displacement)

    @classmethod
    def _unique_and_missing_reverse_indices(
        cls,
        edge_idx: torch.Tensor,
        edge_diff: torch.Tensor,
        offset: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pairs = edge_idx.detach().cpu().tolist()
        displacements = edge_diff.detach().cpu().tolist()

        seen = set()
        keep: List[int] = []
        keys: List[Tuple] = []
        for local_index, (pair, displacement) in enumerate(
            zip(pairs, displacements)
        ):
            key = cls._edge_key(pair, displacement)
            if key in seen:
                continue
            seen.add(key)
            keep.append(offset + local_index)
            keys.append(key)

        reverse_sources: List[int] = []
        for source_index, key in zip(keep, keys):
            reverse_key = (key[1], key[0], *(-value for value in key[2:]))
            if reverse_key not in seen:
                seen.add(reverse_key)
                reverse_sources.append(source_index)

        device = edge_idx.device
        return (
            torch.tensor(keep, dtype=torch.long, device=device),
            torch.tensor(reverse_sources, dtype=torch.long, device=device),
        )

    @staticmethod
    def _edge_slices(
        data: properties.Type,
        num_edges: int,
    ) -> Tuple[List[Tuple[int, int]], Optional[torch.Tensor]]:
        n_pairs = data.get(properties.n_pairs)
        if n_pairs is None:
            return [(0, num_edges)], None
        if not torch.is_tensor(n_pairs):
            raise TypeError(f"'{properties.n_pairs}' must be a tensor.")

        counts = n_pairs.reshape(-1)
        if bool(torch.any(counts < 0)):
            raise ValueError(f"'{properties.n_pairs}' cannot contain negative counts.")
        if int(counts.sum().item()) != num_edges:
            raise ValueError(
                f"'{properties.n_pairs}' sums to {int(counts.sum().item())}, "
                f"but the input contains {num_edges} edges."
            )

        slices = []
        start = 0
        for count in counts.detach().cpu().tolist():
            end = start + int(count)
            slices.append((start, end))
            start = end
        return slices, n_pairs

    @staticmethod
    def _validate_edge_fields(
        data: properties.Type,
        num_edges: int,
    ) -> List[str]:
        edge_fields = []
        for key in properties._EDGE_FIELDS:
            if key not in data:
                continue
            value = data[key]
            if not torch.is_tensor(value):
                raise TypeError(f"Edge field '{key}' must be a tensor.")
            if value.dim() == 0 or value.shape[0] != num_edges:
                raise ValueError(
                    f"Edge field '{key}' has leading dimension "
                    f"{value.shape[0] if value.dim() else 'scalar'}, "
                    f"but the input contains {num_edges} edges."
                )
            edge_fields.append(key)
        return edge_fields

    def forward(self, data: properties.Type) -> properties.Type:
        if properties.edge_idx not in data or properties.edge_diff not in data:
            raise KeyError(
                "EnsureBidirectionalEdges requires both "
                f"'{properties.edge_idx}' and '{properties.edge_diff}'."
            )

        edge_idx = data[properties.edge_idx]
        edge_diff = data[properties.edge_diff]
        if edge_idx.dim() != 2 or edge_idx.shape[1] != 2:
            raise ValueError(
                f"'{properties.edge_idx}' must have shape [num_edges, 2]."
            )
        if edge_diff.dim() < 2 or edge_diff.shape[0] != edge_idx.shape[0]:
            raise ValueError(
                f"'{properties.edge_diff}' must have shape [num_edges, ...]."
            )

        num_edges = edge_idx.shape[0]
        edge_fields = self._validate_edge_fields(data, num_edges)
        slices, original_n_pairs = self._edge_slices(data, num_edges)

        segment_indices = []
        output_counts = []
        for start, end in slices:
            keep, reverse_sources = self._unique_and_missing_reverse_indices(
                edge_idx[start:end],
                edge_diff[start:end],
                start,
            )
            segment_indices.append((keep, reverse_sources))
            output_counts.append(keep.numel() + reverse_sources.numel())

        if not segment_indices:
            data[properties.n_pairs] = original_n_pairs
            return data

        edge_idx_parts = []
        for keep, reverse_sources in segment_indices:
            edge_idx_parts.append(edge_idx.index_select(0, keep))
            edge_idx_parts.append(
                edge_idx.index_select(0, reverse_sources).flip(dims=(1,))
            )
        data[properties.edge_idx] = torch.cat(edge_idx_parts, dim=0)

        for key in edge_fields:
            value = data[key]
            parts = []
            for keep, reverse_sources in segment_indices:
                parts.append(value.index_select(0, keep))
                reverse = value.index_select(0, reverse_sources)
                if key in self.antisymmetric_fields:
                    reverse = -reverse
                parts.append(reverse)
            data[key] = torch.cat(parts, dim=0)

        count_tensor = torch.tensor(
            output_counts,
            dtype=(
                original_n_pairs.dtype
                if original_n_pairs is not None
                else torch.long
            ),
            device=(
                original_n_pairs.device
                if original_n_pairs is not None
                else edge_idx.device
            ),
        )
        data[properties.n_pairs] = (
            count_tensor.reshape_as(original_n_pairs)
            if original_n_pairs is not None
            else count_tensor
        )
        return data
