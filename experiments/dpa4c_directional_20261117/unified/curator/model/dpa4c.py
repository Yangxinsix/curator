from __future__ import annotations

from collections import OrderedDict
from itertools import combinations_with_replacement
from math import ceil, factorial, pi, sqrt
from typing import Any

import numpy as np
import torch
from ase.data import atomic_numbers
from torch import nn

from curator.data import properties
from curator.layer import BesselBasis, Dense, GaussianBasis, OneHotAtomEncoding
from curator.model.base import ParameterGroup, Representation
from curator.model.features import ScalarVectorFeatureSpec


def _degree_channels(channels: int, l_max: int) -> list[int]:
    if channels not in {8, 16, 32, 64, 128}:
        raise ValueError("DPA4C channels must be one of 8, 16, 32, 64, or 128")
    if l_max not in {2, 3, 4}:
        raise ValueError("DPA4C l_max must be 2, 3, or 4")
    degree_one = max(4, 2 ** ceil(0.5 * np.log2(channels)))
    return [channels, degree_one, max(4, degree_one // 2)] + [1] * (l_max - 2)


def _angular_basis(direction: torch.Tensor, l_max: int) -> torch.Tensor:
    """Real solid harmonics in DPA4C Supplementary Eq. (S1)."""
    x, y, z = direction.unbind(-1)
    squared_norm = direction.square().sum(dim=-1)
    blocks: list[torch.Tensor] = [torch.ones_like(x).unsqueeze(-1)]
    blocks.append(torch.stack((x, y, z), dim=-1))
    blocks.append(
        torch.stack(
            (
                sqrt(3.0) * x * y,
                sqrt(3.0) * y * z,
                0.5 * (3.0 * z.square() - squared_norm),
                sqrt(3.0) * x * z,
                0.5 * sqrt(3.0) * (x.square() - y.square()),
            ),
            dim=-1,
        )
    )
    if l_max >= 3:
        blocks.append(
            torch.stack(
                (
                    sqrt(5.0 / 8.0) * y * (3.0 * x.square() - y.square()),
                    sqrt(15.0) * x * y * z,
                    sqrt(3.0 / 8.0) * y * (5.0 * z.square() - squared_norm),
                    0.5 * z * (5.0 * z.square() - 3.0 * squared_norm),
                    sqrt(3.0 / 8.0) * x * (5.0 * z.square() - squared_norm),
                    0.5 * sqrt(15.0) * z * (x.square() - y.square()),
                    sqrt(5.0 / 8.0) * x * (x.square() - 3.0 * y.square()),
                ),
                dim=-1,
            )
        )
    if l_max >= 4:
        z2 = z.square()
        x2_minus_y2 = x.square() - y.square()
        blocks.append(
            torch.stack(
                (
                    0.5 * sqrt(35.0) * x * y * x2_minus_y2,
                    0.25 * sqrt(70.0) * y * z * (3.0 * x.square() - y.square()),
                    0.5 * sqrt(5.0) * x * y * (7.0 * z2 - squared_norm),
                    0.25 * sqrt(10.0) * y * z * (7.0 * z2 - 3.0 * squared_norm),
                    0.125
                    * (35.0 * z2.square() - 30.0 * z2 * squared_norm + 3.0 * squared_norm.square()),
                    0.25 * sqrt(10.0) * x * z * (7.0 * z2 - 3.0 * squared_norm),
                    0.25 * sqrt(5.0) * x2_minus_y2 * (7.0 * z2 - squared_norm),
                    0.25 * sqrt(70.0) * x * z * (x.square() - 3.0 * y.square()),
                    0.125
                    * sqrt(35.0)
                    * (x.pow(4) - 6.0 * x.square() * y.square() + y.pow(4)),
                ),
                dim=-1,
            )
        )
    return torch.cat(blocks, dim=-1)


def _packed_l2_to_stf(packed: torch.Tensor) -> torch.Tensor:
    """DPA4C Eq. (15), preserving the Euclidean/Frobenius inner product."""
    q0, q1, q2, q3, q4 = packed.unbind(-1)
    inv_sqrt_two = 1.0 / sqrt(2.0)
    inv_sqrt_six = 1.0 / sqrt(6.0)
    qxy = q0 * inv_sqrt_two
    qyz = q1 * inv_sqrt_two
    qxz = q3 * inv_sqrt_two
    qxx = -q2 * inv_sqrt_six + q4 * inv_sqrt_two
    qyy = -q2 * inv_sqrt_six - q4 * inv_sqrt_two
    qzz = 2.0 * q2 * inv_sqrt_six
    return torch.stack(
        (
            torch.stack((qxx, qxy, qxz), dim=-1),
            torch.stack((qxy, qyy, qyz), dim=-1),
            torch.stack((qxz, qyz, qzz), dim=-1),
        ),
        dim=-2,
    )


def _degree_triples(l_max: int) -> list[tuple[int, int, int]]:
    return [
        (left, middle, right)
        for left in range(1, l_max + 1)
        for middle in range(left, l_max + 1)
        for right in range(middle, l_max + 1)
        if right <= left + middle and (left + middle + right) % 2 == 0
    ]


def _probe_entries(
    degrees: tuple[int, int, int], ranks: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    widths = [ranks[degree - 1] for degree in degrees]
    entries: list[tuple[int, int, int]] = []
    scales: list[float] = []
    if degrees[0] == degrees[2]:
        for entry in combinations_with_replacement(range(widths[0]), 3):
            multiplicity = factorial(3)
            for value in set(entry):
                multiplicity //= factorial(entry.count(value))
            entries.append(entry)
            scales.append(sqrt(float(multiplicity)))
    elif degrees[0] == degrees[1]:
        for first, second in combinations_with_replacement(range(widths[0]), 2):
            for third in range(widths[2]):
                entries.append((first, second, third))
                scales.append(1.0 if first == second else sqrt(2.0))
    elif degrees[1] == degrees[2]:
        for first in range(widths[0]):
            for second, third in combinations_with_replacement(range(widths[1]), 2):
                entries.append((first, second, third))
                scales.append(1.0 if second == third else sqrt(2.0))
    else:
        for first in range(widths[0]):
            for second in range(widths[1]):
                for third in range(widths[2]):
                    entries.append((first, second, third))
                    scales.append(1.0)
    return torch.tensor(entries, dtype=torch.long), torch.tensor(scales)


def _gaunt_coupling(degrees: tuple[int, int, int], l_max: int) -> torch.Tensor:
    """Evaluate Eq. (25) with a polynomial-exact product quadrature."""
    z, weights = np.polynomial.legendre.leggauss(16)
    phi = np.arange(32, dtype=np.float64) * (2.0 * np.pi / 32.0)
    radius = np.sqrt(1.0 - z[:, None] ** 2)
    points = np.stack(
        (
            (radius * np.cos(phi)[None, :]).reshape(-1),
            (radius * np.sin(phi)[None, :]).reshape(-1),
            np.broadcast_to(z[:, None], (16, 32)).reshape(-1),
        ),
        axis=-1,
    )
    quadrature = torch.from_numpy(np.repeat(weights, 32) * (2.0 * np.pi / 32.0))
    basis = _angular_basis(torch.from_numpy(points), l_max)
    blocks = [basis[:, degree**2 : (degree + 1) ** 2] for degree in degrees]
    coupling = torch.einsum("n,na,nb,nc->abc", quadrature, *blocks)
    coupling = coupling / torch.linalg.vector_norm(coupling)
    significant = coupling.flatten().abs() > 1.0e-13
    first = coupling.flatten()[significant][0]
    return coupling * torch.where(first < 0.0, -1.0, 1.0)


class _SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.input = Dense(in_features, 2 * hidden_features, bias=False)
        self.output = Dense(hidden_features, out_features, bias=False)

    def hidden(self, value: torch.Tensor) -> torch.Tensor:
        gate, linear = self.input(value).chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * linear

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden(value))


class DPA4C(Representation):
    """Trainable reference implementation of the DPA4C potential.

    The equations follow Li et al., *Universal Machine-learning Molecular
    Dynamics at the Speed of Empirical Potentials*, arXiv:2608.19041,
    Eqs. (2)--(38).  It implements the spin-free, charge-free local energy
    model: one ordered-pair radial message, degree-wise Cartesian moments,
    exact Grams, low-rank bispectrum and quartic invariants, and a residual
    atomic-energy MLP.  Forces and virials are obtained through Curator's
    existing ``PairwiseDistance`` and ``GradientOutput`` modules.

    The ordinary PyTorch path supports training across the published widths,
    angular orders and radial-mode counts. Small (C=32, L=3, R=4) can run
    differentiable native CUDA message and invariant contractions through
    the Python representation, including force-loss double backward.
    The compressed Nano backend and
    ``curator/desc/kk`` deployment currently accelerate the paper's Nano
    profile (C=8, L=2, R=0) with a compressed radial table; native training
    replays this reference graph in backward.  Curator does not currently
    import DeePMD checkpoints or implement charge/spin conditioning.
    """

    def __init__(
        self,
        cutoff: float = 6.0,
        channels: int = 32,
        l_max: int = 2,
        radial_modes: int = 0,
        n_radial: int = 16,
        species: list[str] | None = None,
        radial_basis: str = "bessel",
        readout_hidden_features: list[int] | None = None,
        readout_activation: str = "silu",
        inference_backend: str = "reference",
        heads: list | None = None,
    ) -> None:
        super().__init__(heads=heads)
        if cutoff <= 0.0:
            raise ValueError("DPA4C cutoff must be positive")
        if radial_modes < 0:
            raise ValueError("DPA4C radial_modes must be non-negative")
        if n_radial < 2:
            raise ValueError("DPA4C n_radial must be at least two")
        if not species or len(set(species)) != len(species):
            raise ValueError("DPA4C species must contain unique element symbols")
        if inference_backend not in {"reference", "auto", "native"}:
            raise ValueError("DPA4C backend must be reference, auto, or native")

        self.cutoff = float(cutoff)
        self.channels = int(channels)
        self.l_max = int(l_max)
        self.radial_modes = int(radial_modes)
        self.n_radial = int(n_radial)
        self.species = sorted(species, key=atomic_numbers.__getitem__)
        self.num_elements = len(self.species)
        self.radial_basis_name = radial_basis.lower()
        self.readout_hidden_features = list(
            [128, 128, 128]
            if readout_hidden_features is None
            else readout_hidden_features
        )
        if not self.readout_hidden_features or min(self.readout_hidden_features) < 1:
            raise ValueError("DPA4C readout_hidden_features must be positive")
        if len(set(self.readout_hidden_features)) != 1:
            raise ValueError("DPA4C uses equal-width residual hidden layers")
        if readout_activation not in {"silu", "tanh"}:
            raise ValueError("DPA4C readout_activation must be silu or tanh")
        self.readout_activation = readout_activation
        self.inference_backend = inference_backend
        # DPA4C uses its own smooth per-environment normalizers (Eq. 19).
        # This neutral compatibility scalar is retained only for the common
        # Curator descriptor artifact header and is never used by the model.
        self.register_buffer("avg_num_neighbors", torch.tensor(1.0))
        self.degree_channels = _degree_channels(self.channels, self.l_max)
        self.bispectrum_ranks = [self.degree_channels[2], 2] + [1] * (self.l_max - 2)
        self.moment_offsets = [0]
        for degree, width in enumerate(self.degree_channels):
            self.moment_offsets.append(
                self.moment_offsets[-1] + (2 * degree + 1) * width
            )
        self.moment_dim = self.moment_offsets[-1]

        if self.radial_basis_name == "bessel":
            basis: nn.Module = BesselBasis(
                cutoff=self.cutoff,
                num_basis=self.n_radial,
                trainable=True,
            )
        elif self.radial_basis_name == "gaussian":
            basis = GaussianBasis(
                cutoff=self.cutoff,
                n_rbf=self.n_radial,
                trainable=True,
            )
        else:
            raise ValueError("DPA4C radial_basis must be bessel or gaussian")
        self.embeddings = nn.ModuleDict(
            {
                "onehot": OneHotAtomEncoding(
                    num_elements=self.num_elements,
                    species=self.species,
                    set_features=False,
                ),
                "radial": basis,
            }
        )
        self.element_embedding = nn.Embedding(self.num_elements, self.channels)
        radial_hidden = 8 * ceil(self.channels / 3)
        pair_hidden = 8 * ceil(2 * self.channels / 3)
        self.radial_network = _SwiGLU(
            self.n_radial, radial_hidden, self.channels
        )
        self.radial_mode_head = (
            nn.Identity()
            if self.radial_modes == 0
            else Dense(radial_hidden, self.radial_modes, bias=False)
        )
        self.pair_network = _SwiGLU(
            2 * self.channels,
            pair_hidden,
            self.channels * (2 + self.radial_modes),
        )

        self.alignment_1 = nn.Parameter(torch.zeros(self.degree_channels[1], self.degree_channels[1]))
        self.alignment_2 = nn.Parameter(torch.zeros(self.degree_channels[2], self.degree_channels[2]))
        for degree in range(1, 5):
            if degree <= self.l_max:
                width = self.degree_channels[degree]
                rank = self.bispectrum_ranks[degree - 1]
                if rank == width:
                    self.register_buffer(f"projection_{degree}", torch.eye(width))
                else:
                    projection = torch.linalg.qr(torch.randn(width, rank))[0]
                    self.register_parameter(
                        f"projection_{degree}", nn.Parameter(projection)
                    )
            else:
                self.register_buffer(f"projection_{degree}", torch.empty(0, 0))

        triples = _degree_triples(self.l_max)
        for name, degrees in (
            ("112", (1, 1, 2)),
            ("123", (1, 2, 3)),
            ("134", (1, 3, 4)),
            ("222", (2, 2, 2)),
            ("224", (2, 2, 4)),
            ("233", (2, 3, 3)),
            ("244", (2, 4, 4)),
            ("334", (3, 3, 4)),
            ("444", (4, 4, 4)),
        ):
            if degrees in triples:
                indices, scales = _probe_entries(degrees, self.bispectrum_ranks)
                coupling = (
                    torch.empty(0)
                    if name in {"112", "222"}
                    else _gaunt_coupling(degrees, self.l_max)
                )
            else:
                indices = torch.empty((0, 3), dtype=torch.long)
                scales = torch.empty(0)
                coupling = torch.empty(0)
            flat = (
                (indices[:, 0] * self.bispectrum_ranks[degrees[1] - 1] + indices[:, 1])
                * self.bispectrum_ranks[degrees[2] - 1]
                + indices[:, 2]
                if indices.numel()
                else torch.empty(0, dtype=torch.long)
            )
            self.register_buffer(f"bispectrum_index_{name}", flat)
            self.register_buffer(f"bispectrum_scale_{name}", scales)
            self.register_buffer(f"bispectrum_probe_{name}", indices)
            self.register_buffer(f"coupling_{name}", coupling)

        gram_dim = sum(width * (width + 1) // 2 for width in self.degree_channels[1:])
        bispectrum_dim = sum(
            int(getattr(self, f"bispectrum_index_{name}").numel())
            for name in ("112", "123", "134", "222", "224", "233", "244", "334", "444")
        )
        quartic_dim = self.bispectrum_ranks[0] * self.bispectrum_ranks[1]
        self.descriptor_dim = 2 * self.channels + 2 + gram_dim + bispectrum_dim + quartic_dim
        self.register_buffer("calibration_mean", torch.zeros(self.descriptor_dim))
        self.register_buffer("calibration_inv_std", torch.ones(self.descriptor_dim))

        hidden = self.readout_hidden_features[0]
        self.readout_input = Dense(self.descriptor_dim, hidden)
        self.readout_hidden = nn.ModuleList(
            [Dense(hidden, hidden) for _ in self.readout_hidden_features[1:]]
        )
        self.readout_head = Dense(hidden, 1)
        self.atomic_bias = nn.Parameter(torch.zeros(self.num_elements))
        self.model_outputs = [properties.energy]
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.element_embedding.weight, std=self.channels**-0.5)
        nn.init.zeros_(self.atomic_bias)

    @torch.jit.ignore
    def set_inference_backend(self, name: str) -> DPA4C:
        if name not in {"reference", "auto", "native"}:
            raise ValueError("DPA4C backend must be reference, auto, or native")
        self.inference_backend = name
        return self

    @torch.jit.unused
    def direct_force_feature_spec(self) -> ScalarVectorFeatureSpec:
        return ScalarVectorFeatureSpec(
            scalar_key=properties.node_feat,
            vector_key=properties.node_vect,
            scalar_channels=self.descriptor_dim,
            vector_channels=self.degree_channels[1],
        )

    @torch.jit.unused
    def response_feature_spec(self) -> ScalarVectorFeatureSpec:
        return self.direct_force_feature_spec()

    def _vector_features(self, moments: torch.Tensor) -> torch.Tensor:
        """Degree-one polar moments, ordered by Cartesian axis then channel."""
        return moments[:, self.moment_offsets[1] : self.moment_offsets[2]].reshape(
            moments.shape[0], 3, self.degree_channels[1]
        )

    @torch.jit.ignore
    def set_calibration(self, mean: torch.Tensor, std: torch.Tensor) -> DPA4C:
        if mean.shape != (self.descriptor_dim,) or std.shape != (self.descriptor_dim,):
            raise ValueError("DPA4C calibration tensors have the wrong shape")
        if not torch.isfinite(std).all() or (std <= 0.0).any():
            raise ValueError("DPA4C calibration std must be finite and positive")
        self.calibration_mean.copy_(mean)
        self.calibration_inv_std.copy_(std.reciprocal())
        return self

    def _cutoff(self, distance: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(distance / self.cutoff, min=0.0, max=1.0)
        value = 1.0 - 56.0 * x.pow(5) + 140.0 * x.pow(6) - 120.0 * x.pow(7) + 35.0 * x.pow(8)
        return value * (distance < self.cutoff)

    def _radial_profiles(self, distance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.radial_basis_name == "bessel":
            weights = self.embeddings["radial"].bessel_weights
            argument = distance.unsqueeze(-1) * weights
            small = argument.abs() < 1.0e-3
            safe_argument = torch.where(small, torch.ones_like(argument), argument)
            regular = weights * torch.sin(safe_argument) / safe_argument
            squared = argument.square()
            series = weights * (
                1.0 - squared / 6.0 + squared.square() / 120.0 - squared.pow(3) / 5040.0
            )
            radial_basis = torch.where(small, series, regular)
        else:
            radial_basis = self.embeddings["radial"](distance)
        hidden = self.radial_network.hidden(radial_basis)
        radial = self.radial_network.output(hidden)
        modes = self.radial_mode_head(hidden)[:, : self.radial_modes]
        return radial, modes

    def _pair_coefficients(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.element_embedding.weight
        center = embedding[:, None, :].expand(-1, self.num_elements, -1)
        neighbor = embedding[None, :, :].expand(self.num_elements, -1, -1)
        pair = torch.cat((center, neighbor), dim=-1).reshape(-1, 2 * self.channels)
        output = 0.1 * self.pair_network(pair)
        scale = 1.0 + torch.tanh(output[:, : self.channels])
        shift = center.reshape(-1, self.channels) + neighbor.reshape(-1, self.channels)
        shift = shift + torch.tanh(output[:, self.channels : 2 * self.channels])
        mixing = torch.tanh(output[:, 2 * self.channels :]).reshape(
            self.num_elements * self.num_elements,
            self.channels,
            self.radial_modes,
        )
        return scale, shift, mixing

    @staticmethod
    def _select(full: torch.Tensor, index: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return full.flatten(1).index_select(1, index) * scale

    @staticmethod
    def _generic_bispectrum(
        left: torch.Tensor,
        middle: torch.Tensor,
        right: torch.Tensor,
        coupling: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum("abc,nai,nbj,nck->nijk", coupling, left, middle, right)

    def _invariants(self, moments: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        blocks: list[torch.Tensor] = []
        for degree, width in enumerate(self.degree_channels):
            blocks.append(
                moments[:, self.moment_offsets[degree] : self.moment_offsets[degree + 1]].reshape(
                    moments.shape[0], 2 * degree + 1, width
                )
            )
        aligned: list[torch.Tensor] = [blocks[0]]
        aligned.append(blocks[1] @ (torch.eye(self.degree_channels[1], dtype=moments.dtype, device=moments.device) + self.alignment_1))
        aligned.append(blocks[2] @ (torch.eye(self.degree_channels[2], dtype=moments.dtype, device=moments.device) + self.alignment_2))
        if self.l_max >= 3:
            aligned.append(blocks[3])
        if self.l_max >= 4:
            aligned.append(blocks[4])

        grams: list[torch.Tensor] = []
        for block in aligned[1:]:
            gram = torch.einsum("nmc,nmd->ncd", block, block)
            indices = torch.triu_indices(
                block.shape[-1], block.shape[-1], device=block.device
            )
            row, column = indices[0], indices[1]
            scale = torch.where(row == column, torch.ones_like(row), torch.full_like(row, 2)).to(block.dtype).sqrt()
            grams.append(gram[:, row, column] * scale)

        projected: list[torch.Tensor] = []
        projected.append(aligned[1] @ self.projection_1)
        projected.append(aligned[2] @ self.projection_2)
        if self.l_max >= 3:
            projected.append(aligned[3] @ self.projection_3)
        if self.l_max >= 4:
            projected.append(aligned[4] @ self.projection_4)
        vectors = projected[0].transpose(1, 2)
        tensors = _packed_l2_to_stf(projected[1].transpose(1, 2))
        full_112 = -torch.einsum("nai,nbij,ncj->nacb", vectors, tensors, vectors) / sqrt(5.0)
        qv = torch.einsum("nbij,naj->nbai", tensors, vectors)
        quartic = qv.square().sum(dim=-1).flatten(1)
        bispectrum: list[torch.Tensor] = [
            self._select(full_112, self.bispectrum_index_112, self.bispectrum_scale_112)
        ]
        if self.l_max >= 3:
            bispectrum.append(self._select(self._generic_bispectrum(projected[0], projected[1], projected[2], self.coupling_123), self.bispectrum_index_123, self.bispectrum_scale_123))
        if self.l_max >= 4:
            bispectrum.append(self._select(self._generic_bispectrum(projected[0], projected[2], projected[3], self.coupling_134), self.bispectrum_index_134, self.bispectrum_scale_134))
        full_222 = -sqrt(12.0 / 35.0) * torch.einsum("naij,nbjk,ncki->nabc", tensors, tensors, tensors)
        bispectrum.append(self._select(full_222, self.bispectrum_index_222, self.bispectrum_scale_222))
        if self.l_max >= 4:
            bispectrum.append(self._select(self._generic_bispectrum(projected[1], projected[1], projected[3], self.coupling_224), self.bispectrum_index_224, self.bispectrum_scale_224))
        if self.l_max >= 3:
            bispectrum.append(self._select(self._generic_bispectrum(projected[1], projected[2], projected[2], self.coupling_233), self.bispectrum_index_233, self.bispectrum_scale_233))
        if self.l_max >= 4:
            bispectrum.append(self._select(self._generic_bispectrum(projected[1], projected[3], projected[3], self.coupling_244), self.bispectrum_index_244, self.bispectrum_scale_244))
            bispectrum.append(self._select(self._generic_bispectrum(projected[2], projected[2], projected[3], self.coupling_334), self.bispectrum_index_334, self.bispectrum_scale_334))
            bispectrum.append(self._select(self._generic_bispectrum(projected[3], projected[3], projected[3], self.coupling_444), self.bispectrum_index_444, self.bispectrum_scale_444))
        return grams + bispectrum, blocks[0][:, 0], quartic

    def _descriptor_from_geometry(
        self,
        edge_idx: torch.Tensor,
        edge_diff: torch.Tensor,
        atom_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distance = torch.sqrt(edge_diff.square().sum(dim=-1) + 1.0e-14)
        direction = edge_diff / distance.unsqueeze(-1)
        envelope = self._cutoff(distance)
        radial, modes = self._radial_profiles(distance)
        scale, shift, mixing = self._pair_coefficients()
        pair = atom_types[edge_idx[:, 0]] * self.num_elements + atom_types[edge_idx[:, 1]]
        amplitude = scale[pair] * radial + shift[pair]
        if self.radial_modes > 0:
            amplitude = amplitude + (mixing[pair] * modes.unsqueeze(1)).sum(dim=-1)
        amplitude = amplitude * envelope.unsqueeze(-1)
        basis = _angular_basis(direction, self.l_max)
        messages: list[torch.Tensor] = [envelope.square().unsqueeze(-1), envelope.pow(4).unsqueeze(-1)]
        for degree, width in enumerate(self.degree_channels):
            angular = basis[:, degree * degree : (degree + 1) * (degree + 1)]
            value = amplitude[:, :width].unsqueeze(1) * angular.unsqueeze(-1)
            if degree > 0:
                value = value * envelope[:, None, None]
            messages.append(value.flatten(1))
        edge_message = torch.cat(messages, dim=-1)
        reduced = torch.zeros(
            (atom_types.shape[0], edge_message.shape[1]),
            dtype=edge_diff.dtype,
            device=edge_diff.device,
        ).index_add_(0, edge_idx[:, 0], edge_message)
        divisors = torch.sqrt(reduced[:, :2] + 0.25)
        moments = torch.cat(
            (
                reduced[:, 2 : 2 + self.channels] / divisors[:, :1],
                reduced[:, 2 + self.channels :] / divisors[:, 1:],
            ),
            dim=-1,
        )
        invariant_blocks, scalar, quartic = self._invariants(moments)
        descriptor_blocks: list[torch.Tensor] = [
            self.element_embedding(atom_types),
            scalar,
            divisors,
        ]
        descriptor_blocks.extend(invariant_blocks)
        descriptor_blocks.append(quartic)
        raw = torch.cat(descriptor_blocks, dim=-1)
        descriptor = (raw - self.calibration_mean) * self.calibration_inv_std
        return descriptor, moments

    def _atomic_readout(self, descriptor: torch.Tensor, atom_types: torch.Tensor) -> torch.Tensor:
        hidden = self._activate(self.readout_input(descriptor))
        for layer in self.readout_hidden:
            hidden = hidden + self._activate(layer(hidden))
        return self.readout_head(hidden).squeeze(-1) + self.atomic_bias[atom_types]

    def _activate(self, value: torch.Tensor) -> torch.Tensor:
        if self.readout_activation == "silu":
            return torch.nn.functional.silu(value)
        return torch.tanh(value)

    def _reference_from_geometry(
        self,
        edge_idx: torch.Tensor,
        edge_diff: torch.Tensor,
        atom_types: torch.Tensor,
        num_atoms: int,
        n_local: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        descriptor, moments = self._descriptor_from_geometry(edge_idx, edge_diff, atom_types)
        local_atoms = num_atoms if n_local is None else n_local
        return (
            self._atomic_readout(descriptor[:local_atoms], atom_types[:local_atoms]),
            descriptor[:local_atoms],
            moments,
        )

    def forward(
        self,
        data: properties.Type,
        lammps_data: Any | None = None,
        n_local: int | None = None,
        n_ghost: int | None = None,
    ) -> properties.Type:
        if torch.jit.is_scripting() or self.inference_backend == "reference":
            return self._forward_reference(data, lammps_data, n_local, n_ghost)
        return self._forward_inference(data, lammps_data, n_local, n_ghost)

    @torch.jit.unused
    def _forward_inference(
        self,
        data: properties.Type,
        lammps_data: Any | None = None,
        n_local: int | None = None,
        n_ghost: int | None = None,
    ) -> properties.Type:
        del lammps_data, n_ghost
        if (self.channels, self.l_max, self.radial_modes) == (32, 3, 4):
            edges = data[properties.edge_diff]
            if edges.is_cuda and edges.dtype in (torch.float32, torch.float64):
                from curator.model.dpa4c_small import descriptor as small_descriptor

                self.embeddings["onehot"](data)
                atom_types = data[properties.atomic_types]
                features, moments = small_descriptor(self, data[properties.edge_idx], edges, atom_types)
                local = atom_types.shape[0] if n_local is None else n_local
                features = features[:local]
                atomic = self._atomic_readout(features, atom_types[:local])
                return self._store_local_output(data, atomic, features, moments[:local])
            if self.inference_backend == "native":
                raise RuntimeError("native DPA4C Small requires CUDA float32 or float64 inputs")
            return self._forward_reference(data, n_local=n_local)
        try:
            from curator.model.dpa4c_backend import evaluate_with_backend

            output = evaluate_with_backend(self.inference_backend, self, data, n_local=n_local)
        except ImportError:
            if self.inference_backend == "native":
                raise
            output = None
        if output is not None:
            return output
        return self._forward_reference(data, n_local=n_local)

    def _forward_reference(
        self,
        data: properties.Type,
        lammps_data: Any | None = None,
        n_local: int | None = None,
        n_ghost: int | None = None,
    ) -> properties.Type:
        del lammps_data, n_ghost
        self.embeddings["onehot"](data)
        atom_types = data[properties.atomic_types]
        atomic_energy, descriptor, moments = self._reference_from_geometry(
            data[properties.edge_idx],
            data[properties.edge_diff],
            atom_types,
            atom_types.shape[0],
            n_local,
        )
        local_atoms = atom_types.shape[0] if n_local is None else n_local
        return self._store_local_output(data, atomic_energy, descriptor, moments[:local_atoms])

    def _store_local_output(
        self, data: properties.Type, atomic_energy: torch.Tensor, descriptor: torch.Tensor,
        moments: torch.Tensor,
    ) -> properties.Type:
        if properties.precomputed_edge_energy_gradient in data:
            del data[properties.precomputed_edge_energy_gradient]
        if properties.precomputed_forces in data:
            del data[properties.precomputed_forces]
        local_atoms = atomic_energy.shape[0]
        image_idx = data.get(properties.image_idx)
        if image_idx is None:
            image_idx = torch.zeros(local_atoms, dtype=torch.long, device=atomic_energy.device)
            data[properties.image_idx] = image_idx
        else:
            image_idx = image_idx[:local_atoms]
            data[properties.image_idx] = image_idx
        energy = torch.zeros(
            data[properties.n_atoms].reshape(-1).shape[0],
            dtype=atomic_energy.dtype,
            device=atomic_energy.device,
        )
        data[properties.atomic_energy] = atomic_energy
        data[properties.energy] = energy.index_add_(0, image_idx, atomic_energy)
        data[properties.node_feat] = descriptor
        data[properties.node_embedding] = moments
        data[properties.node_vect] = self._vector_features(data[properties.node_embedding])
        return data

    @torch.jit.ignore
    def _materialize_native(self, step: float = 0.002) -> dict[str, Any]:
        if self.channels != 8 or self.l_max != 2 or self.radial_modes != 0:
            raise ValueError("The first DPA4C native profile supports only C=8, L=2, R=0")
        if step <= 0.0:
            raise ValueError("DPA4C table step must be positive")
        intervals = round(self.cutoff / step)
        if abs(intervals * step - self.cutoff) > 1.0e-10 * self.cutoff:
            raise ValueError("DPA4C cutoff must be an integer multiple of table step")
        device = self.element_embedding.weight.device
        dtype = self.element_embedding.weight.dtype
        distance = torch.arange(intervals + 1, device=device, dtype=dtype) * step
        distance.requires_grad_(True)
        radial, _ = self._radial_profiles(distance)
        first = []
        second = []
        for channel in range(self.channels):
            derivative = torch.autograd.grad(radial[:, channel].sum(), distance, create_graph=True, retain_graph=True)[0]
            first.append(derivative)
            second.append(torch.autograd.grad(derivative.sum(), distance, retain_graph=True)[0])
        radial_table = torch.stack((radial, torch.stack(first, dim=-1), torch.stack(second, dim=-1)), dim=-1).detach()
        pair_scale, pair_shift, _ = self._pair_coefficients()
        layers = [self.readout_input, *self.readout_hidden, self.readout_head]
        weights = torch.cat([layer.linear.weight.detach().flatten() for layer in layers])
        biases = torch.cat([layer.linear.bias.detach().flatten() for layer in layers])
        dims = torch.tensor(
            [self.descriptor_dim, *self.readout_hidden_features, 1],
            dtype=torch.long,
            device=device,
        )
        probes = torch.cat(
            (
                self.bispectrum_probe_112[:, [0, 2, 1]],
                self.bispectrum_probe_222,
            )
        )
        degrees = torch.cat(
            (
                torch.zeros(self.bispectrum_probe_112.shape[0], dtype=torch.long, device=device),
                torch.ones(self.bispectrum_probe_222.shape[0], dtype=torch.long, device=device),
            )
        )
        return {
            "profile": torch.tensor(
                [
                    self.channels,
                    self.degree_channels[1],
                    self.degree_channels[2],
                    self.bispectrum_ranks[0],
                    self.bispectrum_ranks[1],
                ],
                dtype=torch.int32,
                device=device,
            ),
            "table_step": float(step),
            "radial_table": radial_table,
            "pair_scale": pair_scale.detach().reshape(self.num_elements, self.num_elements, self.channels),
            "pair_shift": pair_shift.detach().reshape(self.num_elements, self.num_elements, self.channels),
            "type_embedding": self.element_embedding.weight.detach(),
            "align1": (torch.eye(self.degree_channels[1], device=device, dtype=dtype) + self.alignment_1).detach(),
            "align2": (torch.eye(self.degree_channels[2], device=device, dtype=dtype) + self.alignment_2).detach(),
            "proj1": self.projection_1.detach(),
            "proj2": self.projection_2.detach(),
            "bispectrum_indices": probes,
            "bispectrum_degrees": degrees,
            "bispectrum_scale": torch.cat((self.bispectrum_scale_112, self.bispectrum_scale_222)),
            "calibration_mean": self.calibration_mean.detach(),
            "calibration_inv_std": self.calibration_inv_std.detach(),
            "readout_weights": weights,
            "readout_biases": biases,
            "readout_dims": dims,
            "readout_activation": torch.tensor(
                0 if self.readout_activation == "silu" else 1,
                dtype=torch.int32,
                device=device,
            ),
            "atomic_bias": self.atomic_bias.detach(),
        }

    def module_groups(self) -> OrderedDict[str, list[nn.Module]]:
        return OrderedDict(
            (
                ("embeddings", [self.embeddings, self.element_embedding]),
                ("radial", [self.radial_network, self.radial_mode_head, self.pair_network]),
                ("readout", [self.readout_input, self.readout_hidden, self.readout_head]),
            )
        )

    def parameter_groups(self) -> list[ParameterGroup]:
        projection_parameters = [
            value
            for degree in range(1, self.l_max + 1)
            for value in [getattr(self, f"projection_{degree}")]
            if isinstance(value, nn.Parameter)
        ]
        return [
            ParameterGroup(
                "embedding",
                list(self.element_embedding.parameters())
                + list(self.embeddings["radial"].parameters()),
            ),
            ParameterGroup(
                "radial",
                list(self.radial_network.parameters())
                + list(self.radial_mode_head.parameters())
                + list(self.pair_network.parameters()),
            ),
            ParameterGroup(
                "invariants",
                [self.alignment_1, self.alignment_2] + projection_parameters,
            ),
            ParameterGroup(
                "readout",
                list(self.readout_input.parameters())
                + list(self.readout_hidden.parameters())
                + list(self.readout_head.parameters())
                + [self.atomic_bias],
            ),
        ]

    def export_init_kwargs(self) -> dict[str, Any]:
        return {
            "cutoff": self.cutoff,
            "channels": self.channels,
            "l_max": self.l_max,
            "radial_modes": self.radial_modes,
            "n_radial": self.n_radial,
            "species": list(self.species),
            "radial_basis": self.radial_basis_name,
            "readout_hidden_features": list(self.readout_hidden_features),
            "readout_activation": self.readout_activation,
            "inference_backend": self.inference_backend,
            "heads": list(self.heads),
        }
