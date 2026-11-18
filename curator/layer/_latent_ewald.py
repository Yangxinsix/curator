"""Optional adapters for the official ChengUCB/les PyTorch energy kernels."""
import math
import torch
from torch import nn
from curator.data import properties
from curator.model.features import scalar_vector_features

LES_REFERENCE_COMMIT = "35b971a650cf30adb84fef7138761e81aa47b3a2"


class LatentEwaldEnergy(nn.Module):
    """Add official LES energy to an already physical-unit short-range energy.

    This is a latent-charge readout, not QEq and not an ACVR proposal.
    Forces must be differentiated from the resulting total energy, using
    positions (the LES term is not a short-range-edge-only function).
    Call bind before building an optimizer. Angstrom/eV units are required.
    """
    repr_callback = True
    model_outputs = [properties.energy, "les_energy"]

    def __init__(self, hidden_features=(32, 16), output_scale=0.1, sigma=1.0,
                 dl=2.0, remove_self_interaction=True, dipoles=False,
                 induced_dipoles=False, polarizability_scale=0.01):
        super().__init__()
        if not all(math.isfinite(float(x)) and x > 0
                   for x in (sigma, dl, output_scale, polarizability_scale)):
            raise ValueError("LES scales and length parameters must be positive")
        try:
            from les import Les
        except ImportError as error:
            raise ImportError(
                "LatentEwaldEnergy requires the optional official ChengUCB/les "
                "package. See docs/benchmarks/LATENT_EWALD.md for the pinned runtime."
            ) from error
        self.backend = Les(dict(sigma=float(sigma), dl=float(dl),
                                remove_self_interaction=bool(remove_self_interaction),
                                use_atomwise=False, is_periodic=None))
        self.hidden_features = tuple(hidden_features)
        if not self.hidden_features or any(int(n) != n or n <= 0 for n in self.hidden_features):
            raise ValueError("hidden_features must contain positive integer widths")
        self.output_scale = float(output_scale)
        self.dipoles = bool(dipoles)
        self.induced_dipoles = bool(induced_dipoles)
        self.polarizability_scale = float(polarizability_scale)
        self.feature_spec = None
        self.charge_readout = None
        self.dipole_readout = None
        self.alpha_readout = None

    def bind(self, feature_spec, cutoff=None):
        if self.feature_spec is not None:
            if feature_spec != self.feature_spec:
                raise ValueError("LES feature specification cannot change after binding")
            return self
        scalar_channels = sum(mul for mul, ir in feature_spec.irreps if str(ir) == "0e")
        vector_channels = sum(mul for mul, ir in feature_spec.irreps if str(ir) == "1o")
        if not scalar_channels:
            raise ValueError("LES requires invariant scalar response features")
        if self.dipoles and not vector_channels:
            raise ValueError("Permanent LES dipoles require polar-vector (1o) features")
        from les.module.atomwise import Atomwise
        # Materialize the official lazy readout now, so an optimizer sees ALL
        # parameters before the first training forward.
        self.charge_readout = Atomwise(
            n_layers=len(self.hidden_features) + 1,
            n_hidden=list(self.hidden_features), add_linear_nn=True,
            output_scaling_factor=self.output_scale)
        with torch.no_grad():
            self.charge_readout(torch.zeros(1, scalar_channels), torch.zeros(1, dtype=torch.long))
        if self.dipoles:
            self.dipole_readout = nn.Linear(vector_channels, 1, bias=False)
        if self.induced_dipoles:
            self.alpha_readout = nn.Sequential(
                nn.Linear(scalar_channels, self.hidden_features[0]),
                nn.SiLU(), nn.Linear(self.hidden_features[0], 1), nn.Sigmoid())
        self.feature_spec = feature_spec
        return self

    def register_repr_callback(self, model):
        return self.bind(model.representation.response_feature_spec())

    @staticmethod
    def _geometry(data):
        positions = data[properties.positions]
        counts = data[properties.n_atoms].reshape(-1).to(dtype=torch.long)
        if counts.numel() == 0 or bool((counts <= 0).any()) or int(counts.sum()) != positions.shape[0]:
            raise ValueError("LES requires nonempty structures and consistent n_atoms")
        batch = torch.repeat_interleave(
            torch.arange(len(counts), device=positions.device), counts.to(positions.device))
        given_batch = data.get(properties.image_idx)
        if given_batch is not None and not torch.equal(given_batch.reshape(-1), batch):
            raise ValueError("LES expects Curator's contiguous per-structure atom ordering")
        pbc = data.get(properties.pbc)
        if pbc is None:
            raise ValueError("LES requires explicit pbc; a nonzero molecule box is not periodicity")
        pbc = pbc.to(device=positions.device, dtype=torch.bool).reshape(-1, 3)
        if pbc.shape[0] != len(counts):
            raise ValueError("LES pbc batch size differs from n_atoms")
        periodic = pbc.all(-1)
        if bool((pbc.any(-1) != periodic).any()):
            raise ValueError("LES supports fully periodic TTT or isolated FFF, not mixed axes")
        cell = data.get(properties.cell)
        if cell is None:
            if bool(periodic.any()):
                raise ValueError("Periodic LES needs a nonsingular cell")
            cell = positions.new_zeros((len(counts), 3, 3))
        else:
            cell = cell.reshape(-1, 3, 3).to(positions)
            if cell.shape[0] != len(counts):
                raise ValueError("LES cell batch size differs from n_atoms")
        if bool(periodic.any()):
            det = torch.linalg.det(cell[periodic])
            if not bool(torch.isfinite(det).all()) or bool((det <= 1e-6).any()):
                raise ValueError("Periodic LES requires finite right-handed nonsingular cells")
        # Official Ewald dispatches on det(cell), so isolated data MUST receive
        # zero cells even if its storage format supplies an artificial box.
        cell = torch.where(periodic[:, None, None], cell, torch.zeros_like(cell))
        return positions, cell, batch

    def forward(self, data):
        if self.feature_spec is None:
            raise RuntimeError("Bind LES response features before using the adapter")
        features = self.feature_spec.flatten(data)
        scalar, vector = scalar_vector_features(features, self.feature_spec.irreps)
        positions, cell, batch = self._geometry(data)
        charges = self.charge_readout(scalar, batch).squeeze(-1)
        dipoles = (self.dipole_readout(vector).squeeze(-1) * self.output_scale
                   if self.dipoles else None)
        alphas = (self.alpha_readout(scalar).squeeze(-1) * self.polarizability_scale
                  if self.induced_dipoles else None)
        result = self.backend(positions=positions, cell=cell, batch=batch,
                              latent_charges=charges, latent_dipoles=dipoles,
                              latent_alphas=alphas)
        energy = result["E_lr"].reshape(-1)
        short = data[properties.energy]
        if short.numel() != energy.numel():
            raise ValueError("LES supports a single total-energy head per structure")
        data["les_energy"] = energy
        data["les_latent_charges"] = charges
        if dipoles is not None:
            data["les_permanent_dipoles"] = dipoles
        if alphas is not None:
            data["les_polarizability"] = alphas
            data["les_total_dipoles"] = result["latent_dipoles"]
        data[properties.energy] = short + energy.reshape_as(short)
        return data


class PolarizableLatentEwaldEnergy(LatentEwaldEnergy):
    """Official rank-1 extended LES: permanent dipoles and induced dipoles.

    The upstream response is alpha times the permanent-source field, not an
    iterative/self-consistent core-shell PQEq solve. No quadrupoles are included.
    """
    def __init__(self, **kwargs):
        kwargs.setdefault("dipoles", True)
        kwargs.setdefault("induced_dipoles", True)
        super().__init__(**kwargs)
