from __future__ import annotations

from curator.simulate.core.callbacks import Callback
from curator.simulate.core.context import SimContext


class VelocityInitializer(Callback):
    """Initialize ASE velocities from a target temperature before MD setup."""

    def __init__(
        self,
        temperature_K: float,
        *,
        force: bool = False,
        remove_translation: bool = True,
        remove_rotation: bool = True,
    ) -> None:
        self.temperature_K = float(temperature_K)
        self.force = bool(force)
        self.remove_translation = bool(remove_translation)
        self.remove_rotation = bool(remove_rotation)

    def on_sim_start(self, ctx: SimContext) -> None:
        atoms = ctx.atoms
        if atoms is None or isinstance(atoms, list):
            return
        if not self.force and atoms.has("momenta"):
            return

        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

        MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature_K, force_temp=True)
        if self.remove_translation:
            Stationary(atoms)
        if self.remove_rotation:
            ZeroRotation(atoms)
        ctx.state["velocity_initialization"] = {
            "temperature_K": self.temperature_K,
            "force": self.force,
            "remove_translation": self.remove_translation,
            "remove_rotation": self.remove_rotation,
        }
