from __future__ import annotations

from typing import Any, Dict, List

from .base import SimulationBackend, SimulationBackendCapabilities, dependency_available
from ..spec import SimulationCaseSpec, SimulationRequest


class LammpsSimulationBackend(SimulationBackend):
    name = "lammps"

    def capabilities(self) -> SimulationBackendCapabilities:
        return SimulationBackendCapabilities(
            backend=self.name,
            available=dependency_available("lammps"),
            implemented=False,
            tasks=["production_md", "deployment_equivalence"],
            ensembles=["nve", "nvt", "npt"],
            integrators=[],
            devices=["cpu", "cuda"],
            dtypes=[],
            batching=False,
            autobatching=False,
            online_uncertainty=False,
            requires_deployed_model=True,
            notes=[
                "Curator has deployment and MLIAP interface code, but this backend contract is not implemented yet.",
                "LAMMPS should enter through scheduler-aware submit/status/collect support, not local ASE/TorchSim execution.",
            ],
        )

    def plan_cases(self, request: SimulationRequest) -> List[SimulationCaseSpec]:
        self.validate_request(request)
        raise NotImplementedError("LAMMPS simulation backend is not implemented.")

    def build_config(self, case: SimulationCaseSpec) -> Dict[str, Any]:
        raise NotImplementedError("LAMMPS simulation backend is not implemented.")

    def run_case(self, case: SimulationCaseSpec, *, tool_name: str = "run_simulation") -> Dict[str, Any]:
        raise NotImplementedError("LAMMPS simulation backend is not implemented.")
