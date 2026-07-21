from __future__ import annotations

from typing import Optional


ASE_MD_INTEGRATORS = {"langevin", "verlet", "npt_berendsen"}
TORCHSIM_MD_INTEGRATORS = {
    "nve",
    "nvt_langevin",
    "nvt_nose_hoover",
    "npt_langevin",
    "npt_nose_hoover",
}
ALLOWED_INTEGRATORS = ASE_MD_INTEGRATORS | TORCHSIM_MD_INTEGRATORS
SUPPORTED_ASE_MD_ENSEMBLES = {"nvt", "nve", "npt"}
SUPPORTED_TORCHSIM_MD_ENSEMBLES = {"nvt", "nve", "npt"}


def ase_md_integrator_for_ensemble(ensemble: str, requested_integrator: Optional[str]) -> Optional[str]:
    normalized = str(ensemble or "nvt").strip().lower()
    if normalized == "nve":
        return "verlet"
    if normalized == "npt":
        integrator = str(requested_integrator or "npt_berendsen").strip().lower()
        if integrator != "npt_berendsen":
            raise ValueError("ASE NPT direct-use simulation cases currently support integrator='npt_berendsen' only.")
        return integrator
    if normalized == "nvt":
        integrator = str(requested_integrator or "langevin").strip().lower()
        if integrator not in {"langevin", "verlet"}:
            raise ValueError("NVT simulation cases currently support integrator='langevin' or 'verlet'.")
        return integrator
    return None


def torchsim_md_integrator_for_ensemble(
    ensemble: str,
    requested_integrator: Optional[str],
    *,
    thermostat: Optional[str] = None,
    barostat: Optional[str] = None,
) -> Optional[str]:
    normalized = str(ensemble or "nvt").strip().lower()
    requested = str(requested_integrator).strip().lower() if requested_integrator else None
    if requested:
        if requested not in TORCHSIM_MD_INTEGRATORS:
            raise ValueError(
                f"TorchSim integrator must be one of {sorted(TORCHSIM_MD_INTEGRATORS)}, got {requested!r}."
            )
        if normalized == "nve" and requested != "nve":
            raise ValueError("TorchSim NVE simulation cases require integrator='nve'.")
        if normalized == "nvt" and not requested.startswith("nvt_"):
            raise ValueError("TorchSim NVT simulation cases require an nvt_* integrator.")
        if normalized == "npt" and not requested.startswith("npt_"):
            raise ValueError("TorchSim NPT simulation cases require an npt_* integrator.")
        return requested

    if normalized == "nve":
        return "nve"
    if normalized == "nvt":
        thermo = str(thermostat or "langevin").strip().lower()
        return "nvt_nose_hoover" if thermo in {"nose_hoover", "nose-hoover", "nosehoover"} else "nvt_langevin"
    if normalized == "npt":
        baro = str(barostat or "langevin").strip().lower()
        return "npt_nose_hoover" if baro in {"nose_hoover", "nose-hoover", "nosehoover"} else "npt_langevin"
    return None


def md_integrator_for_backend(
    backend: str,
    ensemble: str,
    requested_integrator: Optional[str],
    *,
    thermostat: Optional[str] = None,
    barostat: Optional[str] = None,
) -> Optional[str]:
    normalized_backend = str(backend).strip().lower()
    if normalized_backend == "ase":
        return ase_md_integrator_for_ensemble(ensemble, requested_integrator)
    if normalized_backend == "torchsim":
        return torchsim_md_integrator_for_ensemble(
            ensemble,
            requested_integrator,
            thermostat=thermostat,
            barostat=barostat,
        )
    return None
