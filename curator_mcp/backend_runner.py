from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Mapping, Optional

from curator_mcp.runners.artifacts import ensure_dir, error_result, read_json, utc_now, write_json


_MODULES_BY_RUNTIME = {
    "mace": ["mace"],
    "matgl": ["matgl"],
    "nequip": ["nequip"],
    "fairchem": ["fairchem"],
    "orb": ["orb_models"],
    "sevennet": ["sevenn"],
    "mattersim": ["mattersim"],
}

_RUNTIME_BY_SCHEME = {
    "mace": "mace",
    "matgl": "matgl",
    "nequip": "nequip",
    "nequip_hf": "nequip",
    "nequip_net": "nequip",
    "allegro": "nequip",
    "allegro_net": "nequip",
    "esen": "fairchem",
    "eqv2": "fairchem",
    "orb": "orb",
    "sevennet": "sevennet",
    "mattersim": "mattersim",
}

_ATOM_READER_PARAM_KEYS = {"atoms_index", "max_structures"}


def _runtime_from_request(request: Mapping[str, Any]) -> Optional[str]:
    runtime = request.get("runtime")
    if runtime:
        normalized_runtime = str(runtime).strip().lower()
        if normalized_runtime in _MODULES_BY_RUNTIME:
            return normalized_runtime
    backend = request.get("backend")
    if backend:
        normalized = str(backend).strip().lower()
        if normalized in _MODULES_BY_RUNTIME:
            return normalized
        runtime = _RUNTIME_BY_SCHEME.get(normalized)
        if runtime:
            return runtime
    spec_text = request.get("adapter_spec") or request.get("model_id")
    if isinstance(spec_text, str):
        scheme = spec_text.split(":", 1)[0].strip().lower() if ":" in spec_text else ""
        if scheme:
            return _RUNTIME_BY_SCHEME.get(scheme)
    return None


def _dependency_status(runtime: Optional[str]) -> dict[str, Any]:
    modules = list(_MODULES_BY_RUNTIME.get(str(runtime or ""), []))
    results = {
        module: importlib.util.find_spec(module) is not None
        for module in modules
    }
    missing = [module for module, ok in results.items() if not ok]
    return {
        "runtime": runtime,
        "required_modules": modules,
        "modules": results,
        "dependency_ok": not missing,
        "missing_modules": missing,
    }


def _model_id_for_request(request: Mapping[str, Any]) -> str:
    model_id = request.get("adapter_spec") or request.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("Backend request requires adapter_spec or model_id.")
    return model_id


def _fetch_params(request: Mapping[str, Any]) -> dict[str, Any]:
    params = {
        key: value
        for key, value in dict(request.get("params") or {}).items()
        if key not in _ATOM_READER_PARAM_KEYS
    }
    if "download" not in params:
        params["download"] = "true" if bool(request.get("download")) else "false"
    model_cache = request.get("model_cache")
    if model_cache and "cache_dir" not in params:
        params["cache_dir"] = str(model_cache)
    return params


def _as_float_or_list(value: Any) -> Any:
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.reshape(-1)[0].item())
        return value.tolist()
    try:
        import numpy as np
    except Exception:
        np = None
    if np is not None and isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.reshape(-1)[0])
        return value.tolist()
    return value


def _read_atoms(path: str, params: Mapping[str, Any]) -> list[Any]:
    from ase import Atoms
    from ase.io import read

    index = params.get("atoms_index", ":")
    frames = read(str(Path(path).expanduser()), index=index)
    if isinstance(frames, Atoms):
        atoms_list = [frames]
    else:
        atoms_list = list(frames)
    max_structures = params.get("max_structures")
    if max_structures not in (None, ""):
        atoms_list = atoms_list[: int(max_structures)]
    if not atoms_list:
        raise ValueError(f"No ASE structures were read from atoms_path={path!r}.")
    return atoms_list


def _infer_cutoff(model: Any) -> float:
    candidates = (
        getattr(model, "cutoff", None),
        getattr(getattr(model, "representation", None), "cutoff", None),
        getattr(getattr(model, "model", None), "cutoff", None),
        getattr(getattr(model, "core_model", None), "cutoff", None),
    )
    for value in candidates:
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return 0.0


def _predict_with_ase_calculator(model: Any, atoms_list: list[Any]) -> list[dict[str, Any]]:
    from curator.data import properties

    calculator = getattr(model, "calculator", None)
    if calculator is None:
        raise TypeError("Model does not expose an ASE calculator.")

    outputs = set(getattr(model, "model_outputs", []) or (properties.energy, properties.forces, properties.stress))
    predictions = []
    for index, atoms in enumerate(atoms_list):
        frame = atoms.copy()
        frame.calc = calculator
        item: dict[str, Any] = {
            "index": index,
            "num_atoms": len(frame),
            "symbols": frame.get_chemical_symbols(),
        }
        if properties.energy in outputs:
            item["energy"] = float(frame.get_potential_energy())
        if properties.forces in outputs:
            item["forces"] = _as_float_or_list(frame.get_forces())
        if properties.stress in outputs:
            try:
                item["stress"] = _as_float_or_list(frame.get_stress())
            except Exception:
                item["stress"] = None
        predictions.append(item)
    return predictions


def _move_batch_to_device(data: Mapping[str, Any], device: Any, dtype: Any = None) -> dict[str, Any]:
    import torch

    moved = {}
    for key, value in data.items():
        if torch.is_tensor(value):
            if dtype is not None and value.is_floating_point():
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device)
        elif isinstance(value, Mapping):
            moved[key] = _move_batch_to_device(value, device, dtype=dtype)
        else:
            moved[key] = value
    return moved


def _model_float_dtype(model: Any) -> Any:
    import torch

    for values in (getattr(model, "parameters", None), getattr(model, "buffers", None)):
        if not callable(values):
            continue
        for value in values():
            if torch.is_tensor(value) and value.is_floating_point():
                return value.dtype
    return torch.get_default_dtype()


def _predict_with_curator_forward(model: Any, atoms_list: list[Any], device_name: str) -> list[dict[str, Any]]:
    import torch

    from curator.data import properties
    from curator.data._data_reader import AseDataReader
    from curator.data.collate_atoms_data import collate_atoms_data

    device = torch.device(device_name)
    dtype = _model_float_dtype(model)
    cutoff = _infer_cutoff(model)
    reader = AseDataReader(
        cutoff=cutoff if cutoff > 0.0 else None,
        compute_neighbor_list=cutoff > 0.0,
        return_cell_displacements=True,
        default_dtype=dtype,
    )
    samples = []
    for atoms in atoms_list:
        frame = atoms.copy()
        frame.calc = None
        sample = reader(frame)
        sample[properties.pbc] = torch.as_tensor(frame.get_pbc(), dtype=torch.bool).view(1, 3)
        samples.append(sample)
    batch = _move_batch_to_device(collate_atoms_data(samples), device, dtype=dtype)

    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    with torch.enable_grad():
        output = model(batch)
    if isinstance(output, Mapping):
        batch.update(output)

    n_atoms = batch[properties.n_atoms].detach().cpu().view(-1).to(torch.long).tolist()
    energies = batch.get(properties.energy)
    forces = batch.get(properties.forces)
    stresses = batch.get(properties.stress)
    if energies is None and forces is None and stresses is None:
        raise RuntimeError(
            "The loaded model did not write energy, forces, or stress to the Curator batch."
        )

    predictions = []
    atom_offset = 0
    for index, atoms in enumerate(atoms_list):
        count = int(n_atoms[index])
        item: dict[str, Any] = {
            "index": index,
            "num_atoms": count,
            "symbols": atoms.get_chemical_symbols(),
        }
        if energies is not None:
            item["energy"] = _as_float_or_list(energies[index])
        if forces is not None:
            item["forces"] = _as_float_or_list(forces[atom_offset : atom_offset + count])
        if stresses is not None:
            item["stress"] = _as_float_or_list(stresses[index])
        predictions.append(item)
        atom_offset += count
    return predictions


def _predict_model(adapter_spec: str, atoms_path: str, params: Mapping[str, Any], device: str) -> dict[str, Any]:
    import torch

    from curator.model.adapter import load_external_model

    atoms_list = _read_atoms(atoms_path, params)
    model = load_external_model(adapter_spec, device=torch.device(device))
    try:
        predictions = _predict_with_ase_calculator(model, atoms_list)
        path = "ase_calculator"
    except TypeError:
        predictions = _predict_with_curator_forward(model, atoms_list, device)
        path = "curator_forward"

    return {
        "adapter_spec": adapter_spec,
        "atoms_path": str(Path(atoms_path).expanduser()),
        "num_structures": len(atoms_list),
        "prediction_path": path,
        "predictions": predictions,
    }


def run_request(request: Mapping[str, Any], out: str) -> dict[str, Any]:
    run_dir = ensure_dir(out)
    result_path = run_dir / "result.json"
    task = str(request.get("task") or "probe").strip().lower()
    runtime = _runtime_from_request(request)
    dependencies = _dependency_status(runtime)

    base = {
        "runtime": runtime,
        "task": task,
        "created_at": utc_now(),
        "dependency": dependencies,
        "artifacts": {"result": str(result_path)},
    }

    try:
        if task == "health":
            result = {
                **base,
                "ok": dependencies["dependency_ok"],
                "status": "completed" if dependencies["dependency_ok"] else "dependency_missing",
            }
            write_json(result_path, result)
            return result

        if task not in {"resolve", "fetch", "probe", "predict"}:
            raise ValueError("task must be one of: health, resolve, fetch, probe, predict.")

        if task in {"probe", "predict"} and not dependencies["dependency_ok"]:
            result = {
                **base,
                "ok": False,
                "status": "dependency_missing",
                "error_info": {
                    "type": "DependencyMissing",
                    "message": "Missing backend runtime modules: " + ", ".join(dependencies["missing_modules"]),
                    "recoverable": True,
                    "log_path": None,
                },
            }
            write_json(result_path, result)
            return result

        model_id = _model_id_for_request(request)
        params = _fetch_params(request)
        from curator.model.foundation import fetch_model

        fetch_result = fetch_model(
            model_id,
            params=params,
            out=str(run_dir / "model_fetch"),
            cache_dir=request.get("model_cache"),
            download=bool(request.get("download")),
            probe=task == "probe",
            device=str(request.get("device") or "cpu"),
            timeout_sec=int(request.get("timeout_sec") or 300),
        )
        if task == "predict":
            if not fetch_result.get("ok"):
                result = {
                    **base,
                    "ok": False,
                    "status": fetch_result.get("status", "model_fetch_failed"),
                    "adapter_spec": fetch_result.get("adapter_spec"),
                    "model": fetch_result,
                    "error_info": fetch_result.get("error") or fetch_result.get("error_info"),
                }
                write_json(result_path, result)
                return result
            atoms_path = request.get("atoms_path")
            if not isinstance(atoms_path, str) or not atoms_path:
                raise ValueError("predict task requires atoms_path.")
            prediction = _predict_model(
                str(fetch_result["adapter_spec"]),
                atoms_path,
                params,
                str(request.get("device") or "cpu"),
            )
            prediction_path = run_dir / "predictions.json"
            write_json(prediction_path, prediction)
            result = {
                **base,
                "ok": True,
                "status": "completed",
                "adapter_spec": fetch_result.get("adapter_spec"),
                "model": fetch_result,
                "prediction": prediction,
                "artifacts": {
                    **base["artifacts"],
                    "predictions": str(prediction_path),
                },
            }
            write_json(result_path, result)
            return result

        result = {
            **base,
            "ok": bool(fetch_result.get("ok")),
            "status": fetch_result.get("status", "completed" if fetch_result.get("ok") else "failed"),
            "adapter_spec": fetch_result.get("adapter_spec"),
            "model": fetch_result,
        }
        write_json(result_path, result)
        return result
    except Exception as exc:
        result = error_result(
            exc.__class__.__name__,
            str(exc),
            artifacts={"result": str(result_path)},
            runtime=runtime,
            task=task,
            dependency=dependencies,
        )
        write_json(result_path, result)
        return result


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run an isolated Curator foundation-model backend task.")
    parser.add_argument("--request", required=True, help="Path to backend request JSON.")
    parser.add_argument("--out", required=True, help="Output directory for result.json and artifacts.")
    args = parser.parse_args(argv)

    request = read_json(args.request)
    result = run_request(request, args.out)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
