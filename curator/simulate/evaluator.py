import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
from ase import Atoms

from curator.data import AseDataset, properties
from curator.data._data_reader import AseDataReader
from curator.data.utils import iter_batches, read_trajectory, _prepare_data_source
from curator.layer.utils import find_layer_by_name_recursive
from curator.utils import save_npz, write_json

log = logging.getLogger(__name__)


def _infer_cutoff(model) -> Optional[float]:
    rep = getattr(model, "representation", None)
    if rep is not None and hasattr(rep, "cutoff"):
        return float(rep.cutoff)
    cutoff_layer = find_layer_by_name_recursive(model, "cutoff")
    if cutoff_layer is not None and hasattr(cutoff_layer, "cutoff"):
        return float(cutoff_layer.cutoff)
    return None


class Evaluator:
    def __init__(
        self,
        model,
        data_reader: Optional[object] = None,
        save_data: bool = False,
        plot_figure: bool = True,
        output_dir: Union[str, Path] = "evaluate",
        energy_unit: Optional[str] = "eV",
        force_unit: Optional[str] = "eV/Angstrom",
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: Optional[bool] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.data_reader = data_reader
        self.save_data = save_data
        self.plot_figure = plot_figure
        self.output_dir = Path(output_dir)
        self.energy_unit = energy_unit
        self.force_unit = force_unit
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.device = device

    def _init_reader(self) -> AseDataReader:
        cutoff = _infer_cutoff(self.model)
        if cutoff is None:
            raise RuntimeError("Failed to infer model cutoff; please pass a data_reader with cutoff.")
        if isinstance(self.data_reader, AseDataReader):
            return self.data_reader
        if callable(self.data_reader):
            try:
                return self.data_reader(cutoff=cutoff)
            except TypeError:
                return self.data_reader(cutoff)
        return AseDataReader(cutoff=cutoff, compute_neighbor_list=True)

    def _resolve_output_dir(self, datapath: Union[str, Sequence[str], Iterable[Atoms]]) -> Path:
        tag = "dataset"
        if isinstance(datapath, (str, Path)):
            tag = Path(datapath).stem or "dataset"
        elif isinstance(datapath, (list, tuple)) and datapath:
            tag = Path(datapath[0]).stem if len(datapath) == 1 else "multi_dataset"
            tag = tag or "dataset"
        return (self.output_dir / tag).resolve()

    def evaluate(self, datapath: Union[str, Sequence[str], Iterable[Atoms]]) -> None:
        reader = self._init_reader()
        out_dir = self._resolve_output_dir(datapath)
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            param = next(self.model.parameters())
            model_device = param.device
            model_dtype = param.dtype
        except Exception:
            model_device = torch.device(self.device or "cpu")
            model_dtype = torch.get_default_dtype()
        if self.device is not None:
            model_device = torch.device(self.device)
        try:
            self.model.to(model_device)
        except Exception:
            pass
        self.model.eval()

        energy_true: List[float] = []
        energy_pred: List[float] = []
        forces_true: List[np.ndarray] = []
        forces_pred: List[np.ndarray] = []
        atomic_numbers: List[np.ndarray] = []

        model_outputs = getattr(self.model, "model_outputs", []) or []
        needs_grad = properties.forces in model_outputs
        if needs_grad:
            from curator.data._neighborlist import NeighborListTransform

            for transform in getattr(reader, "transforms", []) or []:
                if isinstance(transform, NeighborListTransform):
                    transform.requires_grad = True

        dataset = AseDataset(
            datapath,
            cutoff=reader.cutoff,
            compute_neighbor_list=reader.compute_neighbor_list,
            transforms=reader.transforms,
            default_dtype=reader.default_dtype,
            task="ase",
            return_atoms_data=True,
        )
        total = len(dataset) if hasattr(dataset, "__len__") else None
        desc = f"evaluate size={total if total is not None else '?'} bs={self.batch_size}"
        with torch.set_grad_enabled(needs_grad):
            for batch in iter_batches(
                dataset=dataset,
                batch_size=self.batch_size,
                device=model_device,
                dtype=model_dtype,
                desc=desc,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            ):
                if needs_grad:
                    if properties.edge_diff in batch:
                        batch[properties.edge_diff].requires_grad_()
                    elif properties.positions in batch:
                        batch[properties.positions].requires_grad_()
                outputs = self.model(batch)
                if not isinstance(outputs, dict):
                    raise RuntimeError("Model output must be a dict with energy/forces.")

                pred_energy = outputs.get(properties.energy)
                if isinstance(pred_energy, torch.Tensor):
                    pred_energy = pred_energy.detach().cpu().numpy().reshape(-1)
                pred_forces = outputs.get(properties.forces)
                if isinstance(pred_forces, torch.Tensor):
                    pred_forces = pred_forces.detach().cpu().numpy()

                true_energy = batch.get(properties.energy)
                if isinstance(true_energy, torch.Tensor):
                    true_energy = true_energy.detach().cpu().numpy().reshape(-1)
                true_forces = batch.get(properties.forces)
                if isinstance(true_forces, torch.Tensor):
                    true_forces = true_forces.detach().cpu().numpy()

                if pred_energy is not None and true_energy is not None:
                    energy_true.extend(true_energy.tolist())
                    energy_pred.extend(pred_energy.tolist())

                if pred_forces is not None and true_forces is not None:
                    forces_true.append(true_forces)
                    forces_pred.append(pred_forces)
                    if properties.Z in batch:
                        atomic_numbers.append(batch[properties.Z].detach().cpu().numpy())

        metrics = {"num_structures": total if total is not None else len(energy_true)}
        if energy_true:
            e_true = np.asarray(energy_true)
            e_pred = np.asarray(energy_pred)
            e_err = e_pred - e_true
            metrics["energy"] = {
                "mae": float(np.mean(np.abs(e_err))),
                "rmse": float(np.sqrt(np.mean(e_err ** 2))),
                "count": int(e_true.size),
            }
        else:
            log.warning("No reference energies found; skipping energy metrics/plots.")

        if forces_true:
            f_true = np.concatenate(forces_true, axis=0)
            f_pred = np.concatenate(forces_pred, axis=0)
            f_err = f_pred - f_true
            f_norm = np.linalg.norm(f_err, axis=1)
            metrics["forces"] = {
                "mae": float(np.mean(np.abs(f_err))),
                "rmse": float(np.sqrt(np.mean(f_err ** 2))),
                "norm_mae": float(np.mean(f_norm)),
                "norm_rmse": float(np.sqrt(np.mean(f_norm ** 2))),
                "count": int(f_true.shape[0]),
            }
            if atomic_numbers:
                z_all = np.concatenate(atomic_numbers, axis=0)
                per_elem = {}
                for z in sorted(set(z_all.tolist())):
                    mask = z_all == z
                    if np.any(mask):
                        per_elem[int(z)] = float(np.mean(f_norm[mask]))
                if per_elem:
                    metrics["forces"]["norm_mae_by_element"] = per_elem
        else:
            log.warning("No reference forces found; skipping force metrics/plots.")

        write_json(out_dir / "metrics.json", metrics, indent=2)

        if self.save_data:
            payload = {}
            if energy_true:
                payload["energy_true"] = np.asarray(energy_true)
                payload["energy_pred"] = np.asarray(energy_pred)
            if forces_true:
                payload["forces_true"] = np.concatenate(forces_true, axis=0)
                payload["forces_pred"] = np.concatenate(forces_pred, axis=0)
                payload["atomic_numbers"] = np.concatenate(atomic_numbers, axis=0)
            if payload:
                save_npz(out_dir / "results.npz", **payload)

        if self.plot_figure:
            self._plot_results(out_dir, energy_true, energy_pred, forces_true, forces_pred, metrics)

    def _plot_results(
        self,
        out_dir: Path,
        energy_true: List[float],
        energy_pred: List[float],
        forces_true: List[np.ndarray],
        forces_pred: List[np.ndarray],
        metrics: dict,
    ) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise RuntimeError("matplotlib is required for plotting.") from exc

        def _label_with_unit(label: str, unit: Optional[str]) -> str:
            return f"{label} ({unit})" if unit else label

        if energy_true:
            e_true = np.asarray(energy_true)
            e_pred = np.asarray(energy_pred)
            vmin = min(e_true.min(), e_pred.min())
            vmax = max(e_true.max(), e_pred.max())
            pad = 0.05 * (vmax - vmin) if vmax > vmin else 1.0
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(e_true, e_pred, s=10, alpha=0.6, color="black")
            ax.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad], color="gray", linestyle="--")
            ax.set_xlabel(_label_with_unit("True Energy", self.energy_unit))
            ax.set_ylabel(_label_with_unit("Pred Energy", self.energy_unit))
            if "energy" in metrics:
                ax.text(
                    0.02,
                    0.98,
                    f"MAE={metrics['energy']['mae']:.4g}\nRMSE={metrics['energy']['rmse']:.4g}",
                    transform=ax.transAxes,
                    va="top",
                )
            fig.tight_layout()
            fig.savefig(out_dir / "parity_energy.png", dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(e_pred - e_true, bins=50, color="gray", alpha=0.8)
            ax.set_xlabel(_label_with_unit("Energy Error", self.energy_unit))
            ax.set_ylabel("Count")
            fig.tight_layout()
            fig.savefig(out_dir / "hist_energy_error.png", dpi=150)
            plt.close(fig)

        if forces_true:
            f_true = np.concatenate(forces_true, axis=0)
            f_pred = np.concatenate(forces_pred, axis=0)
            vmin = min(f_true.min(), f_pred.min())
            vmax = max(f_true.max(), f_pred.max())
            pad = 0.05 * (vmax - vmin) if vmax > vmin else 1.0

            fig, ax = plt.subplots(figsize=(5.5, 5))
            colors = {"x": "tab:red", "y": "tab:green", "z": "tab:blue"}
            for idx, key in enumerate(("x", "y", "z")):
                ax.scatter(
                    f_true[:, idx],
                    f_pred[:, idx],
                    s=6,
                    alpha=0.3,
                    color=colors[key],
                    label=f"F{key}",
                )
            ax.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad], color="gray", linestyle="--")
            ax.set_xlabel(_label_with_unit("True Force", self.force_unit))
            ax.set_ylabel(_label_with_unit("Pred Force", self.force_unit))
            ax.legend(frameon=False, fontsize=9)
            if "forces" in metrics:
                ax.text(
                    0.02,
                    0.98,
                    f"MAE={metrics['forces']['mae']:.4g}\nRMSE={metrics['forces']['rmse']:.4g}",
                    transform=ax.transAxes,
                    va="top",
                )
            fig.tight_layout()
            fig.savefig(out_dir / "parity_forces_xyz.png", dpi=150)
            plt.close(fig)

            f_err_norm = np.linalg.norm(f_pred - f_true, axis=1)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(f_err_norm, bins=50, color="steelblue", alpha=0.85)
            ax.set_xlabel(_label_with_unit("Force Error Norm", self.force_unit))
            ax.set_ylabel("Count")
            fig.tight_layout()
            fig.savefig(out_dir / "hist_force_error_norm.png", dpi=150)
            plt.close(fig)

            per_elem = metrics.get("forces", {}).get("norm_mae_by_element")
            if per_elem:
                elements = list(per_elem.keys())
                values = [per_elem[z] for z in elements]
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.bar([str(z) for z in elements], values, color="teal", alpha=0.85)
                ax.set_xlabel("Atomic Number")
                ax.set_ylabel(_label_with_unit("Force Norm MAE", self.force_unit))
                fig.tight_layout()
                fig.savefig(out_dir / "bar_force_mae_by_element.png", dpi=150)
                plt.close(fig)
