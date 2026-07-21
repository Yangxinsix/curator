import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from curator.data import AseDataset, properties
from curator.data._data_reader import AseDataReader
from curator.data.utils import iter_batches
from curator.layer.utils import find_layer_by_name_recursive
from curator.utils import save_npz, write_json


log = logging.getLogger(__name__)


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
        benchmark: bool = False,
        benchmark_warmup_batches: int = 3,
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
        self.benchmark = benchmark
        self.benchmark_warmup_batches = benchmark_warmup_batches

    def _init_reader(self) -> AseDataReader:
        if isinstance(self.data_reader, AseDataReader):
            return self.data_reader
        rep = getattr(self.model, "representation", None)
        cutoff = float(rep.cutoff) if rep is not None and hasattr(rep, "cutoff") else None
        if cutoff is None:
            cutoff_layer = find_layer_by_name_recursive(self.model, "cutoff")
            if cutoff_layer is not None and hasattr(cutoff_layer, "cutoff"):
                cutoff = float(cutoff_layer.cutoff)
        if callable(self.data_reader):
            if cutoff is None:
                raise RuntimeError("Failed to infer model cutoff; please pass a data_reader with cutoff.")
            try:
                return self.data_reader(cutoff=cutoff)
            except TypeError:
                return self.data_reader(cutoff)
        if cutoff is None:
            raise RuntimeError("Failed to infer model cutoff; please pass a data_reader with cutoff.")
        return AseDataReader(cutoff=cutoff, compute_neighbor_list=True)

    def _resolve_output_dir(self, datapath: Union[str, List[str], Dataset, DataLoader]) -> Path:
        if isinstance(datapath, (str, Path)):
            tag = Path(datapath).stem or "dataset"
        elif isinstance(datapath, list) and datapath:
            tag = Path(datapath[0]).stem if len(datapath) == 1 else "multi_dataset"
        elif isinstance(datapath, DataLoader):
            tag = datapath.dataset.__class__.__name__ if hasattr(datapath, "dataset") else "dataloader"
        elif isinstance(datapath, Dataset):
            tag = datapath.__class__.__name__
        else:
            tag = "dataset"
        return (self.output_dir / tag).resolve()

    def _iter_dataset_transforms(self, dataset) -> Iterable[object]:
        seen = set()

        def visit(obj):
            obj_id = id(obj)
            if obj_id in seen:
                return
            seen.add(obj_id)
            for transform in getattr(obj, "transforms", []) or []:
                yield transform
            nested = getattr(obj, "dataset", None)
            if nested is not None:
                yield from visit(nested)

        yield from visit(dataset)

    def evaluate(
        self,
        datapath: Union[str, List[str], Dataset, DataLoader],
        sqlite_output: Optional[Union[str, Path]] = None,
        batch_columns: Optional[Dict[str, str]] = None,
        output_columns: Optional[Dict[str, str]] = None,
        sqlite_append: bool = False,
        start_index: int = 0,
    ) -> None:
        out_dir = self._resolve_output_dir(datapath)
        out_dir.mkdir(parents=True, exist_ok=True)
        if sqlite_output is None:
            resolved_sqlite = None
            if batch_columns or output_columns:
                raise ValueError("`sqlite_output` is required when sqlite column mappings are provided.")
        else:
            resolved_sqlite = Path(sqlite_output).expanduser()
            if not resolved_sqlite.is_absolute():
                resolved_sqlite = out_dir / resolved_sqlite
        labels = self.get_labels(
            datapath,
            sqlite_output=resolved_sqlite,
            batch_columns=batch_columns,
            output_columns=output_columns,
            sqlite_append=sqlite_append,
            start_index=start_index,
        )
        metrics = self.calculate_metrics(labels)
        self.save_results(out_dir, labels, metrics)
        if self.plot_figure:
            self._plot_results(out_dir, labels, metrics)
        self.labels = labels
        self.metrics = metrics

    def get_labels(
        self,
        datapath: Union[str, List[str], Dataset, DataLoader],
        sqlite_output: Optional[Union[str, Path]] = None,
        batch_columns: Optional[Dict[str, str]] = None,
        output_columns: Optional[Dict[str, str]] = None,
        sqlite_append: bool = False,
        start_index: int = 0,
    ) -> Dict[str, Any]:
        reader = self._init_reader()
        batch_columns = {} if batch_columns is None else {str(key): str(value) for key, value in batch_columns.items()}
        output_columns = {} if output_columns is None else {str(key): str(value) for key, value in output_columns.items()}
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
        labels = {
            "num_structures": 0,
            "energy_true": [],
            "energy_pred": [],
            "forces_true": [],
            "forces_pred": [],
            "atomic_numbers": [],
            "n_atoms": [],
        }
        benchmark_seconds = 0.0
        benchmark_batches = 0
        benchmark_structures = 0

        model_outputs = getattr(self.model, "model_outputs", []) or []
        needs_grad = any(
            key in model_outputs
            for key in (properties.forces, properties.energy_hessian, properties.energy_hessian_sampled)
        )
        if needs_grad:
            from curator.data._neighborlist import NeighborListTransform

            for transform in getattr(reader, "transforms", []) or []:
                if isinstance(transform, NeighborListTransform):
                    transform.requires_grad = True

        owned_dataset = None
        if isinstance(datapath, (Dataset, DataLoader)):
            dataset = datapath
            if needs_grad:
                from curator.data._neighborlist import NeighborListTransform

                for transform in self._iter_dataset_transforms(dataset):
                    if isinstance(transform, NeighborListTransform):
                        transform.requires_grad = True
        else:
            dataset = AseDataset(
                datapath,
                cutoff=reader.cutoff,
                compute_neighbor_list=reader.compute_neighbor_list,
                transforms=reader.transforms,
                default_dtype=reader.default_dtype,
                task="ase",
                return_atoms_data=True,
            )
            owned_dataset = dataset
        if start_index:
            if start_index < 0 or start_index > len(dataset):
                raise ValueError(
                    f"start_index must be between 0 and {len(dataset)}, got {start_index}."
                )
            dataset = torch.utils.data.Subset(dataset, range(start_index, len(dataset)))
        labels["num_structures"] = len(dataset) if hasattr(dataset, "__len__") else 0
        desc = f"evaluate size={labels['num_structures'] if labels['num_structures'] else '?'} bs={self.batch_size}"

        # create SQLite database and prepare for writing if needed
        db = None
        if sqlite_output is not None:
            import apsw
            from curator.data.sql_database import QMDatabase, STANDARD_COLUMN_SPECS

            extra_columns = {}
            used_columns = set()
            for name, mapping in (("batch_columns", batch_columns), ("output_columns", output_columns)):
                for source_key, column_name in mapping.items():
                    if source_key not in STANDARD_COLUMN_SPECS:
                        raise KeyError(
                            f"{name} source '{source_key}' is not supported. "
                            f"Known keys: {sorted(STANDARD_COLUMN_SPECS)}"
                        )
                    if column_name in STANDARD_COLUMN_SPECS:
                        if column_name != source_key:
                            raise ValueError(
                                f"{name} target '{column_name}' conflicts with a standard SQLite column."
                            )
                    elif column_name in used_columns:
                        raise ValueError(f"Duplicate SQLite column mapping for '{column_name}'.")
                    used_columns.add(column_name)
                    if column_name not in STANDARD_COLUMN_SPECS:
                        extra_columns[column_name] = dict(STANDARD_COLUMN_SPECS[source_key])

            sqlite_output = Path(sqlite_output).expanduser()
            if sqlite_output.exists():
                if not sqlite_append:
                    raise FileExistsError(f"SQLite file already exists: {sqlite_output}")
                db = QMDatabase(
                    str(sqlite_output),
                    flags=apsw.SQLITE_OPEN_READWRITE,
                    extra_columns=extra_columns,
                )
            else:
                sqlite_output.parent.mkdir(parents=True, exist_ok=True)
                db = QMDatabase(
                    str(sqlite_output),
                    flags=apsw.SQLITE_OPEN_READWRITE | apsw.SQLITE_OPEN_CREATE,
                    extra_columns=extra_columns,
                )
        elif batch_columns or output_columns:
            raise ValueError("`sqlite_output` is required when sqlite column mappings are provided.")

        with torch.set_grad_enabled(needs_grad):
            for batch_index, batch in enumerate(iter_batches(
                dataset=dataset,
                batch_size=self.batch_size,
                device=model_device,
                dtype=model_dtype,
                desc=desc,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )):
                if needs_grad:
                    if properties.positions in batch:
                        batch[properties.positions].requires_grad_()
                    if properties.edge_diff in batch:
                        batch[properties.edge_diff].requires_grad_()

                true_energy = batch.get(properties.energy)
                if isinstance(true_energy, torch.Tensor):
                    true_energy = true_energy.detach().cpu().numpy().reshape(-1)
                true_forces = batch.get(properties.forces)
                if isinstance(true_forces, torch.Tensor):
                    true_forces = true_forces.detach().cpu().numpy()

                if self.benchmark and model_device.type == "cuda":
                    torch.cuda.synchronize(model_device)
                start = time.perf_counter() if self.benchmark else None
                outputs = self.model(batch)
                if self.benchmark and model_device.type == "cuda":
                    torch.cuda.synchronize(model_device)
                if self.benchmark and batch_index >= self.benchmark_warmup_batches:
                    benchmark_seconds += time.perf_counter() - start
                    benchmark_batches += 1
                    benchmark_structures += int(batch[properties.n_atoms].numel())
                if not isinstance(outputs, dict):
                    raise RuntimeError("Model output must be a dict with energy/forces.")
                if db is not None:
                    self.write_sqlite(
                        db,
                        batch,
                        outputs,
                        batch_columns=batch_columns,
                        output_columns=output_columns,
                    )

                pred_energy = outputs.get(properties.energy)
                if isinstance(pred_energy, torch.Tensor):
                    pred_energy = pred_energy.detach().cpu().numpy().reshape(-1)
                pred_forces = outputs.get(properties.forces)
                if isinstance(pred_forces, torch.Tensor):
                    pred_forces = pred_forces.detach().cpu().numpy()

                if pred_energy is not None and true_energy is not None:
                    labels["energy_true"].extend(true_energy.tolist())
                    labels["energy_pred"].extend(pred_energy.tolist())
                    labels["n_atoms"].extend(batch[properties.n_atoms].detach().cpu().view(-1).tolist())

                if pred_forces is not None and true_forces is not None:
                    labels["forces_true"].append(true_forces)
                    labels["forces_pred"].append(pred_forces)
                    if properties.Z in batch:
                        labels["atomic_numbers"].append(batch[properties.Z].detach().cpu().numpy())

        if benchmark_structures:
            labels["throughput"] = {
                "warmup_batches": self.benchmark_warmup_batches,
                "timed_batches": benchmark_batches,
                "timed_structures": benchmark_structures,
                "seconds": benchmark_seconds,
            }
        if owned_dataset is not None and hasattr(owned_dataset.db, "close"):
            owned_dataset.db.close()
        return labels

    def calculate_metrics(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        metrics = {
            "num_structures": labels["num_structures"] if labels["num_structures"] else len(labels["energy_true"])
        }
        if labels["energy_true"]:
            e_true = np.asarray(labels["energy_true"])
            e_pred = np.asarray(labels["energy_pred"])
            e_err = e_pred - e_true
            metrics["energy"] = {
                "mae": float(np.mean(np.abs(e_err))),
                "rmse": float(np.sqrt(np.mean(e_err ** 2))),
                "count": int(e_true.size),
            }
            if len(labels["n_atoms"]) == e_true.size:
                n_atoms = np.asarray(labels["n_atoms"])
                metrics["energy"]["mae_per_atom"] = float(np.mean(np.abs(e_err) / n_atoms))
                metrics["energy"]["rmse_per_atom"] = float(np.sqrt(np.mean((e_err / n_atoms) ** 2)))
        else:
            log.warning("No reference energies found; skipping energy metrics/plots.")

        if labels["forces_true"]:
            f_true = np.concatenate(labels["forces_true"], axis=0)
            f_pred = np.concatenate(labels["forces_pred"], axis=0)
            f_err = f_pred - f_true
            f_norm = np.linalg.norm(f_err, axis=1)
            metrics["forces"] = {
                "mae": float(np.mean(np.abs(f_err))),
                "rmse": float(np.sqrt(np.mean(f_err ** 2))),
                "norm_mae": float(np.mean(f_norm)),
                "norm_rmse": float(np.sqrt(np.mean(f_norm ** 2))),
                "count": int(f_true.shape[0]),
            }

            if labels["atomic_numbers"]:
                z_all = np.concatenate(labels["atomic_numbers"], axis=0)
                per_elem = {}
                for z in sorted(set(z_all.tolist())):
                    mask = z_all == z
                    if np.any(mask):
                        per_elem[int(z)] = float(np.mean(f_norm[mask]))
                if per_elem:
                    metrics["forces"]["norm_mae_by_element"] = per_elem
        else:
            log.warning("No reference forces found; skipping force metrics/plots.")

        if labels.get("throughput"):
            timing = labels["throughput"]
            samples_per_second = timing["timed_structures"] / timing["seconds"]
            metrics["throughput"] = {
                **timing,
                "samples_per_second": samples_per_second,
                "ns_per_day_at_1fs": samples_per_second * 0.0864,
            }

        return metrics

    def save_results(self, out_dir: Path, labels: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        write_json(out_dir / "metrics.json", metrics, indent=2)

        if self.save_data:
            payload = {}
            if labels["energy_true"]:
                payload["energy_true"] = np.asarray(labels["energy_true"])
                payload["energy_pred"] = np.asarray(labels["energy_pred"])
            if labels["forces_true"]:
                payload["forces_true"] = np.concatenate(labels["forces_true"], axis=0)
                payload["forces_pred"] = np.concatenate(labels["forces_pred"], axis=0)
                if labels["atomic_numbers"]:
                    payload["atomic_numbers"] = np.concatenate(labels["atomic_numbers"], axis=0)
            if payload:
                save_npz(out_dir / "results.npz", **payload)

    def write_sqlite(
        self,
        db,
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        batch_columns: Optional[Dict[str, str]] = None,
        output_columns: Optional[Dict[str, str]] = None,
    ) -> None:
        import apsw
        from curator.data.sql_database import STANDARD_COLUMN_SPECS
        from curator.model.utils import extract_cells

        batch_columns = {} if batch_columns is None else batch_columns
        output_columns = {} if output_columns is None else output_columns
        n_atoms = batch[properties.n_atoms].view(-1).to(torch.long)
        cells = extract_cells(batch, int(n_atoms.shape[0]))
        available_batch_columns = {
            properties.atomic_numbers,
            properties.positions,
            properties.pbc,
        }
        if cells is not None:
            available_batch_columns.add(properties.cell)
        available_batch_columns.update(
            key for key in STANDARD_COLUMN_SPECS
            if key not in {properties.atomic_numbers, properties.positions, properties.pbc, properties.cell}
            and key in batch
        )
        missing_batch = [key for key in batch_columns if key not in available_batch_columns]
        if missing_batch:
            raise KeyError(
                f"Batch is missing sqlite export source keys: {missing_batch}. "
                f"Available keys: {sorted(available_batch_columns)}"
            )
        missing_output = [key for key in output_columns if key not in outputs]
        if missing_output:
            raise KeyError(
                f"Model output is missing sqlite export source keys: {missing_output}. "
                f"Available keys: {sorted(outputs.keys())}"
            )

        def get_structure_value(source: Dict[str, torch.Tensor], key: str, idx: int, atom_offset: int, count: int):
            spec = STANDARD_COLUMN_SPECS[key]
            value = source[key]
            if spec["storage"] == "scalar":
                scalar = value[idx]
                if isinstance(scalar, torch.Tensor):
                    scalar = scalar.detach().cpu().view(-1)[0].item()
                else:
                    scalar = np.asarray(scalar).reshape(-1)[0].item()
                if spec["dtype"] == "bool":
                    return int(bool(scalar))
                if np.issubdtype(np.dtype(spec["dtype"]), np.integer):
                    return int(scalar)
                return float(scalar)

            if spec.get("shape") is not None and "n_atoms" in spec["shape"]:
                index = tuple(
                    slice(atom_offset, atom_offset + count) if dim == "n_atoms" else slice(None)
                    for dim in spec["shape"]
                )
                value = value[index]
            else:
                value = value[idx]
                if isinstance(value, torch.Tensor) and value.dim() == 1 and spec.get("shape", [None])[0] == 1:
                    value = value.unsqueeze(0)
                elif not isinstance(value, torch.Tensor):
                    value = np.asarray(value)
                    if value.ndim == 1 and spec.get("shape", [None])[0] == 1:
                        value = value[None, :]

            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            else:
                value = np.asarray(value)
            if spec.get("shape") == ["n_atoms"] and value.ndim == 2 and value.shape[-1] == 1:
                value = value.reshape(-1)
            return value

        cursor = db._get_connection(flags=apsw.SQLITE_OPEN_READWRITE).cursor()
        cursor.execute('''BEGIN EXCLUSIVE''')
        atom_offset = 0

        try:
            for idx, count_tensor in enumerate(n_atoms):
                count = int(count_tensor.item())
                row = {
                    properties.atomic_numbers: get_structure_value(batch, properties.atomic_numbers, idx, atom_offset, count),
                    properties.positions: get_structure_value(batch, properties.positions, idx, atom_offset, count),
                    properties.pbc: int(cells is not None),
                }

                if cells is not None:
                    row[properties.cell] = cells[idx].detach().cpu().numpy()
                for key in STANDARD_COLUMN_SPECS:
                    if key in {properties.atomic_numbers, properties.positions, properties.pbc, properties.cell}:
                        continue
                    if key in batch:
                        row[key] = get_structure_value(batch, key, idx, atom_offset, count)

                for source_key, column_name in batch_columns.items():
                    value = row[source_key]
                    row[column_name] = value.copy() if isinstance(value, np.ndarray) else value
                for source_key, column_name in output_columns.items():
                    row[column_name] = get_structure_value(outputs, source_key, idx, atom_offset, count)

                db.add_data(row, flags=apsw.SQLITE_OPEN_READWRITE, transaction=False)
                atom_offset += count
        except Exception:
            cursor.execute('''ROLLBACK''')
            raise

        cursor.execute('''COMMIT''')

    def _plot_results(
        self,
        out_dir: Path,
        labels: Dict[str, Any],
        metrics: dict,
    ) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("matplotlib is required for plotting.") from exc

        def label_with_unit(label: str, unit: Optional[str]) -> str:
            return f"{label} ({unit})" if unit else label

        if labels["energy_true"]:
            e_true = np.asarray(labels["energy_true"])
            e_pred = np.asarray(labels["energy_pred"])
            vmin = min(e_true.min(), e_pred.min())
            vmax = max(e_true.max(), e_pred.max())
            pad = 0.05 * (vmax - vmin) if vmax > vmin else 1.0

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(e_true, e_pred, s=10, alpha=0.6, color="black")
            ax.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad], color="gray", linestyle="--")
            ax.set_xlabel(label_with_unit("True Energy", self.energy_unit))
            ax.set_ylabel(label_with_unit("Pred Energy", self.energy_unit))
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
            ax.set_xlabel(label_with_unit("Energy Error", self.energy_unit))
            ax.set_ylabel("Count")
            fig.tight_layout()
            fig.savefig(out_dir / "hist_energy_error.png", dpi=150)
            plt.close(fig)

        if labels["forces_true"]:
            f_true = np.concatenate(labels["forces_true"], axis=0)
            f_pred = np.concatenate(labels["forces_pred"], axis=0)
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
            ax.set_xlabel(label_with_unit("True Force", self.force_unit))
            ax.set_ylabel(label_with_unit("Pred Force", self.force_unit))
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
            ax.set_xlabel(label_with_unit("Force Error Norm", self.force_unit))
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
                ax.set_ylabel(label_with_unit("Force Norm MAE", self.force_unit))
                fig.tight_layout()
                fig.savefig(out_dir / "bar_force_mae_by_element.png", dpi=150)
                plt.close(fig)
