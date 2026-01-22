from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator


_STRUCTURE_COLUMNS = (
    "n_atoms",
    "atomic_numbers",
    "positions",
    "pbc",
    "cell",
    "energy",
    "forces",
    "virial",
    "stress",
    "dataset",
    "source_path",
    "converged",
    "label_status",
    "meta",
)

_RESERVED_META_KEYS = {
    "dataset",
    "source_path",
    "converged",
    "label_status",
    "structure_id",
}


def _pbc_to_int(pbc: Union[bool, Sequence[bool], np.ndarray]) -> int:
    pbc_arr = np.asarray(pbc, dtype=bool).reshape(-1)
    if pbc_arr.size == 1:
        pbc_arr = np.repeat(pbc_arr, 3)
    if pbc_arr.size != 3:
        raise ValueError("pbc must be a bool or length-3 sequence.")
    return int(pbc_arr[0]) | (int(pbc_arr[1]) << 1) | (int(pbc_arr[2]) << 2)


def _int_to_pbc(value: int) -> np.ndarray:
    return np.array(
        [bool(value & 1), bool(value & 2), bool(value & 4)],
        dtype=bool,
    )


def _blob(array: Optional[np.ndarray], dtype: np.dtype) -> Optional[sqlite3.Binary]:
    if array is None:
        return None
    arr = np.asarray(array, dtype=dtype)
    if not np.little_endian:
        arr = arr.byteswap()
    return sqlite3.Binary(np.ascontiguousarray(arr).tobytes())


def _deblob(buf: Optional[bytes], dtype: np.dtype, shape: Sequence[int]) -> Optional[np.ndarray]:
    if buf is None:
        return None
    arr = np.frombuffer(buf, dtype=dtype)
    if not np.little_endian:
        arr = arr.byteswap()
    arr = arr.reshape(shape)
    return np.copy(arr)


def _to_voigt_6(value: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == (6,):
        return arr
    if arr.shape == (1, 6):
        return arr.reshape(6)
    if arr.shape == (3, 3):
        return np.array(
            [arr[0, 0], arr[1, 1], arr[2, 2], arr[1, 2], arr[0, 2], arr[0, 1]],
            dtype=np.float32,
        )
    raise ValueError("stress/virial must be shape (6,), (1, 6), or (3, 3).")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


class SqliteTrajectory:
    """Lightweight SQLite-backed store for ASE Atoms with queryable metadata."""

    def __init__(self, path: Union[str, Path], mode: str = "a") -> None:
        self.path = Path(path)
        self.mode = mode
        self._conn = self._open_connection(self.path, mode)
        self._conn.row_factory = sqlite3.Row
        if mode != "r":
            self._init_schema()

    @staticmethod
    def _open_connection(path: Path, mode: str) -> sqlite3.Connection:
        if mode not in {"r", "a", "w"}:
            raise ValueError("mode must be one of {'r', 'a', 'w'}.")
        if mode == "r":
            if not path.exists():
                raise FileNotFoundError(f"SQLite db not found: {path}")
            uri = f"file:{path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
        else:
            if mode == "w" and path.exists():
                raise FileExistsError(f"Refusing to overwrite existing db: {path}")
            conn = sqlite3.connect(str(path))
        conn.execute("PRAGMA foreign_keys = ON")
        if mode != "r":
            conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS structures (
                    id INTEGER PRIMARY KEY,
                    n_atoms INTEGER NOT NULL,
                    atomic_numbers BLOB NOT NULL,
                    positions BLOB NOT NULL,
                    pbc INTEGER NOT NULL,
                    cell BLOB,
                    energy REAL,
                    forces BLOB,
                    virial BLOB,
                    stress BLOB,
                    dataset TEXT,
                    source_path TEXT,
                    converged INTEGER,
                    label_status TEXT,
                    meta TEXT
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_structures_dataset ON structures(dataset)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_structures_source_path ON structures(source_path)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_structures_converged ON structures(converged)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_structures_label_status ON structures(label_status)"
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS db_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()

    def __enter__(self) -> "SqliteTrajectory":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS count FROM structures").fetchone()
        return int(row["count"])

    def _normalize_meta(
        self,
        meta: Optional[Dict[str, Any]],
        dataset: Optional[str],
        source_path: Optional[str],
        converged: Optional[bool],
        label_status: Optional[str],
    ) -> tuple[Optional[str], Optional[str], Optional[str], Optional[bool], Optional[str]]:
        meta_dict: Dict[str, Any] = {}
        if meta:
            meta_dict.update(meta)
        if dataset is None and "dataset" in meta_dict:
            dataset = str(meta_dict.pop("dataset"))
        if source_path is None and "source_path" in meta_dict:
            source_path = str(meta_dict.pop("source_path"))
        if converged is None and "converged" in meta_dict:
            converged = bool(meta_dict.pop("converged"))
        if label_status is None and "label_status" in meta_dict:
            label_status = str(meta_dict.pop("label_status"))
        if meta_dict:
            meta_json = json.dumps(meta_dict, default=_json_default)
        else:
            meta_json = None
        return meta_json, dataset, source_path, converged, label_status

    def add(
        self,
        atoms: Atoms,
        dataset: Optional[str] = None,
        source_path: Optional[str] = None,
        converged: Optional[bool] = None,
        label_status: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        properties: Optional[Dict[str, Any]] = None,
        commit: bool = True,
    ) -> int:
        props = {} if properties is None else dict(properties)
        energy = props.get("energy")
        if energy is None:
            energy = atoms.info.get("energy")
        if energy is None and getattr(atoms, "calc", None) is not None:
            energy = atoms.calc.results.get("energy") if hasattr(atoms.calc, "results") else None

        forces = props.get("forces")
        if forces is None:
            forces = atoms.arrays.get("forces") if hasattr(atoms, "arrays") else None
        if forces is None:
            forces = atoms.info.get("forces")
        if forces is None and getattr(atoms, "calc", None) is not None:
            forces = atoms.calc.results.get("forces") if hasattr(atoms.calc, "results") else None

        stress = props.get("stress")
        if stress is None:
            stress = atoms.info.get("stress")
        if stress is None and getattr(atoms, "calc", None) is not None:
            stress = atoms.calc.results.get("stress") if hasattr(atoms.calc, "results") else None
        stress = _to_voigt_6(stress)

        virial = props.get("virial")
        if virial is None:
            virial = atoms.info.get("virial")
        if virial is None and getattr(atoms, "calc", None) is not None:
            virial = atoms.calc.results.get("virial") if hasattr(atoms.calc, "results") else None
        virial = _to_voigt_6(virial)

        atomic_numbers = atoms.get_atomic_numbers()
        positions = atoms.get_positions()
        n_atoms = int(len(atomic_numbers))
        pbc = _pbc_to_int(atoms.get_pbc())
        cell = atoms.cell.array if atoms.cell is not None else None

        meta_json, dataset, source_path, converged, label_status = self._normalize_meta(
            meta,
            dataset,
            source_path,
            converged,
            label_status,
        )

        values = (
            n_atoms,
            _blob(atomic_numbers, np.int32),
            _blob(positions, np.float32),
            pbc,
            _blob(cell, np.float32) if cell is not None else None,
            float(energy) if energy is not None else None,
            _blob(forces, np.float32) if forces is not None else None,
            _blob(virial, np.float32) if virial is not None else None,
            _blob(stress, np.float32) if stress is not None else None,
            dataset,
            source_path,
            int(converged) if converged is not None else None,
            label_status,
            meta_json,
        )

        cursor = self._conn.cursor()
        cursor.execute(
            f"INSERT INTO structures ({', '.join(_STRUCTURE_COLUMNS)}) VALUES ({', '.join('?' * len(_STRUCTURE_COLUMNS))})",
            values,
        )
        if commit:
            self._conn.commit()
        return int(cursor.lastrowid)

    def extend(
        self,
        atoms_list: Iterable[Atoms],
        dataset: Optional[str] = None,
        source_path: Optional[str] = None,
        converged: Optional[bool] = None,
        label_status: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        ids: List[int] = []
        with self._conn:
            for atoms in atoms_list:
                ids.append(
                    self.add(
                        atoms,
                        dataset=dataset,
                        source_path=source_path,
                        converged=converged,
                        label_status=label_status,
                        meta=meta,
                        properties=properties,
                        commit=False,
                    )
                )
        return ids

    def _row_to_atoms(self, row: sqlite3.Row) -> Atoms:
        n_atoms = int(row["n_atoms"])
        atomic_numbers = _deblob(row["atomic_numbers"], np.int32, (n_atoms,))
        positions = _deblob(row["positions"], np.float32, (n_atoms, 3))
        pbc = _int_to_pbc(int(row["pbc"]))
        cell = _deblob(row["cell"], np.float32, (3, 3)) if row["cell"] is not None else None
        atoms = Atoms(numbers=atomic_numbers, positions=positions, pbc=pbc)
        if cell is not None:
            atoms.set_cell(cell)

        results: Dict[str, Any] = {}
        if row["energy"] is not None:
            results["energy"] = float(row["energy"])
        forces = _deblob(row["forces"], np.float32, (n_atoms, 3)) if row["forces"] is not None else None
        if forces is not None:
            results["forces"] = forces
            atoms.arrays["forces"] = forces
        stress = _deblob(row["stress"], np.float32, (6,)) if row["stress"] is not None else None
        if stress is not None:
            results["stress"] = stress
        virial = _deblob(row["virial"], np.float32, (6,)) if row["virial"] is not None else None
        if virial is not None:
            results["virial"] = virial
        if results:
            atoms.calc = SinglePointCalculator(atoms, **results)

        meta: Dict[str, Any] = {}
        if row["meta"]:
            try:
                meta = json.loads(row["meta"])
            except json.JSONDecodeError:
                meta = {}
        atoms.info.update(meta)
        atoms.info.setdefault("structure_id", int(row["id"]))
        if row["dataset"] is not None:
            atoms.info.setdefault("dataset", row["dataset"])
        if row["source_path"] is not None:
            atoms.info.setdefault("source_path", row["source_path"])
        if row["converged"] is not None:
            atoms.info.setdefault("converged", bool(row["converged"]))
        if row["label_status"] is not None:
            atoms.info.setdefault("label_status", row["label_status"])
        return atoms

    def get_row(self, structure_id: int) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM structures WHERE id = ?",
            (int(structure_id),),
        ).fetchone()

    def get_atoms(self, structure_id: int) -> Optional[Atoms]:
        row = self.get_row(structure_id)
        if row is None:
            return None
        return self._row_to_atoms(row)

    def __getitem__(self, idx: Union[int, slice, Sequence[int]]) -> Union[Atoms, List[Atoms]]:
        if isinstance(idx, slice):
            indices = list(range(*idx.indices(len(self))))
            return [self[i] for i in indices]
        if isinstance(idx, (list, tuple, np.ndarray)):
            return [self[int(i)] for i in idx]
        if not isinstance(idx, int):
            raise TypeError("Index must be int, slice, or sequence of ints.")
        if idx < 0:
            idx = len(self) + idx
        if idx < 0:
            raise IndexError("Index out of range.")
        row = self._conn.execute(
            "SELECT * FROM structures ORDER BY id LIMIT 1 OFFSET ?",
            (idx,),
        ).fetchone()
        if row is None:
            raise IndexError("Index out of range.")
        return self._row_to_atoms(row)

    def __iter__(self) -> Iterator[Atoms]:
        cursor = self._conn.execute("SELECT * FROM structures ORDER BY id")
        for row in cursor:
            yield self._row_to_atoms(row)

    def ids(self) -> List[int]:
        rows = self._conn.execute("SELECT id FROM structures ORDER BY id").fetchall()
        return [int(row["id"]) for row in rows]

    def select_ids(
        self,
        dataset: Optional[str] = None,
        source_path: Optional[str] = None,
        converged: Optional[bool] = None,
        label_status: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[int]:
        where: List[str] = []
        params: List[Any] = []
        if dataset is not None:
            where.append("dataset = ?")
            params.append(dataset)
        if source_path is not None:
            where.append("source_path = ?")
            params.append(source_path)
        if converged is not None:
            where.append("converged = ?")
            params.append(int(bool(converged)))
        if label_status is not None:
            where.append("label_status = ?")
            params.append(label_status)
        sql = "SELECT id, meta FROM structures"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        if offset is not None:
            sql += " OFFSET ?"
            params.append(int(offset))
        rows = self._conn.execute(sql, params).fetchall()
        if meta:
            filtered: List[int] = []
            for row in rows:
                meta_dict = {}
                if row["meta"]:
                    try:
                        meta_dict = json.loads(row["meta"])
                    except json.JSONDecodeError:
                        meta_dict = {}
                if all(meta_dict.get(k) == v for k, v in meta.items()):
                    filtered.append(int(row["id"]))
            return filtered
        return [int(row["id"]) for row in rows]

    def select(
        self,
        dataset: Optional[str] = None,
        source_path: Optional[str] = None,
        converged: Optional[bool] = None,
        label_status: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Atoms]:
        ids = self.select_ids(
            dataset=dataset,
            source_path=source_path,
            converged=converged,
            label_status=label_status,
            meta=meta,
            limit=limit,
            offset=offset,
        )
        return [self.get_atoms(structure_id) for structure_id in ids if structure_id is not None]

    def set_db_meta(self, key: str, value: Any) -> None:
        payload = json.dumps(value, default=_json_default)
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO db_meta(key, value) VALUES (?, ?)",
                (str(key), payload),
            )

    def get_db_meta(self, key: str, default: Any = None) -> Any:
        row = self._conn.execute(
            "SELECT value FROM db_meta WHERE key = ?",
            (str(key),),
        ).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return row["value"]

    def merge_from(self, sources: Sequence[Union[str, Path]]) -> None:
        if not sources:
            return
        for idx, source in enumerate(sources):
            source_path = Path(source)
            if source_path.resolve() == self.path.resolve():
                continue
            alias = f"src_{idx}"
            with self._conn:
                self._conn.execute(f"ATTACH DATABASE ? AS {alias}", (str(source_path),))
                self._conn.execute(
                    f"""
                    INSERT INTO structures ({', '.join(_STRUCTURE_COLUMNS)})
                    SELECT {', '.join(_STRUCTURE_COLUMNS)}
                    FROM {alias}.structures
                    ORDER BY id
                    """
                )
                self._conn.execute(f"DETACH DATABASE {alias}")

    def __add__(self, other: Union["SqliteTrajectory", str, Path]) -> "CombinedSqliteTrajectory":
        return CombinedSqliteTrajectory([self, other])


class CombinedSqliteTrajectory:
    def __init__(self, sources: Sequence[Union[SqliteTrajectory, str, Path]]) -> None:
        self._owned: List[SqliteTrajectory] = []
        self._sources: List[SqliteTrajectory] = []
        for source in sources:
            if isinstance(source, SqliteTrajectory):
                self._sources.append(source)
            else:
                traj = SqliteTrajectory(source, mode="r")
                self._sources.append(traj)
                self._owned.append(traj)

    def close(self) -> None:
        for traj in self._owned:
            traj.close()
        self._owned = []

    def __enter__(self) -> "CombinedSqliteTrajectory":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __len__(self) -> int:
        return sum(len(src) for src in self._sources)

    def __getitem__(self, idx: Union[int, slice, Sequence[int]]) -> Union[Atoms, List[Atoms]]:
        if isinstance(idx, slice):
            indices = list(range(*idx.indices(len(self))))
            return [self[i] for i in indices]
        if isinstance(idx, (list, tuple, np.ndarray)):
            return [self[int(i)] for i in idx]
        if not isinstance(idx, int):
            raise TypeError("Index must be int, slice, or sequence of ints.")
        if idx < 0:
            idx = len(self) + idx
        if idx < 0:
            raise IndexError("Index out of range.")
        for src in self._sources:
            if idx < len(src):
                return src[idx]
            idx -= len(src)
        raise IndexError("Index out of range.")

    def __iter__(self) -> Iterator[Atoms]:
        for src in self._sources:
            yield from src

    def __add__(self, other: Union[SqliteTrajectory, str, Path]) -> "CombinedSqliteTrajectory":
        return CombinedSqliteTrajectory(self._sources + [other])
