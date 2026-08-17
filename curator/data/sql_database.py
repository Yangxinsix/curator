import apsw
import numpy as np
import io
import os
import multiprocessing
import torch
import json
from typing import Dict, Any, List, Optional
from curator.data import properties, NeighborListTransform, TorchNeighborList
from ase.data import chemical_symbols, atomic_numbers
from ase import units

from .variational_data import (
    VariationalCondition,
    VariationalData,
    VariationalResponse,
)

'''
This is a class to store large amounts of ab initio reference data
for training a neural network in a SQLite database

Data structure:
input data:
 atomic_numbers (N)    (int)   nuclear charges
 pbc ()   (int)   has PBC or not
 positions (N, 3) (float) Cartesian coordinates in A
 cell (3, 3) (float) Cell length in A
 energy ()     (float) energy in eV
 forces (N, 3) (float) forces in eV/A
 total_charge ()     (float) total charge
 atomic_charge (N)  (float) atomic charge
 total_magmom ()     (float) total magnetic moment (number of unpaired electrons, i.e. for singlet S=0, doublet S=1, etc.)
dipole (3)    (float) dipole moment in eV*A (with respect to origin)
'''

EXTRA_COLUMNS_INFO_KEY = "schema.extra_columns"
ACVR_SCHEMA_INFO_KEY = "acvr.schema_version"
ACVR_SCHEMA_VERSION = "1.0"

BASE_COLUMN_SPECS = {
    properties.atomic_numbers: {"sql_type": "BLOB", "storage": "blob", "dtype": "int32", "shape": ["n_atoms"]},
    properties.pbc: {"sql_type": "INTEGER", "storage": "scalar", "dtype": "bool"},
    properties.positions: {"sql_type": "BLOB", "storage": "blob", "dtype": "float32", "shape": ["n_atoms", 3]},
}

OPTIONAL_COLUMN_SPECS = {
    properties.cell: {"sql_type": "BLOB", "storage": "blob", "dtype": "float32", "shape": [3, 3]},
    properties.energy: {"sql_type": "FLOAT", "storage": "scalar", "dtype": "float32"},
    properties.forces: {"sql_type": "BLOB", "storage": "blob", "dtype": "float32", "shape": ["n_atoms", 3]},
    properties.energy_hessian: {"sql_type": "BLOB", "storage": "blob", "dtype": "float32", "shape": ["n_atoms", 3, "n_atoms", 3]},
    properties.energy_hessian_projected: {"sql_type": "BLOB", "storage": "blob", "dtype": "float32", "shape": ["num_probes", "n_atoms", 3]},
    properties.energy_hessian_probe_vectors: {"sql_type": "BLOB", "storage": "blob", "dtype": "float32", "shape": ["num_probes", "n_atoms", 3]},
    properties.node_final_feature: {"sql_type": "BLOB", "storage": "blob", "dtype": "float32", "shape": ["n_atoms", "feature_dim"]},
    properties.virial: {"sql_type": "BLOB", "storage": "blob", "dtype": "float32", "shape": [1, 6]},
    properties.stress: {"sql_type": "BLOB", "storage": "blob", "dtype": "float32", "shape": [1, 6]},
    properties.total_charge: {"sql_type": "FLOAT", "storage": "scalar", "dtype": "float32"},
    properties.atomic_charge: {"sql_type": "BLOB", "storage": "blob", "dtype": "float32", "shape": ["n_atoms"]},
    properties.total_magmom: {"sql_type": "FLOAT", "storage": "scalar", "dtype": "float32"},
    properties.dipole: {"sql_type": "BLOB", "storage": "blob", "dtype": "float32", "shape": [1, 3]},
}

STANDARD_COLUMN_SPECS = {**BASE_COLUMN_SPECS, **OPTIONAL_COLUMN_SPECS}
STANDARD_COLUMN_ORDER = list(STANDARD_COLUMN_SPECS.keys())

class QMDatabase:
    def __init__(
        self,
        filename,
        flags=apsw.SQLITE_OPEN_READONLY,
        extra_columns: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.db = filename
        self.connections = {}  # allow multiple connections (needed for multi-threading)
        self._all_data_columns: List[str] = []
        self._read_columns: List[str] = []
        self._open(flags=flags)  # creates the database if it doesn't exist yet
        stored_extra_columns = self.get_info(EXTRA_COLUMNS_INFO_KEY, default={}) or {}
        self.extra_columns: Dict[str, Dict[str, Any]] = {}
        normalized_extra_columns = {}
        for key, spec in (extra_columns or {}).items():
            normalized_extra_columns[str(key)] = {
                "sql_type": str(spec["sql_type"]).upper(),
                "storage": str(spec.get("storage", "blob")).lower(),
                "dtype": np.dtype(spec.get("dtype", "float32")).name,
                "shape": None if spec.get("shape") is None else list(spec["shape"]),
            }
        if stored_extra_columns and normalized_extra_columns and stored_extra_columns != normalized_extra_columns:
            raise ValueError("extra_columns does not match the schema stored in the SQLite database.")
        self.extra_columns = normalized_extra_columns or stored_extra_columns
        if self.extra_columns and (flags & apsw.SQLITE_OPEN_READWRITE):
            self.set_info(EXTRA_COLUMNS_INFO_KEY, self.extra_columns, flags=apsw.SQLITE_OPEN_READWRITE)
        self._column_specs = {**STANDARD_COLUMN_SPECS, **self.extra_columns}
        self._refresh_schema()

    def __len__(self):
        cursor = self._get_connection(flags=apsw.SQLITE_OPEN_READONLY).cursor()
        return cursor.execute('''SELECT * FROM metadata WHERE id=1''').fetchone()[-1]

    def __getitem__(self, idx):
        cursor = self._get_connection(flags=apsw.SQLITE_OPEN_READONLY).cursor()
        columns = ", ".join(self._read_columns)
        if type(idx) == list:  # for batched data retrieval
            data = cursor.execute(f'''SELECT {columns} FROM data WHERE id IN (''' + str(idx)[1:-1] + ')').fetchall()
            return [self._unpack_data_tuple(i, self._read_columns) for i in data]
        else:
            data = cursor.execute(f'''SELECT {columns} FROM data WHERE id=''' + str(idx)).fetchone()
            return self._unpack_data_tuple(data, self._read_columns)

    def _refresh_schema(self):
        cursor = self._get_connection(flags=apsw.SQLITE_OPEN_READONLY).cursor()
        self._all_data_columns = [row[1] for row in cursor.execute("PRAGMA table_info(data)").fetchall()]
        self._read_columns = ["id"]
        self._read_columns += [column for column in STANDARD_COLUMN_ORDER if column in self._all_data_columns]
        self._read_columns += [column for column in self.extra_columns if column in self._all_data_columns]

    def _unpack_data_tuple(self, data, columns=None):
        if data is None:
            raise KeyError("Requested index does not exist in SQLite database.")
        if columns is None:
            columns = self._all_data_columns if len(data) == len(self._all_data_columns) else self._read_columns
        row = dict(zip(columns, data))
        atomic_numbers_blob = row.get(properties.atomic_numbers)
        if atomic_numbers_blob is None:
            raise KeyError("SQLite row is missing atomic_numbers.")
        n_atoms = len(atomic_numbers_blob) // 4
        atoms_data = {properties.n_atoms: np.array([n_atoms], dtype=np.int64)}

        for key, value in row.items():
            if key == "id" or value is None:
                continue
            spec = self._column_specs.get(key)
            if spec is None:
                continue
            if spec["storage"] == "scalar":
                scalar = bool(value) if spec["dtype"] == "bool" else value
                atoms_data[key] = np.array([scalar], dtype=np.dtype(spec["dtype"]))
                continue
            shape = self._resolve_blob_shape(spec.get("shape"), value, spec["dtype"], n_atoms)
            atoms_data[key] = self._deblob(value, dtype=np.dtype(spec["dtype"]), shape=shape)

        if properties.atomic_numbers in atoms_data:
            atoms_data[properties.atomic_numbers] = atoms_data[properties.atomic_numbers].astype(np.int64)
        return atoms_data

    def _ensure_data_columns(self, keys: List[str], cursor) -> None:
        existing = set(self._all_data_columns)
        for key in keys:
            if key in existing or key == "id":
                continue
            sql_type = self._column_specs.get(key, {}).get("sql_type")
            if sql_type is None:
                raise KeyError(f"Unsupported column '{key}'. Register it in extra_columns before writing.")
            cursor.execute(f"ALTER TABLE data ADD COLUMN {key} {sql_type}")
            existing.add(key)
        self._all_data_columns = [column for column in ["id", *STANDARD_COLUMN_ORDER, *self.extra_columns.keys()] if column in existing]
        self._read_columns = list(self._all_data_columns)

    def add_data(self, data_dict, flags=apsw.SQLITE_OPEN_READWRITE, transaction=True):
        """
        Add data from a dictionary to the SQLite database.
        :param data_dict: Dictionary containing the data to insert
        :param flags: SQLite access flags
        :param transaction: Boolean flag for handling transactions
        """
        # Check for NaN values
        vals = []
        for key in (properties.atomic_numbers, properties.positions):
            value = data_dict.get(key)
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            elif value is not None:
                value = np.asarray(value)
            vals.append(value)
        if self._any_is_nan(*vals):
            print("encountered NaN, data is not added")
            return

        encoded = {}
        for key, value in data_dict.items():
            spec = self._column_specs.get(key)
            if spec is None:
                raise KeyError(f"Unsupported column '{key}'. Register it in extra_columns before writing.")
            if value is None:
                encoded[key] = None
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            if spec["storage"] == "scalar":
                scalar = np.asarray(value).reshape(-1)[0]
                if spec["dtype"] == "bool":
                    encoded[key] = int(bool(scalar))
                elif np.issubdtype(np.dtype(spec["dtype"]), np.integer):
                    encoded[key] = int(scalar)
                else:
                    encoded[key] = float(scalar)
            else:
                encoded[key] = self._blob(np.asarray(value, dtype=np.dtype(spec["dtype"])))

        cursor = self._get_connection(flags=flags).cursor()

        if transaction:
            cursor.execute('''BEGIN EXCLUSIVE''')  # Begin exclusive transaction to lock the DB

        try:
            self._ensure_data_columns(list(encoded.keys()), cursor)
            length = cursor.execute('''SELECT * FROM metadata WHERE id=1''').fetchone()[-1]
            keys = ['id']   # id
            vals = [None if length > 0 else 0]
            keys += [k for k in encoded.keys()]
            vals += [v for v in encoded.values()]
            columns = ', '.join(keys)
            placeholders = ', '.join('?' * len(vals))
            sql_cmd = f'INSERT INTO data ({columns}) VALUES ({placeholders})'
            cursor.execute(sql_cmd, vals)

            # insert metadata
            cursor.execute('''INSERT OR REPLACE INTO metadata VALUES (?,?)''', (1, length + 1))
            Nmax = cursor.execute('''SELECT * FROM metadata WHERE id=0''').fetchone()[-1]
            atomic_numbers = data_dict[properties.atomic_numbers]
            if isinstance(atomic_numbers, torch.Tensor):
                atomic_numbers = atomic_numbers.detach().cpu().numpy()
            n_atoms = len(np.asarray(atomic_numbers).reshape(-1))
            if n_atoms > Nmax:  # Update Nmax if necessary
                cursor.execute('''INSERT OR REPLACE INTO metadata VALUES (?,?)''', (0, n_atoms))

            if transaction:
                cursor.execute('''COMMIT''')  # End transaction
        except Exception as exc:
            if transaction:
                cursor.execute('''ROLLBACK''')  # Rollback transaction on error
            raise exc

    @staticmethod
    def _any_is_nan(*vals):
        nan = False
        for val in vals:
            if val is None:
                return True
            elif np.any(np.isnan(val)):
                return True
        return nan

    def _blob(self, array):
        """Convert numpy array to blob/buffer object."""
        if array is None:
            return None
        if array.dtype == np.float64:
            array = array.astype(np.float32)
        if array.dtype == np.int64:
            array = array.astype(np.int32)
        if not np.little_endian:
            array = array.byteswap()
        return memoryview(np.ascontiguousarray(array))

    @staticmethod
    def _resolve_blob_shape(shape_spec, buf, dtype, n_atoms):
        if shape_spec is None:
            return None
        dtype = np.dtype(dtype)
        flat_size = len(buf) // dtype.itemsize
        shape = []
        unknown_idx = None
        known = 1
        for idx, dim in enumerate(shape_spec):
            if dim == "n_atoms":
                value = int(n_atoms)
            elif isinstance(dim, str):
                if unknown_idx is not None:
                    raise ValueError(f"Only one inferred dimension is supported in SQLite shape specs: {shape_spec}")
                unknown_idx = idx
                value = -1
            else:
                value = int(dim)
            shape.append(value)
            if value != -1:
                known *= value
        if unknown_idx is not None:
            if known == 0 or flat_size % known != 0:
                raise ValueError(
                    f"Cannot infer SQLite blob shape {shape_spec} from flat size {flat_size} and n_atoms={n_atoms}."
                )
            shape[unknown_idx] = flat_size // known
        return tuple(shape)

    def _deblob(self, buf, dtype=np.float32, shape=None):
        """Convert blob/buffer object to numpy array."""
        if buf is None:
            return np.zeros(shape)
        array = np.frombuffer(buf, dtype)
        if not np.little_endian:
            array = array.byteswap()
        array.shape = shape
        return np.copy(array)

    def _open(self, flags=apsw.SQLITE_OPEN_READONLY):
        newdb = not os.path.isfile(self.db)
        cursor = self._get_connection(flags=flags).cursor()
        if newdb:
            columns_sql = ",\n                 ".join(
                [f"{column} {STANDARD_COLUMN_SPECS[column]['sql_type']}" for column in STANDARD_COLUMN_ORDER]
            )
            # Create table to store data with full names
            cursor.execute(
                f'''CREATE TABLE IF NOT EXISTS data
                (id INTEGER NOT NULL PRIMARY KEY,
                 {columns_sql})'''
            )

            # Create table to store metadata (information about Nmax and the length, i.e., number of entries)
            cursor.execute('''CREATE TABLE IF NOT EXISTS metadata
                (id INTEGER PRIMARY KEY, N INTEGER)''')
            self._ensure_info_table(cursor)

            # Initialize metadata values
            cursor.execute('''INSERT OR IGNORE INTO metadata (id, N) VALUES (?,?)''', (0, 0))  # Nmax
            cursor.execute('''INSERT OR IGNORE INTO metadata (id, N) VALUES (?,?)''', (1, 0))  # num_data

    @staticmethod
    def _ensure_acvr_tables(cursor):
        cursor.execute(
            '''CREATE TABLE IF NOT EXISTS acvr_geometry
               (data_id INTEGER PRIMARY KEY,
                schema_version TEXT NOT NULL,
                geometry_uid TEXT NOT NULL UNIQUE,
                geometry_hash TEXT NOT NULL,
                variable_spec_json TEXT NOT NULL,
                provenance_json TEXT NOT NULL)'''
        )
        cursor.execute(
            '''CREATE TABLE IF NOT EXISTS acvr_group
               (data_id INTEGER NOT NULL,
                group_index INTEGER NOT NULL,
                name TEXT NOT NULL,
                role TEXT NOT NULL,
                member_indices BLOB NOT NULL,
                PRIMARY KEY (data_id, group_index))'''
        )
        cursor.execute(
            '''CREATE TABLE IF NOT EXISTS acvr_condition
               (data_id INTEGER NOT NULL,
                condition_index INTEGER NOT NULL,
                condition_uid TEXT NOT NULL,
                electron_count REAL,
                electron_count_convention TEXT,
                net_charge REAL,
                boundary_json TEXT NOT NULL,
                boundary_hash TEXT NOT NULL,
                gauge_json TEXT NOT NULL,
                weight REAL NOT NULL,
                provenance_json TEXT NOT NULL,
                PRIMARY KEY (data_id, condition_index),
                UNIQUE (data_id, condition_uid))'''
        )
        cursor.execute(
            '''CREATE TABLE IF NOT EXISTS acvr_tensor
               (data_id INTEGER NOT NULL,
                condition_index INTEGER NOT NULL,
                key TEXT NOT NULL,
                present INTEGER NOT NULL,
                dtype TEXT NOT NULL,
                shape_json TEXT NOT NULL,
                unit TEXT NOT NULL,
                value BLOB NOT NULL,
                metadata_json TEXT NOT NULL,
                PRIMARY KEY (data_id, condition_index, key))'''
        )
        cursor.execute(
            '''CREATE TABLE IF NOT EXISTS acvr_response
               (data_id INTEGER NOT NULL,
                condition_index INTEGER NOT NULL,
                response_index INTEGER NOT NULL,
                input_ref TEXT NOT NULL,
                output_ref TEXT NOT NULL,
                dtype TEXT NOT NULL,
                shape_json TEXT NOT NULL,
                unit TEXT NOT NULL,
                method TEXT NOT NULL,
                value BLOB NOT NULL,
                gauge_json TEXT NOT NULL,
                provenance_json TEXT NOT NULL,
                PRIMARY KEY (data_id, condition_index, response_index))'''
        )

    def ensure_acvr_schema(self, flags=apsw.SQLITE_OPEN_READWRITE):
        cursor = self._get_connection(flags=flags).cursor()
        cursor.execute("BEGIN IMMEDIATE")
        try:
            self._ensure_acvr_tables(cursor)
            self._ensure_info_table(cursor)
            row = cursor.execute(
                "SELECT value FROM info WHERE key=?", (ACVR_SCHEMA_INFO_KEY,)
            ).fetchone()
            if row is not None and json.loads(row[0]) != ACVR_SCHEMA_VERSION:
                raise ValueError("unsupported ACVR SQLite schema version")
            cursor.execute(
                "INSERT OR REPLACE INTO info (key, value) VALUES (?, ?)",
                (ACVR_SCHEMA_INFO_KEY, json.dumps(ACVR_SCHEMA_VERSION)),
            )
            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise

    @staticmethod
    def _encode_acvr_tensor(value):
        if not isinstance(value, torch.Tensor):
            raise TypeError("ACVR tensor values must be torch.Tensor instances")
        array = value.detach().cpu().contiguous().numpy()
        return array.dtype.str, json.dumps(list(array.shape)), memoryview(array)

    @staticmethod
    def _decode_acvr_tensor(dtype, shape_json, value):
        dtype = np.dtype(dtype)
        shape = tuple(json.loads(shape_json))
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if len(value) != expected:
            raise ValueError("ACVR tensor byte length does not match dtype and shape")
        return torch.from_numpy(np.frombuffer(value, dtype=dtype).copy().reshape(shape))

    def add_variational_data(
        self,
        data_id: int,
        payload: VariationalData,
        replace: bool = False,
        flags=apsw.SQLITE_OPEN_READWRITE,
    ) -> None:
        cursor = self._get_connection(flags=flags).cursor()
        row = cursor.execute("SELECT atomic_numbers FROM data WHERE id=?", (int(data_id),)).fetchone()
        if row is None:
            raise KeyError("SQLite geometry row %d does not exist" % data_id)
        payload.validate(len(row[0]) // np.dtype("int32").itemsize)

        cursor.execute("BEGIN IMMEDIATE")
        try:
            self._ensure_acvr_tables(cursor)
            self._ensure_info_table(cursor)
            cursor.execute(
                "INSERT OR REPLACE INTO info (key, value) VALUES (?, ?)",
                (ACVR_SCHEMA_INFO_KEY, json.dumps(ACVR_SCHEMA_VERSION)),
            )
            existing = cursor.execute(
                "SELECT 1 FROM acvr_geometry WHERE data_id=?", (int(data_id),)
            ).fetchone()
            if existing and not replace:
                raise ValueError("variational data already exist for row %d" % data_id)
            if existing:
                for table in ("acvr_response", "acvr_tensor", "acvr_condition", "acvr_group", "acvr_geometry"):
                    cursor.execute("DELETE FROM %s WHERE data_id=?" % table, (int(data_id),))

            cursor.execute(
                "INSERT INTO acvr_geometry VALUES (?, ?, ?, ?, ?, ?)",
                (
                    int(data_id),
                    payload.schema_version,
                    payload.geometry_uid,
                    payload.geometry_hash,
                    json.dumps(payload.variable_spec, sort_keys=True, allow_nan=False),
                    json.dumps(payload.provenance, sort_keys=True, allow_nan=False),
                ),
            )
            for group_index, (name, role, start, stop) in enumerate(
                zip(
                    payload.group_names,
                    payload.group_roles,
                    payload.group_member_ptr[:-1],
                    payload.group_member_ptr[1:],
                )
            ):
                members = payload.group_atom_index[int(start) : int(stop)].cpu().numpy().astype("<i8")
                cursor.execute(
                    "INSERT INTO acvr_group VALUES (?, ?, ?, ?, ?)",
                    (int(data_id), group_index, name, role, memoryview(members)),
                )
            for condition_index, condition in enumerate(payload.conditions):
                cursor.execute(
                    "INSERT INTO acvr_condition VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        int(data_id),
                        condition_index,
                        condition.condition_uid,
                        None if condition.electron_count is None else float(condition.electron_count),
                        condition.electron_count_convention,
                        None if condition.net_charge is None else float(condition.net_charge),
                        json.dumps(condition.boundary_spec, sort_keys=True, allow_nan=False),
                        condition.boundary_hash,
                        json.dumps(condition.gauge, sort_keys=True, allow_nan=False),
                        float(condition.weight),
                        json.dumps(condition.provenance, sort_keys=True, allow_nan=False),
                    ),
                )
                for key, value in condition.labels.items():
                    dtype, shape, encoded = self._encode_acvr_tensor(value)
                    cursor.execute(
                        "INSERT INTO acvr_tensor VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            int(data_id),
                            condition_index,
                            key,
                            1,
                            dtype,
                            shape,
                            condition.label_units[key],
                            encoded,
                            json.dumps(condition.label_metadata.get(key, {}), sort_keys=True, allow_nan=False),
                        ),
                    )
                for response_index, response in enumerate(condition.responses):
                    dtype, shape, encoded = self._encode_acvr_tensor(response.value)
                    cursor.execute(
                        "INSERT INTO acvr_response VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            int(data_id),
                            condition_index,
                            response_index,
                            response.input_ref,
                            response.output_ref,
                            dtype,
                            shape,
                            response.unit,
                            response.method,
                            encoded,
                            json.dumps(response.gauge, sort_keys=True, allow_nan=False),
                            json.dumps(response.provenance, sort_keys=True, allow_nan=False),
                        ),
                    )
            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise

    def get_variational_data(
        self, data_id: int, flags=apsw.SQLITE_OPEN_READONLY
    ) -> Optional[VariationalData]:
        cursor = self._get_connection(flags=flags).cursor()
        if cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='acvr_geometry'"
        ).fetchone() is None:
            return None
        geometry = cursor.execute(
            "SELECT schema_version, geometry_uid, geometry_hash, variable_spec_json, provenance_json "
            "FROM acvr_geometry WHERE data_id=?",
            (int(data_id),),
        ).fetchone()
        if geometry is None:
            return None
        if geometry[0] != ACVR_SCHEMA_VERSION:
            raise ValueError("unsupported ACVR SQLite schema version %s" % geometry[0])
        groups = cursor.execute(
            "SELECT name, role, member_indices FROM acvr_group WHERE data_id=? ORDER BY group_index",
            (int(data_id),),
        ).fetchall()
        group_ptr = [0]
        group_indices = []
        for _, _, value in groups:
            members = np.frombuffer(value, dtype="<i8").copy()
            group_indices.extend(members.tolist())
            group_ptr.append(len(group_indices))

        conditions = []
        rows = cursor.execute(
            "SELECT condition_index, condition_uid, electron_count, electron_count_convention, "
            "net_charge, boundary_json, boundary_hash, gauge_json, weight, provenance_json "
            "FROM acvr_condition WHERE data_id=? ORDER BY condition_index",
            (int(data_id),),
        ).fetchall()
        for row in rows:
            condition_index = row[0]
            labels = {}
            units = {}
            metadata = {}
            for key, present, dtype, shape, unit, value, meta in cursor.execute(
                "SELECT key, present, dtype, shape_json, unit, value, metadata_json "
                "FROM acvr_tensor WHERE data_id=? AND condition_index=? ORDER BY key",
                (int(data_id), condition_index),
            ).fetchall():
                if present:
                    labels[key] = self._decode_acvr_tensor(dtype, shape, value)
                    units[key] = unit
                    metadata[key] = json.loads(meta)
            responses = []
            for response in cursor.execute(
                "SELECT input_ref, output_ref, dtype, shape_json, unit, method, value, "
                "gauge_json, provenance_json FROM acvr_response "
                "WHERE data_id=? AND condition_index=? ORDER BY response_index",
                (int(data_id), condition_index),
            ).fetchall():
                responses.append(
                    VariationalResponse(
                        input_ref=response[0],
                        output_ref=response[1],
                        value=self._decode_acvr_tensor(response[2], response[3], response[6]),
                        unit=response[4],
                        method=response[5],
                        gauge=json.loads(response[7]),
                        provenance=json.loads(response[8]),
                    )
                )
            conditions.append(
                VariationalCondition(
                    condition_uid=row[1],
                    electron_count=None if row[2] is None else torch.tensor(row[2], dtype=torch.float64),
                    electron_count_convention=row[3],
                    net_charge=None if row[4] is None else torch.tensor(row[4], dtype=torch.float64),
                    boundary_spec=json.loads(row[5]),
                    boundary_hash=row[6],
                    gauge=json.loads(row[7]),
                    labels=labels,
                    label_units=units,
                    label_metadata=metadata,
                    responses=tuple(responses),
                    weight=row[8],
                    provenance=json.loads(row[9]),
                )
            )
        payload = VariationalData(
            schema_version=geometry[0],
            geometry_uid=geometry[1],
            geometry_hash=geometry[2],
            variable_spec=json.loads(geometry[3]),
            group_member_ptr=torch.tensor(group_ptr, dtype=torch.long),
            group_atom_index=torch.tensor(group_indices, dtype=torch.long),
            group_names=tuple(row[0] for row in groups),
            group_roles=tuple(row[1] for row in groups),
            conditions=tuple(conditions),
            provenance=json.loads(geometry[4]),
        )
        atomic_numbers = cursor.execute(
            "SELECT atomic_numbers FROM data WHERE id=?", (int(data_id),)
        ).fetchone()
        if atomic_numbers is None:
            raise KeyError("SQLite geometry row %d does not exist" % data_id)
        payload.validate(len(atomic_numbers[0]) // np.dtype("int32").itemsize)
        return payload

    @staticmethod
    def _ensure_info_table(cursor):
        cursor.execute(
            '''CREATE TABLE IF NOT EXISTS info
               (key TEXT PRIMARY KEY, value TEXT NOT NULL)'''
        )

    def set_info(self, key: str, value: Any, flags=apsw.SQLITE_OPEN_READWRITE):
        cursor = self._get_connection(flags=flags).cursor()
        self._ensure_info_table(cursor)
        payload = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        cursor.execute(
            '''INSERT OR REPLACE INTO info (key, value) VALUES (?, ?)''',
            (str(key), payload),
        )

    def get_info(self, key: str, default: Any = None, flags=apsw.SQLITE_OPEN_READONLY):
        cursor = self._get_connection(flags=flags).cursor()
        if not self._has_info_table(cursor):
            return default
        row = cursor.execute(
            '''SELECT value FROM info WHERE key=?''',
            (str(key),),
        ).fetchone()
        if row is None:
            return default
        value = row[0]
        try:
            return json.loads(value)
        except Exception:
            return value

    def get_all_info(self, flags=apsw.SQLITE_OPEN_READONLY) -> Dict[str, Any]:
        cursor = self._get_connection(flags=flags).cursor()
        if not self._has_info_table(cursor):
            return {}
        entries = cursor.execute('''SELECT key, value FROM info''').fetchall()
        info: Dict[str, Any] = {}
        for key, value in entries:
            try:
                info[key] = json.loads(value)
            except Exception:
                info[key] = value
        return info

    def set_cache_metadata(self, metadata: Dict[str, Any], namespace: str = "cache", flags=apsw.SQLITE_OPEN_READWRITE):
        for key, value in metadata.items():
            self.set_info(f"{namespace}.{key}", value, flags=flags)

    def get_cache_metadata(self, namespace: str = "cache", flags=apsw.SQLITE_OPEN_READONLY) -> Dict[str, Any]:
        prefix = f"{namespace}."
        info = self.get_all_info(flags=flags)
        return {
            key[len(prefix):]: value
            for key, value in info.items()
            if key.startswith(prefix)
        }

    @staticmethod
    def _has_info_table(cursor) -> bool:
        row = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='info'"
        ).fetchone()
        return row is not None

    def _get_connection(self, flags=apsw.SQLITE_OPEN_READONLY):
        '''
        This allows multiple processes to access the database at once,
        every process must have its own connection
        '''
        key = (multiprocessing.current_process().name, int(flags))
        if key not in self.connections.keys():
            self.connections[key] = apsw.Connection(self.db, flags=flags)
            self.connections[key].setbusytimeout(300000)  # 5-minute timeout
        return self.connections[key]

    @property
    def Nmax(self):
        cursor = self._get_connection(flags=apsw.SQLITE_OPEN_READONLY).cursor()
        return cursor.execute('''SELECT * FROM metadata WHERE id=0''').fetchone()[-1]
    
    # def write_xyz(self, idx, filename=None):
    #     if filename == None:
    #         filename = str(idx)+".xyz"
    #     data_dict = self[idx]
    #     with open(filename, "w") as file:
    #         file.write(str(data_dict[properties.atomic_numbers].shape[0])+"\n")
    #         file.write("total_charge: {0} total_magmom: {1} energy: {2: 15.6f} dipole: {3: 11.6f} {4: 11.6f} {5: 11.6f}\n".format(int(total_charge[0]), int(total_magmom[0]), energy[0], dipole[0,0], dipole[0,1], dipole[0,2]))
    #         for atomic_number, atomic_ch, pos, force in zip(atomic_numbers, atomic_charge, positions, forces):
    #             file.write('{0} {1: 11.6f} {2: 11.6f} {3: 11.6f} {4: 11.6f} {5: 11.6f} {6: 11.6f} {7: 11.6f}\n'.format(chemical_symbols[atomic_number], atomic_ch, *pos, *force))

class Sqlite3Dataset(QMDatabase, torch.utils.data.Dataset):
    def __init__(
            self,
            filename, 
            cutoff = None, 
            compute_neighbor_list = False,
            return_cell_displacements = False,
            transforms = None,
            task: str = "sqlite3",
            weight: float = 1.0,
            meta: Dict = None,
            return_atoms_data: bool = True,
            default_dtype: torch.dtype = torch.float32,
            **kwargs,
        ):
        flags = kwargs.pop("flags", apsw.SQLITE_OPEN_READONLY)
        extra_columns = kwargs.pop("extra_columns", None)
        super().__init__(filename, flags=flags, extra_columns=extra_columns)
        if isinstance(default_dtype, str):
            default_dtype = getattr(torch, default_dtype)
        self.cutoff = cutoff
        self.compute_neighbor_list = compute_neighbor_list
        self.transforms = transforms if transforms is not None else []
        self.task = task
        self.weight = weight
        self.meta = meta
        self.return_atoms_data = return_atoms_data
        self.default_dtype = default_dtype
        if self.compute_neighbor_list:
            assert isinstance(self.cutoff, float), "Cutoff radius must be given when compute the neighbor list"
            if not any([isinstance(t, NeighborListTransform) for t in self.transforms]):
                self.transforms.append(TorchNeighborList(cutoff=self.cutoff, return_cell_displacements=return_cell_displacements))
        
    def __getitem__(self, idx):
        idx = int(idx)
        length = len(self)
        if idx < 0:
            idx += length
        if idx < 0 or idx >= length:
            raise IndexError(idx)
        cursor = self._get_connection(flags=apsw.SQLITE_OPEN_READONLY).cursor()
        columns = ", ".join(self._read_columns)
        data = cursor.execute(f'''SELECT {columns} FROM data WHERE id=''' + str(idx)).fetchone()
        atoms_data = self._unpack_data_tuple(data, self._read_columns)
        atoms_data = self.dict_to_torch_tensors(atoms_data, default_dtype=self.default_dtype)
        # transform
        for t in self.transforms:
            atoms_data = t(atoms_data)
        if not self.return_atoms_data:
            return atoms_data
        meta = None if self.meta is None else dict(self.meta)
        if meta is not None:
            meta.setdefault("index", idx)
        from .atoms_data import atoms_data_from_dict
        sample = atoms_data_from_dict(atoms_data, task=self.task, weight=self.weight, meta=meta)
        sample.variational = self.get_variational_data(idx)
        return sample
    
    @staticmethod
    def dict_to_torch_tensors(data_dict, default_dtype=torch.float32):
        """
        Converts a dictionary of numpy arrays to a dictionary of PyTorch tensors.
        Int arrays will be converted to torch.long, and float arrays to the given default_dtype.

        :param data_dict: Dictionary containing np.ndarray
        :param default_dtype: PyTorch dtype for floating-point arrays (e.g., torch.float32, torch.float64)
        :return: Dictionary with torch tensors
        """
        tensor_dict = {}
        
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                if np.issubdtype(value.dtype, np.integer):  # Check if it's an integer array
                    tensor_dict[key] = torch.tensor(value, dtype=torch.long)
                elif np.issubdtype(value.dtype, np.floating):  # Check if it's a float array
                    tensor_dict[key] = torch.tensor(value, dtype=default_dtype)
                elif np.issubdtype(value.dtype, bool):
                    tensor_dict[key] = torch.tensor(value, dtype=torch.bool)
                else:
                    raise ValueError(f"Unsupported data type for key '{key}': {value.dtype}")
            else:
                raise ValueError(f"Value for key '{key}' is not a numpy array")

        return tensor_dict

def write_runner_to_db(path_to_input, db_path):
    flags = apsw.SQLITE_OPEN_READWRITE | apsw.SQLITE_OPEN_CREATE
    db = QMDatabase(db_path, flags=flags)
    for line in open(path_to_input):
        if line.startswith('begin'):
            PBC = 0
            C = None
            R, Z, Q_a, F = [], [], [], []
        elif line.startswith('atom'):
            l = line.strip().split()
            R.append(np.array([l[1], l[2], l[3]], dtype=float))
            Z.append(atomic_numbers[l[4]])
            Q_a.append(float(l[5]))
            F.append(np.array([l[-3], l[-2], l[-1]], dtype=float))
        elif line.startswith('energy'):
            E = float(line.strip().split()[1])
        elif line.startswith('charge'):
            Q = float(line.strip().split()[1])
        elif line.startswith('lattice'):
            PBC = 1
            l = line.strip().split()
            if C == None:
                C = [np.array([l[1], l[2], l[3]], dtype=float)]
            else:
                C.append(np.array([l[1], l[2], l[3]], dtype=float))
        elif line.startswith('end'):
            R = np.asarray(R)
            Z = np.asarray(Z)
            Q_a = np.asarray(Q_a)
            F = np.asarray(F)
            D = np.sum(R * Q_a[:, None], axis=0)
            if C != None:
                C = np.asarray(C)

            atoms_data = {
                properties.atomic_numbers: Z,
                properties.pbc: PBC,
                properties.positions: R * units.Bohr / units.Angstrom,
                properties.cell: C * units.Bohr / units.Angstrom,
                properties.energy: E * units.Hartree / units.eV,
                properties.forces: F * (units.Hartree / units.Bohr) / (units.eV / units.Angstrom),
                properties.total_charge: Q,
                properties.atomic_charge: Q_a,
                properties.dipole: D * units.Bohr / units.Angstrom,
            }

            db.add_data(atoms_data)
            
    print(f'{len(db)} structures are extracted from {path_to_input} to {db_path}')
