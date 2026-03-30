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
            shape = None if spec.get("shape") is None else tuple(
                n_atoms if dim == "n_atoms" else dim for dim in spec["shape"]
            )
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
        cursor = self._get_connection(flags=apsw.SQLITE_OPEN_READONLY).cursor()
        columns = ", ".join(self._read_columns)
        data = cursor.execute(f'''SELECT {columns} FROM data WHERE id=''' + str(int(idx))).fetchone()
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
        return atoms_data_from_dict(atoms_data, task=self.task, weight=self.weight, meta=meta)
    
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
