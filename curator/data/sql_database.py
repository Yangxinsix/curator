import apsw
import numpy as np
import io
import os
import multiprocessing
import torch
from curator.data import properties, NeighborListTransform, Asap3NeighborList
from ase.data import chemical_symbols, atomic_numbers

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

class QMDatabase:
    def __init__(self, filename, flags=apsw.SQLITE_OPEN_READONLY):
        self.db = filename
        self.connections = {}  # allow multiple connections (needed for multi-threading)
        self._open(flags=flags)  # creates the database if it doesn't exist yet

    def __len__(self):
        cursor = self._get_connection(flags=apsw.SQLITE_OPEN_READONLY).cursor()
        return cursor.execute('''SELECT * FROM metadata WHERE id=1''').fetchone()[-1]

    def __getitem__(self, idx):
        cursor = self._get_connection(flags=apsw.SQLITE_OPEN_READONLY).cursor()
        if type(idx) == list:  # for batched data retrieval
            data = cursor.execute('''SELECT * FROM data WHERE id IN (''' + str(idx)[1:-1] + ')').fetchall()
            return [self._unpack_data_tuple(i) for i in data]
        else:
            data = cursor.execute('''SELECT * FROM data WHERE id=''' + str(idx)).fetchone()
            return self._unpack_data_tuple(data)

    def _unpack_data_tuple(self, data):

        # access data tuples
        n_atoms = len(data[1]) // 4  # a single int32 is 4 bytes
        atomic_numbers = self._deblob(data[1], dtype=np.int32, shape=(n_atoms,))
        pbc = np.array([bool(data[2])])
        positions = self._deblob(data[3], dtype=np.float32, shape=(n_atoms, 3))
        cell = None if data[4] is None else self._deblob(data[4], dtype=np.float32, shape=(3, 3))
        energy = np.array([data[5]], dtype=np.float32) if data[5] is not None else None
        forces = self._deblob(data[6], dtype=np.float32, shape=(n_atoms, 3)) if data[6] is not None else None
        virial = None if data[7] is None else self._deblob(data[7], dtype=np.float32, shape=(1, 6))
        stress = None if data[8] is None else self._deblob(data[8], dtype=np.float32, shape=(1, 6))
        total_charge = np.array([data[9]], dtype=np.float32) if data[9] is not None else None
        atomic_charge = None if data[10] is None else self._deblob(data[10], dtype=np.float32, shape=(n_atoms,))
        total_magmom = np.array([data[11]], dtype=np.float32) if data[11] is not None else None
        dipole = None if data[12] is None else self._deblob(data[12], dtype=np.float32, shape=(1, 3))

        # must have properties
        atoms_data = {
            properties.n_atoms: np.array([n_atoms], dtype=np.int64),  # for indexing
            properties.atomic_numbers: atomic_numbers.astype(np.int64),  # for indexing
            properties.pbc: pbc,
            properties.positions: positions,
        }

        # optional properties
        if cell is not None:
            atoms_data[properties.cell] = cell
        if energy is not None:
            atoms_data[properties.energy] = energy
        if forces is not None:
            atoms_data[properties.forces] = forces
        if total_charge is not None:
            atoms_data[properties.total_charge] = total_charge
        if total_magmom is not None:
            atoms_data[properties.total_magmom] = total_magmom
        if atomic_charge is not None:
            atoms_data[properties.atomic_charge] = atomic_charge
        if dipole is not None:
            atoms_data[properties.dipole] = dipole
        if virial is not None:
            atoms_data[properties.virial] = virial
        if stress is not None:
            atoms_data[properties.stress] = stress

        return atoms_data

    def add_data(self, data_dict, flags=apsw.SQLITE_OPEN_READWRITE, transaction=True):
        """
        Add data from a dictionary to the SQLite database.
        :param data_dict: Dictionary containing the data to insert
        :param flags: SQLite access flags
        :param transaction: Boolean flag for handling transactions
        """
        # blob np.ndarray
        data_dict = self._blob_dict(data_dict)

        # Check for NaN values
        vals = [v for k, v in data_dict.items() if k in [properties.atomic_numbers, properties.positions]]
        if self._any_is_nan(*vals):
            print("encountered NaN, data is not added")
            return

        cursor = self._get_connection(flags=flags).cursor()

        if transaction:
            cursor.execute('''BEGIN EXCLUSIVE''')  # Begin exclusive transaction to lock the DB

        try:
            length = cursor.execute('''SELECT * FROM metadata WHERE id=1''').fetchone()[-1]
            keys = ['id']   # id
            vals = [None if length > 0 else 0]
            keys += [k for k in data_dict.keys()]
            vals += [v for v in data_dict.values()]
            columns = ', '.join(keys)
            placeholders = ', '.join('?' * len(vals))
            sql_cmd = f'INSERT INTO data ({columns}) VALUES ({placeholders})'
            cursor.execute(sql_cmd, vals)

            # insert metadata
            cursor.execute('''INSERT OR REPLACE INTO metadata VALUES (?,?)''', (1, length + 1))
            Nmax = cursor.execute('''SELECT * FROM metadata WHERE id=0''').fetchone()[-1]
            if data_dict[properties.atomic_numbers].shape[0] > Nmax:  # Update Nmax if necessary
                cursor.execute('''INSERT OR REPLACE INTO metadata VALUES (?,?)''', (0, data_dict[properties.atomic_numbers].shape[0]))

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

    def _blob_dict(self, data_dict):
        new_dict = {}
        for k, v in data_dict.items():
            if isinstance(v, np.ndarray):
                new_dict[k] = self._blob(v)
            elif v is None:
                new_dict[k] = None
            elif k == properties.pbc:
                new_dict[k] = int(v)
            elif k == properties.energy:
                new_dict[k] = float(v)
            elif k == properties.total_charge:
                new_dict[k] = float(v)
            elif k == properties.total_magmom:
                new_dict[k] = float(v)

        return new_dict

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
            # Create table to store data with full names
            cursor.execute('''CREATE TABLE IF NOT EXISTS data
                (id INTEGER NOT NULL PRIMARY KEY,
                 atomic_numbers BLOB,
                 pbc INTEGER,
                 positions BLOB,
                 cell BLOB,
                 energy FLOAT,
                 forces BLOB,
                 virial BLOB,
                 stress BLOB,
                 total_charge FLOAT,
                 atomic_charge BLOB,
                 total_magmom FLOAT,
                 dipole BLOB)''')

            # Create table to store metadata (information about Nmax and the length, i.e., number of entries)
            cursor.execute('''CREATE TABLE IF NOT EXISTS metadata
                (id INTEGER PRIMARY KEY, N INTEGER)''')

            # Initialize metadata values
            cursor.execute('''INSERT OR IGNORE INTO metadata (id, N) VALUES (?,?)''', (0, 0))  # Nmax
            cursor.execute('''INSERT OR IGNORE INTO metadata (id, N) VALUES (?,?)''', (1, 0))  # num_data

    def _get_connection(self, flags=apsw.SQLITE_OPEN_READONLY):
        '''
        This allows multiple processes to access the database at once,
        every process must have its own connection
        '''
        key = multiprocessing.current_process().name
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
            transforms = [],
            **kwargs,
        ):
        super().__init__(filename, **kwargs)
        self.cutoff = cutoff
        self.compute_neighbor_list = compute_neighbor_list
        self.transforms = transforms
        if self.compute_neighbor_list:
            assert isinstance(self.cutoff, float), "Cutoff radius must be given when compute the neighbor list"
            if not any([isinstance(t, NeighborListTransform) for t in self.transforms]):
                self.transforms.append(Asap3NeighborList(cutoff=self.cutoff, return_cell_displacements=return_cell_displacements))
        
    def __getitem__(self, idx):
        cursor = self._get_connection(flags=apsw.SQLITE_OPEN_READONLY).cursor()
        data = cursor.execute('''SELECT * FROM data WHERE id='''+str(int(idx))).fetchone()
        atoms_data = self._unpack_data_tuple(data)
        atoms_data = self.dict_to_torch_tensors(atoms_data)
        # transform
        for t in self.transforms:
            atoms_data = t(atoms_data)
        return atoms_data
    
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
                properties.positions: R,
                properties.cell: C,
                properties.energy: E,
                properties.forces: F,
                properties.total_charge: Q,
                properties.atomic_charge: Q_a,
                properties.dipole: D,
            }

            db.add_data(atoms_data)
            
    print(f'{len(db)} structures are extracted from {path_to_input} to {db_path}')