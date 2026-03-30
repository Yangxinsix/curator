from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("apsw")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import apsw

from curator.data import properties
from curator.data.sql_database import QMDatabase, Sqlite3Dataset


TEACHER_ENERGY = "teacher_energy"
TEACHER_FORCES = "teacher_forces"
EXTRA_COLUMNS = {
    TEACHER_ENERGY: {"sql_type": "FLOAT", "storage": "scalar", "dtype": "float32"},
    TEACHER_FORCES: {"sql_type": "BLOB", "storage": "blob", "dtype": "float32", "shape": ["n_atoms", 3]},
}


def _sample_row() -> dict:
    return {
        properties.atomic_numbers: np.array([1, 8], dtype=np.int64),
        properties.pbc: 1,
        properties.positions: np.array([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]], dtype=np.float32),
        properties.cell: np.eye(3, dtype=np.float32),
        properties.energy: np.array([1.25], dtype=np.float32),
        properties.forces: np.array([[0.1, 0.0, -0.1], [0.2, -0.2, 0.3]], dtype=np.float32),
        TEACHER_ENERGY: np.array([2.5], dtype=np.float32),
        TEACHER_FORCES: np.array([[1.0, 1.5, 2.0], [2.5, 3.0, 3.5]], dtype=np.float32),
    }


def test_sqlite_roundtrip_with_teacher_fields(tmp_path):
    db_path = tmp_path / "distill.sqlite"
    db = QMDatabase(
        str(db_path),
        flags=apsw.SQLITE_OPEN_READWRITE | apsw.SQLITE_OPEN_CREATE,
        extra_columns=EXTRA_COLUMNS,
    )
    row = _sample_row()
    db.add_data(row)

    restored = QMDatabase(str(db_path), flags=apsw.SQLITE_OPEN_READONLY)[0]
    assert np.allclose(restored[properties.energy], np.array([1.25], dtype=np.float32))
    assert np.allclose(restored[TEACHER_ENERGY], np.array([2.5], dtype=np.float32))
    assert np.allclose(restored[TEACHER_FORCES], row[TEACHER_FORCES])

    dataset = Sqlite3Dataset(str(db_path), default_dtype=torch.float64)
    sample = dataset[0].to_dict()
    assert sample[TEACHER_ENERGY].dtype == torch.float64
    assert sample[TEACHER_FORCES].dtype == torch.float64
    assert torch.allclose(sample[TEACHER_ENERGY], torch.tensor([2.5], dtype=torch.float64))


def test_sqlite_legacy_schema_without_teacher_columns_is_readable(tmp_path):
    db_path = tmp_path / "legacy.sqlite"
    connection = apsw.Connection(str(db_path), flags=apsw.SQLITE_OPEN_READWRITE | apsw.SQLITE_OPEN_CREATE)
    cursor = connection.cursor()
    cursor.execute(
        '''CREATE TABLE data
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
            dipole BLOB)'''
    )
    cursor.execute('''CREATE TABLE metadata (id INTEGER PRIMARY KEY, N INTEGER)''')
    cursor.execute('''INSERT INTO metadata VALUES (?, ?)''', (0, 2))
    cursor.execute('''INSERT INTO metadata VALUES (?, ?)''', (1, 1))
    cursor.execute(
        '''INSERT INTO data
           (id, atomic_numbers, pbc, positions, cell, energy, forces)
           VALUES (?, ?, ?, ?, ?, ?, ?)''',
        (
            0,
            memoryview(np.array([1, 8], dtype=np.int32)),
            1,
            memoryview(np.array([[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]], dtype=np.float32)),
            memoryview(np.eye(3, dtype=np.float32)),
            0.5,
            memoryview(np.array([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]], dtype=np.float32)),
        ),
    )

    dataset = Sqlite3Dataset(str(db_path))
    sample = dataset[0].to_dict()
    assert properties.energy in sample
    assert TEACHER_ENERGY not in sample
    assert TEACHER_FORCES not in sample


def test_sqlite_cache_metadata_roundtrip(tmp_path):
    db_path = tmp_path / "meta.sqlite"
    db = QMDatabase(str(db_path), flags=apsw.SQLITE_OPEN_READWRITE | apsw.SQLITE_OPEN_CREATE)
    metadata = {
        "schema_version": 1,
        "teacher_labels": [TEACHER_ENERGY, TEACHER_FORCES],
        "source": [{"path": "/tmp/data.traj", "mtime_ns": 1, "size": 2}],
    }

    db.set_cache_metadata(metadata, flags=apsw.SQLITE_OPEN_READWRITE)

    reopened = QMDatabase(str(db_path), flags=apsw.SQLITE_OPEN_READONLY)
    assert reopened.get_cache_metadata() == metadata
