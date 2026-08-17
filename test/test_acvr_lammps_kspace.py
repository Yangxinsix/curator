"""Parity tests for the CURATOR-owned LAMMPS ACVR KSpace operator.

Set ``CURATOR_LAMMPS_ACVR_BIN`` to a LAMMPS executable built with the
CURATOR, ELECTRODE, and KSPACE packages.  The tests are skipped otherwise.
"""

import math
import os
import re
import subprocess

from ase import Atoms
from ase.neighborlist import neighbor_list
import pytest
import torch

from curator.layer import EwaldSummation


LAMMPS_BIN = os.environ.get("CURATOR_LAMMPS_ACVR_BIN")
pytestmark = pytest.mark.skipif(
    not LAMMPS_BIN,
    reason="CURATOR_LAMMPS_ACVR_BIN does not name an ACVR-enabled LAMMPS",
)

CELL_LENGTH = 12.0
CUTOFF = 5.5
ALPHA = 0.4
KMAX = 8
POSITIONS = ((2.1, 3.2, 4.3), (5.4, 4.6, 5.1))
RESULT_PATTERN = re.compile(
    r"ACVR_RESULT\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)"
)


def _input_text(charges, dump_path, *, kspace_style="ewald/electrode"):
    if kspace_style == "pppm/electrode":
        kspace_modify = (
            f"kspace_modify gewald {ALPHA} mesh 32 32 32 order 5"
        )
    else:
        kspace_modify = (
            f"kspace_modify gewald {ALPHA} "
            f"kmax/ewald {KMAX} {KMAX} {KMAX}"
        )
    return f"""units metal
atom_style charge
boundary p p p
region box block 0 {CELL_LENGTH} 0 {CELL_LENGTH} 0 {CELL_LENGTH} units box
create_box 1 box
create_atoms 1 single {POSITIONS[0][0]} {POSITIONS[0][1]} {POSITIONS[0][2]} units box
create_atoms 1 single {POSITIONS[1][0]} {POSITIONS[1][1]} {POSITIONS[1][2]} units box
mass 1 1.0
set atom 1 charge {charges[0]:.17g}
set atom 2 charge {charges[1]:.17g}
pair_style coul/long {CUTOFF}
pair_coeff * *
kspace_style {kspace_style} 1.0e-12
{kspace_modify}
compute phi all acvr/kspace all
thermo 1
thermo_style custom step pe ecoul elong c_phi
dump values all custom 1 {dump_path} id q c_phi
dump_modify values sort id format line "%d %.17g %.17g"
run 0
variable total equal ecoul+elong
variable acvr equal c_phi
print "ACVR_RESULT $(v_total:%.17g) $(v_acvr:%.17g)"
"""


def _read_dump(path):
    lines = path.read_text().splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("ITEM: ATOMS"))
    values = [line.split() for line in lines[header + 1 : header + 3]]
    return torch.tensor([float(row[2]) for row in values], dtype=torch.float64)


def _run_lammps(tmp_path, charges, *, name="state", kspace_style="ewald/electrode"):
    run_dir = tmp_path / name
    run_dir.mkdir()
    input_path = run_dir / "in.acvr"
    dump_path = run_dir / "phi.dump"
    input_path.write_text(_input_text(charges, dump_path, kspace_style=kspace_style))
    process = subprocess.run(
        [LAMMPS_BIN, "-in", str(input_path)],
        cwd=run_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    output = process.stdout + process.stderr
    if process.returncode:
        return process, output, None
    match = RESULT_PATTERN.search(output)
    if match is None:
        raise AssertionError(f"LAMMPS output did not contain ACVR_RESULT:\n{output}")
    total_energy, operator_energy = (float(value) for value in match.groups())
    return process, output, (total_energy, operator_energy, _read_dump(dump_path))


def _torch_kernel():
    cell = torch.eye(3, dtype=torch.float64) * CELL_LENGTH
    positions = torch.tensor(POSITIONS, dtype=torch.float64)
    atoms = Atoms(
        numbers=[1, 1],
        positions=positions.numpy(),
        cell=cell.numpy(),
        pbc=True,
    )
    center, neighbor, shift = neighbor_list(
        "ijS", atoms, CUTOFF, self_interaction=False
    )
    edge_index = torch.tensor(
        list(zip(center, neighbor)), dtype=torch.long
    ).reshape(-1, 2)
    cell_shift = torch.tensor(shift, dtype=torch.float64) @ cell
    edge_difference = (
        positions[edge_index[:, 1]]
        - positions[edge_index[:, 0]]
        + cell_shift
    )
    edge_distance = torch.linalg.vector_norm(edge_difference, dim=1)

    # LAMMPS Ewald retains vectors with |k|^2 <= gsqmx and expands gsqmx by
    # 1.00001 after applying kmax/ewald.  Match that finite operator exactly.
    k_cutoff = (
        2.0 * math.pi / CELL_LENGTH * KMAX * math.sqrt(1.00001)
    )
    ewald = EwaldSummation(
        alpha=ALPHA,
        cutoff=CUTOFF,
        k_cutoff=k_cutoff,
        neutralizing_background=True,
    )
    return ewald.get_ewald_kernel(
        cell.unsqueeze(0),
        torch.tensor([2], dtype=torch.long),
        positions,
        edge_distance,
        edge_index,
    )[0]


def test_lammps_operator_energy_and_torch_kernel_parity(tmp_path):
    charge = torch.tensor([0.7, -0.2], dtype=torch.float64)
    process, output, result = _run_lammps(tmp_path, charge.tolist())
    assert process.returncode == 0, output
    total_energy, operator_energy, potential = result

    assert operator_energy == pytest.approx(total_energy, abs=2.0e-8)
    kernel = _torch_kernel()
    torch.testing.assert_close(potential, kernel @ charge, rtol=0.0, atol=1.0e-7)
    torch.testing.assert_close(
        torch.tensor(operator_energy, dtype=torch.float64),
        0.5 * charge @ kernel @ charge,
        rtol=0.0,
        atol=5.0e-8,
    )


def test_site_potential_is_charge_derivative(tmp_path):
    charge = torch.tensor([0.7, -0.2], dtype=torch.float64)
    process, output, result = _run_lammps(tmp_path, charge.tolist(), name="base")
    assert process.returncode == 0, output
    potential = result[2]

    step = 1.0e-5
    derivative = torch.empty_like(charge)
    for index in range(charge.numel()):
        plus = charge.clone()
        minus = charge.clone()
        plus[index] += step
        minus[index] -= step
        plus_result = _run_lammps(
            tmp_path, plus.tolist(), name=f"plus_{index}"
        )[2]
        minus_result = _run_lammps(
            tmp_path, minus.tolist(), name=f"minus_{index}"
        )[2]
        derivative[index] = (plus_result[0] - minus_result[0]) / (2.0 * step)

    torch.testing.assert_close(potential, derivative, rtol=0.0, atol=2.0e-7)


def test_pppm_operator_energy_and_charge_derivative(tmp_path):
    charge = torch.tensor([0.7, -0.2], dtype=torch.float64)
    process, output, result = _run_lammps(
        tmp_path,
        charge.tolist(),
        name="pppm_base",
        kspace_style="pppm/electrode",
    )
    assert process.returncode == 0, output
    total_energy, operator_energy, potential = result
    assert operator_energy == pytest.approx(total_energy, abs=2.0e-8)

    step = 1.0e-5
    derivative = torch.empty_like(charge)
    for index in range(charge.numel()):
        plus = charge.clone()
        minus = charge.clone()
        plus[index] += step
        minus[index] -= step
        plus_result = _run_lammps(
            tmp_path,
            plus.tolist(),
            name=f"pppm_plus_{index}",
            kspace_style="pppm/electrode",
        )[2]
        minus_result = _run_lammps(
            tmp_path,
            minus.tolist(),
            name=f"pppm_minus_{index}",
            kspace_style="pppm/electrode",
        )[2]
        derivative[index] = (plus_result[0] - minus_result[0]) / (2.0 * step)

    torch.testing.assert_close(potential, derivative, rtol=0.0, atol=3.0e-7)


def test_regular_ewald_is_rejected_instead_of_silently_changing_operator(tmp_path):
    process, output, result = _run_lammps(
        tmp_path,
        [0.7, -0.2],
        kspace_style="ewald",
    )
    assert process.returncode != 0
    assert result is None
    assert "requires ewald/electrode or pppm/electrode" in output
