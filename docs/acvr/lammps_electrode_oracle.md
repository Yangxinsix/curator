# LAMMPS ELECTRODE oracle for ACVR

The ACVR production backend will eventually reuse LAMMPS KSpace site
potentials and matrix-vector products.  Until that interface exists, the
official ELECTRODE examples provide an independent end-to-end reference for
the thermodynamic meaning of constant potential and constant charge.

## Reproducible reference build

Reference source: `examples/PACKAGES/electrode/graph-il` in LAMMPS
`22 Jul 2025 - Update 1`.

```bash
cmake -S /path/to/lammps/cmake -B /tmp/acvr-lammps-electrode \
  -D CMAKE_BUILD_TYPE=Release \
  -D BUILD_MPI=off -D BUILD_OMP=off -D BUILD_SHARED_LIBS=off \
  -D PKG_ELECTRODE=on -D PKG_KSPACE=on \
  -D PKG_MOLECULE=on -D PKG_RIGID=on
cmake --build /tmp/acvr-lammps-electrode --parallel 8
```

Run `in.conp` and `in.conq` in separate copies of the official example
directory.  The isolated CPU build uses double precision KISS FFT and does not
modify the Curator production LAMMPS installation.

## Recorded smoke-test result

Both official 500-step simulations completed for 3776 atoms.

| Boundary | Imposed quantity | Step 0 response | Step 500 response |
|---|---|---:|---:|
| `electrode/conp` | bottom/top potential = -1/+1 V | bottom/top charge = +0.008514/-0.008514 e | -0.210375/+0.210375 e |
| `electrode/conq` | bottom/top total charge = -1/+1 e | potential difference = 20.029196 V | 8.767136 V |

These values are not cross-ensemble parity targets because the two official
inputs impose different thermodynamic states.  They verify the essential
roles: constant potential solves for charge, whereas constant charge measures
the resulting potential.

The public parser is deliberately read-only:

```python
from curator.variational import read_electrode_thermo

rows = read_electrode_thermo("log.conq", boundary="constant_charge")
print(rows[-1].potential_difference)
```

## What this oracle does not yet prove

This smoke test does not establish Curator/LAMMPS operator parity.  Gate G6
still requires the same structure, Green-function boundary, Gaussian width,
units and internal state to agree in:

- site potential / Hessian-vector product;
- optimized charge and KKT residual;
- energy, force and virial;
- single rank, MPI and GPU execution.

The parser and recorded run make that future comparison reproducible without
pretending that a successful stock LAMMPS example is already an ACVR backend.
