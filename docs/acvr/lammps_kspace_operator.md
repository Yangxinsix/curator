# LAMMPS ACVR point-charge KSpace operator

`compute acvr/kspace` exposes the same point-charge Green operator used by
LAMMPS Coulomb energy and force evaluation:

```text
compute ID sensor-group acvr/kspace source-group
```

For source charges `q`, the per-atom output on `sensor-group` is

```text
phi = K q
```

in energy/charge units.  When the sensor and source groups are the same, the
global scalar is `0.5 q^T K q`.  For different groups it is the cross bilinear
form `q_sensor^T K q_source`.

The operator includes all terms that must remain identical between training,
KKT correction, and production dynamics:

- the real-space point-charge term, including LAMMPS special-Coulomb scaling;
- the reciprocal Ewald/PPPM action;
- active slab or wire boundary corrections;
- the point-charge Ewald self term;
- the uniform neutralizing-background term used by LAMMPS for charged cells.

The implementation deliberately reuses `ElectrodeKSpace::compute_vector` for
the reciprocal and boundary actions, but does not reuse the ELECTRODE package's
Gaussian-electrode matrix.  The latter contains a different short-range/self
model and is not the point-charge kernel used by Curator's Torch Ewald
reference.

## Required LAMMPS styles

The compute requires:

- `atom_style charge`;
- `pair_style coul/long` or a compatible pair style exposing `cut_coul`;
- `kspace_style ewald/electrode` or `pppm/electrode`;
- the CURATOR, ELECTRODE, and KSPACE packages at build time.

A regular `ewald` or `pppm` style is rejected explicitly because it does not
implement the site-potential interface.  This is a capability check, not a
claim that the regular styles compute a different Coulomb energy.

The real-space cutoff is read directly from the active pair style through its
`cut_coul` capability.  Missing or invalid capabilities are rejected, so a
user-supplied cutoff cannot silently create an operator different from the
one used for LAMMPS energy and force evaluation.

## Energy and unit contract

With `units metal`, `phi` is in eV/e, numerically volts for an elementary-charge
state variable, and the scalar is in eV.  The background term matches
`curator.layer.EwaldSummation(neutralizing_background=True)` and LAMMPS's
charged-cell Ewald convention.  There is intentionally no switch for removing
it, because such a switch would make `0.5 q^T K q` inconsistent with LAMMPS's
reported Coulomb energy.

The compute is not itself a constant-potential solver.  It is the matrix-vector
backend needed by `fix curator/acvr` to evaluate and certify

```text
grad_q F = chi + eta * q + K q + boundary terms.
```

Until that fix and the shared timestep cache are implemented, this compute is
an independently testable backend and parity oracle; it must not be combined
with Torch Ewald in one production energy path.

## Regression test

Build an ACVR-enabled LAMMPS and run:

```bash
CURATOR_LAMMPS_ACVR_BIN=/path/to/lmp \
  python -m pytest -q test/test_acvr_lammps_kspace.py
```

The test checks:

1. `0.5 q^T K q == ecoul + elong` for both Ewald and PPPM;
2. `K q == dE/dq` by central finite differences for both backends;
3. per-atom potential and energy parity with the Torch Ewald kernel;
4. fail-closed behavior for a KSpace style without the potential interface.
