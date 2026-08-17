# ADR 0003: Thermodynamic boundaries, Green functions, and backends

Status: accepted, 2026-08-17

## Decision

Thermodynamic and spatial electrostatic boundaries are orthogonal settings.

- Fixed charge, fixed potential, fragment charge, and Robin reservoirs specify
  what is controlled thermodynamically.
- 3D periodic tin-foil, slab-corrected 2D periodicity, open boundaries, and
  metallic image boundaries select the electrostatic Green function.

Choosing fixed potential does not automatically choose a slab Green function;
choosing Ewald tin-foil does not impose a constant electrode potential.

The PyTorch Ewald implementation is the differentiable reference and training
backend. Production long-range molecular dynamics will use LAMMPS Ewald/PPPM.
Both backends must expose the same energy, state-gradient/matvec, force, and
virial semantics and pass numerical parity tests. The local model and KSpace
backend must never both count the same long-range energy.

Static ACVR solves the fast internal state for the current boundary. Dynamic
constant-current, RC/RLC, AC, or stochastic potentiostat protocols are separate
controllers that update the boundary or an extended state; they are not hidden
inside a static minimizer.
