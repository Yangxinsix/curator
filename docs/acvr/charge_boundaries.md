# Charge boundary recipes and their physical meaning

The charge plugin uses one functional and changes only the thermodynamic
boundary. For

\[
F(q)=\chi^Tq+\frac12q^THq,
\]

the atomic charge chemical potential is

\[
\mu=\frac{\partial F}{\partial q}=Hq+\chi.
\]

## Fixed total or fragment charge

A fixed group charge is the equality

\[
\sum_{i\in g}q_i=Q_g.
\]

At the optimum, every freely redistributing atom in that group has
\(\mu_i=-\lambda_g\). Therefore a floating conductor is represented by a
fixed group charge: its common potential is an output, not an input.

## Fixed electrode potential

A fixed potential is a Legendre control,

\[
\Omega(q;V)=F(q)-\sum_gV_g\sum_{i\in g}q_i.
\]

It is never compiled as `potential(q) = V` or as an extra charge equality.
For an unconstrained electrode atom, stationarity gives \(\mu_i=V_g\). The
electrode charge is then a response. A fixed-\(Q\) solve at that response
charge recovers the same state and has \(-\lambda=V\).

The differential capacitance of a single quadratic electrode is

\[
\frac{dQ}{dV}=\mathbf1^TH^{-1}\mathbf1,
\]

which is tested against finite differences.

## Electrode--electrolyte mixed boundaries

Individual electrolyte charges may always redistribute. Their *total* charge
still needs a physical boundary:

- `electrolyte_total_charge=Q`: closed electrolyte electron count;
- an overall `total_charge`/`neutrality`: charge can move between electrolyte
  and electrodes while the complete variable-charge system is closed; this
  requires `allow_electrode_electrolyte_charge_transfer=True` because it is a
  redox-capable model, not an innocent numerical default;
- `electrolyte_potential=mu_e`: electron-open electrolyte connected to an
  explicit reservoir.

Omitting all three is rejected. It would silently impose a zero reference
chemical potential and make an apparently innocent implementation choice into
a physical electron reservoir.

The specialized recipes also require every declared fragment/electrode to have
one target/control and, at compile time, require their groups to cover every
variable-charge atom exactly once. Use the general `charge_boundary_recipe`
only when partially controlled groups are intentional.

Fixing the complete electrolyte total still permits charge transfer between
its molecules. Molecular/ionic fragment constraints are needed when each
species must retain its electron count.

## Checkpoint identifiability

A checkpoint trained only at one fixed total charge does not determine the
absolute electronegativity/potential gauge. Adding the same constant to every
component of \(\chi\) leaves the fixed-charge state unchanged, but changes the
predicted total charge in a fixed-potential run. Therefore switching an old
fixed-\(Q\) checkpoint to electron-open fixed-\(V\) is not justified by the new
solver alone. It needs multiple charge/potential states, chemical-potential
labels, or an explicit absolute-potential calibration.

## Gauge and stability

If a total charge equality makes a common shift of all controlled potentials
irrelevant, declare a `GaugeSpec` (`zero_mean`, `reference`, or an explicit
`allow`). This thermodynamic gauge is separate from the Ewald/Poisson spatial
potential reference.

Fixed potential allows total charge fluctuations. The effective Hessian must
therefore remain positive on every allowed fluctuation direction. Positive
onsite hardness alone is insufficient: the complete local plus long-range
Hessian is checked, and an unbounded fixed-potential problem is rejected.

## Public recipes

```python
from curator.variational.variables import (
    fixed_total_charge_recipe,
    fixed_fragment_charges_recipe,
    fixed_electrode_potentials_recipe,
    floating_conductor_recipe,
    mixed_electrode_electrolyte_recipe,
)
```

These recipes only define thermodynamic boundaries. The Green function,
periodicity, slab correction and metallic image boundary belong to the chosen
electrostatic operator/backend.
