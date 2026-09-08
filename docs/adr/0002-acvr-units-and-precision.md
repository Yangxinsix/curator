# ADR 0002: ACVR units and precision

Status: accepted, 2026-08-17

## Physical units

Curator's reference electrostatic implementation uses:

| Quantity | Unit |
|---|---|
| position, cell | Å |
| charge | elementary-charge number $e$ |
| energy | eV |
| force | eV/Å |
| virial | eV |
| electric potential / charge chemical potential | V, numerically $\mathrm{eV}/e$ |
| charge Hessian / hardness | $\mathrm{eV}/e^2$ |
| dipole | $e\,\text{Å}$ |
| electric field conjugate to dipole | V/Å |

Potential and chemical potential may have the same numerical conversion but
retain different semantic names. A unit string is validated at boundary
compilation; implicit conversions are not performed by the algebra core.

## Precision

- Neural-network features and amortized predictions may use float32, bfloat16,
  or a configured mixed-precision policy.
- Constraint compilation, exact solve, correction, response, rank decisions,
  stability checks, and certification use float64 reference arithmetic unless
  a caller explicitly supplies another floating dtype.
- Rank thresholds scale with the largest singular value and matrix dimension.
- Stationarity and primal residuals are stored separately because they have
  different units. Reliability must not compare their raw values by taking a
  single unscaled maximum.
- The reported dense KKT condition number is explicitly an unscaled diagnostic;
  it cannot be compared across variables or constraint scalings as a physical
  reliability measure.
