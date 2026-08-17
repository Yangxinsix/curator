# ACVR multi-condition training contract

ACVR batches keep one geometry and attach a ragged list of charge, voltage, or
other boundary conditions.  The loss therefore must not treat each condition as
an independent duplicated structure.

## Prediction namespace

Targets are stored in `batch["variational"]`.  A model must write predictions to
`prediction["variational_prediction"]`.  The names are intentionally different:
the legacy Curator training loop copies missing batch keys into its prediction
dictionary, and sharing a name would silently compare a target with itself.

`VariationalJointLoss` defaults to geometry-balanced reduction:

1. average scalar errors inside each label or response block;
2. average active conditions inside each geometry using condition weights;
3. average geometries using structure weights.

This prevents a geometry with eight voltage labels from receiving eight times
the weight of a geometry with one label.  `balance_mode="condition"` is available
only when that alternative sampling measure is intentional.  Every term reports
its unweighted numerator and denominator so distributed logging can reduce sums,
not averages of rank-local means.

## Units and semantic identity

Every positive supervised term requires a physical scale.  Prediction and target
units and ragged shapes must agree.  Energy-like labels can additionally require
metadata such as `energy_kind` and `reference`; this is necessary because a
fixed-electron-number energy and an electronic grand potential are not
interchangeable labels.

Response blocks are matched by condition, input/output references, tensor shape,
unit, gauge, and boundary hash.  List order is not part of their identity.

## Variational certificates

KKT penalties are accepted only with
`source="compiled_variational_operator"` and explicit dimensionless
normalization.  They must be recomputed from the learned functional, predicted
state and multipliers, and the compiled boundary.  A neural network output named
"residual" is not a certificate.

Stability supervision similarly uses the metric-normalized minimum eigenvalue of
the **reduced Lagrangian Hessian**.  A full objective Hessian is incorrect in the
presence of constraints.  Conditions whose boundary semantics are marked
reference-only or pending audit are rejected from these certificate losses.

The current TRECI reader intentionally marks its fixed-potential thermodynamic
mapping as pending audit.  Its energies and Bader charges can be inspected and
trained as explicitly named source labels, but must not yet be relabeled as an
ACVR grand potential or exact physical electrode charge.
