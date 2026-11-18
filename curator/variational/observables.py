"""Configured derivatives of one connected scalar energy.

This module knows derivative variables and tensor axes, not property names. The
caller builds the energy graph, including any proposal, coordinate deformation,
and independent uniform-field probe, before passing the live inputs here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real
from typing import Optional

import torch


@dataclass(frozen=True)
class ObservableSpec:
    """Describe a scalar-energy derivative and its presentation.

    ``derivatives`` lists zero, one, or two input names in differentiation order.
    A mixed derivative has axes ``first_input.shape + second_input.shape``.
    ``permutation`` acts on these axes before ``reshape``. Finally, ``factor``
    multiplies the result and a named positive scalar ``normalization`` divides
    it. Normalization happens after differentiation, so stress can divide an
    energy-strain derivative by volume without differentiating energy/volume.
    ``unit`` records the output convention; no implicit unit conversion occurs.

    For a three-vector field and positions shaped [N, 3], a mixed derivative
    with ``derivatives=("external_field", "positions")`` and
    ``permutation=(1, 0, 2)`` has shape [N, 3, 3].

    Boundary derivatives use the live ``control_value`` and
    ``constraint_target`` inputs. For fixed potentials, ``factor=-1`` gives
    minus dE/dV; for fixed charges, ``factor=1`` gives dE/dQ. In predicted
    mode both derivatives include the proposal path, even off stationarity.
    """

    name: str
    derivatives: tuple[str, ...] = ()
    factor: float = 1.0
    permutation: Optional[tuple[int, ...]] = None
    reshape: Optional[tuple[int, ...]] = None
    normalization: Optional[str] = None
    unit: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("observable name must be a nonempty string")
        if isinstance(self.derivatives, str) or not isinstance(self.derivatives, Sequence):
            raise TypeError("derivatives must be a sequence of input names")
        derivatives = tuple(self.derivatives)
        if len(derivatives) > 2:
            raise ValueError("observable specifications support derivative orders zero, one and two")
        if any(not isinstance(name, str) or not name.strip() for name in derivatives):
            raise ValueError("derivative input names must be nonempty strings")
        object.__setattr__(self, "derivatives", derivatives)
        if isinstance(self.factor, bool) or not isinstance(self.factor, Real) or not math.isfinite(self.factor):
            raise ValueError("factor must be a finite real number")
        for name in ("permutation", "reshape"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, str) or not isinstance(value, Sequence):
                raise TypeError(f"{name} must be a sequence of integers")
            value = tuple(value)
            if any(isinstance(axis, bool) or not isinstance(axis, int) for axis in value):
                raise TypeError(f"{name} must contain integers")
            if name == "permutation" and sorted(value) != list(range(len(value))):
                raise ValueError("permutation must contain each output axis exactly once")
            if name == "reshape" and (value.count(-1) > 1 or any(size == 0 or size < -1 for size in value)):
                raise ValueError("reshape dimensions must be positive, with at most one -1")
            object.__setattr__(self, name, value)
        for name in ("normalization", "unit"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty string or None")


# Charge-state conventions: dG/dV = -Q at stationarity and dG/dQ = -lambda.
# These are energy derivatives, not direct state or multiplier predictions.
CHARGE_BOUNDARY_OBSERVABLES = (
    ObservableSpec("control_derivative", ("control_value",), unit="e"),
    ObservableSpec("constraint_derivative", ("constraint_target",), unit="V"),
)


class EnergyObservables:
    """Evaluate configured observables from a single existing energy graph.

    Inputs must be the exact tensors connected to energy and must already have
    ``requires_grad=True`` when a derivative is requested. An unused live input
    gives a zero derivative; a disconnected replacement cannot recover a graph.
    No input or model parameter is detached along the calculation path.

    The short-lived executor caches derivative prefixes. Dense second derivatives
    use one VJP per component of the first input, capped by
    ``max_jacobian_rows`` (nine by default). Put the small perturbation first in a
    mixed derivative. Use ``hvp`` for a large coordinate response instead of a
    full coordinate Hessian. ``create_graph=True`` supports property losses and
    higher parameter derivatives; otherwise returned values are detached.

    ``state_mode='converged'`` allows only detached first derivatives. A detached
    stationary state needs an implicit response calculation for second or higher
    derivatives; this executor deliberately does not substitute its partial AD
    Hessian for the relaxed response. The caller must supply the appropriate
    Lagrangian for first derivatives of geometry-dependent constraints.
    """

    def __init__(
        self,
        energy: torch.Tensor,
        inputs: Mapping[str, torch.Tensor],
        *,
        create_graph: bool = False,
        state_mode: str = "predicted",
        normalizers: Optional[Mapping[str, object]] = None,
        max_jacobian_rows: int = 9,
    ) -> None:
        if not isinstance(energy, torch.Tensor) or energy.ndim != 0 or not energy.is_floating_point():
            raise ValueError("energy must be a floating-point scalar tensor")
        if not isinstance(inputs, Mapping):
            raise TypeError("inputs must map variable names to live tensors")
        for name, value in inputs.items():
            if not isinstance(name, str) or not name:
                raise ValueError("input names must be nonempty strings")
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                raise ValueError(f"input {name!r} must be a floating-point tensor")
        if state_mode not in ("predicted", "converged"):
            raise ValueError("state_mode must be predicted or converged")
        if isinstance(max_jacobian_rows, bool) or not isinstance(max_jacobian_rows, int) or max_jacobian_rows <= 0:
            raise ValueError("max_jacobian_rows must be a positive integer")
        if normalizers is not None and not isinstance(normalizers, Mapping):
            raise TypeError("normalizers must map names to positive scalar values")
        self.energy = energy
        self.inputs = dict(inputs)
        self.create_graph = bool(create_graph)
        self.state_mode = state_mode
        self.normalizers = dict(normalizers or {})
        self.max_jacobian_rows = max_jacobian_rows
        self._cache: dict[tuple[tuple[str, ...], bool], torch.Tensor] = {}

    def _input(self, name: str) -> torch.Tensor:
        if name not in self.inputs:
            raise ValueError(f"unknown derivative input {name!r}")
        value = self.inputs[name]
        if not value.requires_grad:
            raise ValueError(f"input {name!r} must require gradients before constructing energy")
        return value

    def _check_mode(self, order: int, create_graph: bool) -> None:
        if self.state_mode == "converged" and (order > 1 or (order and create_graph)):
            raise ValueError(
                "converged-state higher derivatives require implicit linear response (IFT/HVP); "
                "AD through a detached stationary state gives an incomplete relaxed response"
            )

    @staticmethod
    def _gradient(value: torch.Tensor, variable: torch.Tensor, create_graph: bool) -> torch.Tensor:
        result = None
        if value.requires_grad:
            result = torch.autograd.grad(
                value, variable, create_graph=create_graph, retain_graph=True,
                allow_unused=True,
            )[0]
        if result is None:
            result = torch.zeros_like(variable)
        if create_graph:
            # An affine or unused input still permits another derivative and a
            # zero-gradient backward, without attaching unrelated parameters.
            result = result + variable * 0.0
        return result

    def _raw_derivative(self, variables: tuple[str, ...], create_graph: bool) -> torch.Tensor:
        if not variables:
            return self.energy
        self._check_mode(len(variables), create_graph)
        if (variables, True) in self._cache:
            return self._cache[(variables, True)]
        if (variables, create_graph) in self._cache:
            return self._cache[(variables, create_graph)]
        variable = self._input(variables[-1])
        if len(variables) == 1:
            result = self._gradient(self.energy, variable, create_graph)
        else:
            first = self._input(variables[0])
            if first.numel() > self.max_jacobian_rows:
                raise ValueError(
                    f"dense second derivative needs {first.numel()} VJP rows, exceeding "
                    f"max_jacobian_rows={self.max_jacobian_rows}; put the small perturbation "
                    "first in a mixed derivative, or request hvp for coordinate response"
                )
            gradient = self._raw_derivative(variables[:-1], True)
            rows = [self._gradient(component, variable, create_graph) for component in gradient.reshape(-1)]
            result = (torch.stack(rows).reshape(gradient.shape + variable.shape) if rows
                      else gradient.new_zeros(gradient.shape + variable.shape))
            if create_graph and not rows:
                result = result + gradient.sum() * 0 + variable.sum() * 0
        self._cache[(variables, create_graph)] = result
        return result

    def derivative(self, variables: Sequence[str], *, create_graph: Optional[bool] = None) -> torch.Tensor:
        """Return a raw ordered derivative, retaining native input-axis shapes."""

        spec = ObservableSpec("derivative", derivatives=variables)
        graph = self.create_graph if create_graph is None else bool(create_graph)
        with torch.enable_grad():
            value = self._raw_derivative(spec.derivatives, graph)
        return value if graph else value.detach()

    def _normalizer(self, name: str, reference: torch.Tensor) -> torch.Tensor:
        if name not in self.normalizers:
            raise ValueError(f"unknown normalization {name!r}")
        value = self.normalizers[name]
        if not isinstance(value, (Real, torch.Tensor)) or isinstance(value, bool):
            raise ValueError(f"normalization {name!r} must be a positive scalar")
        value = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
        if value.numel() != 1 or not bool(torch.isfinite(value).all()) or not bool((value > 0).all()):
            raise ValueError(f"normalization {name!r} must be a finite positive scalar")
        return value.reshape(())

    def evaluate(self, specs: Sequence[ObservableSpec | Mapping]) -> dict[str, torch.Tensor]:
        """Evaluate specifications or configuration dictionaries with shared AD."""

        if isinstance(specs, (str, Mapping)) or not isinstance(specs, Sequence):
            raise TypeError("specs must be a sequence of ObservableSpec objects or dictionaries")
        parsed = [ObservableSpec(**spec) if isinstance(spec, Mapping) else spec for spec in specs]
        if any(not isinstance(spec, ObservableSpec) for spec in parsed):
            raise TypeError("specs must contain ObservableSpec objects or dictionaries")
        if len({spec.name for spec in parsed}) != len(parsed):
            raise ValueError("observable names must be unique")
        results = {}
        with torch.enable_grad():
            # Precompute shared prefixes with graphs, even when the requested
            # first-derivative output occurs before a second derivative in YAML.
            for spec in parsed:
                self._check_mode(len(spec.derivatives), self.create_graph)
                if len(spec.derivatives) == 2:
                    self._raw_derivative(spec.derivatives[:1], True)
            for spec in parsed:
                value = self._raw_derivative(spec.derivatives, self.create_graph)
                if spec.permutation is not None:
                    if len(spec.permutation) != value.ndim:
                        raise ValueError(f"observable {spec.name!r} permutation does not match derivative rank")
                    value = value.permute(spec.permutation)
                if spec.reshape is not None:
                    try:
                        value = value.reshape(spec.reshape)
                    except RuntimeError as error:
                        raise ValueError(f"observable {spec.name!r} reshape does not match derivative size") from error
                value = value * spec.factor
                if spec.normalization is not None:
                    value = value / self._normalizer(spec.normalization, value)
                results[spec.name] = value if self.create_graph else value.detach()
        return results

    def hvp(
        self, variable: str, direction: torch.Tensor, *, create_graph: Optional[bool] = None,
    ) -> torch.Tensor:
        """Return the energy Hessian-vector product without a dense Hessian.

        ``direction`` is an autograd cotangent: if it depends on the input, its
        derivative is excluded from the HVP value but retained for a later
        backward when ``create_graph=True``.
        """

        graph = self.create_graph if create_graph is None else bool(create_graph)
        self._check_mode(2, graph)
        point = self._input(variable)
        if not isinstance(direction, torch.Tensor) or direction.shape != point.shape:
            raise ValueError("HVP direction must have the same shape as the input")
        if direction.dtype != point.dtype or direction.device != point.device:
            raise ValueError("HVP direction must share the input dtype and device")
        with torch.enable_grad():
            gradient = self._raw_derivative((variable,), True)
            result = torch.autograd.grad(
                gradient, point, grad_outputs=direction, create_graph=graph,
                retain_graph=True, allow_unused=True,
            )[0]
            if result is None:
                result = torch.zeros_like(point)
            if graph:
                result = result + point * 0.0 + direction * 0.0
        return result if graph else result.detach()


class BatchedEnergyObservables:
    """Shared VJPs for explicitly independent scalar-energy instances.

    The caller must guarantee that energy i depends only on input views i, apart
    from shared model parameters. Cross-geometry pooling or training BatchNorm
    violates that assumption. This class does not infer independence from tensor
    shapes. Supply the existing live input views, not slices made after energy
    evaluation. Multiple conditions for one geometry must use the ordinary
    executor instead.

    One VJP computes all requested first derivatives. For each small first-input
    component, another VJP computes every requested second-input derivative at
    once. Three-vector field derivatives therefore need four VJPs in total,
    including forces, dipoles, polarizabilities and mixed coordinate responses.
    Observable definitions and output transformations remain unchanged.
    """

    def __init__(
        self, energies: Sequence[torch.Tensor], inputs: Mapping[str, Sequence[torch.Tensor]],
        *, create_graph: bool = False, state_mode: str = "predicted",
        normalizers: Optional[Mapping[str, Sequence[object]]] = None,
        max_jacobian_rows: int = 9,
    ) -> None:
        if isinstance(energies, torch.Tensor) or not isinstance(energies, Sequence) or not energies:
            raise ValueError("energies must be a nonempty sequence of existing scalar tensors")
        if not isinstance(inputs, Mapping):
            raise TypeError("inputs must map variable names to sequences of live input views")
        count = len(energies)
        for name, values in inputs.items():
            if isinstance(values, torch.Tensor) or not isinstance(values, Sequence) or len(values) != count:
                raise ValueError(f"input {name!r} must contain one existing live tensor per energy")
            if len({id(value) for value in values}) != count:
                raise ValueError(f"input {name!r} repeats a tensor; independent input views are required")
        normalizers = normalizers or {}
        if not isinstance(normalizers, Mapping):
            raise TypeError("normalizers must map names to per-instance scalar values")
        for name, values in normalizers.items():
            if not isinstance(values, (Sequence, torch.Tensor)) or len(values) != count:
                raise ValueError(f"normalization {name!r} must contain one scalar per energy")
        self.energies = tuple(energies)
        self.instances = tuple(EnergyObservables(
            energy, {name: values[index] for name, values in inputs.items()},
            create_graph=create_graph, state_mode=state_mode,
            normalizers={name: values[index] for name, values in normalizers.items()},
            max_jacobian_rows=max_jacobian_rows,
        ) for index, energy in enumerate(energies))
        self.create_graph = bool(create_graph)
        self.max_jacobian_rows = max_jacobian_rows

    @staticmethod
    def _gradients(value, inputs, create_graph):
        gradients = ((None,) * len(inputs) if not value.requires_grad else torch.autograd.grad(
            value, inputs, create_graph=create_graph, retain_graph=True, allow_unused=True,
        ))
        return tuple(
            (torch.zeros_like(point) if gradient is None else gradient)
            + (point * 0.0 if create_graph else 0.0)
            for point, gradient in zip(inputs, gradients)
        )

    def evaluate(self, specs: Sequence[ObservableSpec | Mapping]) -> dict[str, tuple[torch.Tensor, ...]]:
        if isinstance(specs, (str, Mapping)) or not isinstance(specs, Sequence):
            raise TypeError("specs must be a sequence of ObservableSpec objects or dictionaries")
        parsed = tuple(ObservableSpec(**spec) if isinstance(spec, Mapping) else spec for spec in specs)
        if any(not isinstance(spec, ObservableSpec) for spec in parsed):
            raise TypeError("specs must contain ObservableSpec objects or dictionaries")
        if len({spec.name for spec in parsed}) != len(parsed):
            raise ValueError("observable names must be unique")
        first_names = tuple(dict.fromkeys(spec.derivatives[0] for spec in parsed if spec.derivatives))
        second_names = {}
        for spec in parsed:
            self.instances[0]._check_mode(len(spec.derivatives), self.create_graph)
            if len(spec.derivatives) == 2:
                first, second = spec.derivatives
                second_names.setdefault(first, {})[second] = None
        for first in second_names:
            sizes = [instance._input(first).numel() for instance in self.instances]
            if len(set(sizes)) != 1 or sizes[0] > self.max_jacobian_rows:
                raise ValueError("batched second derivatives require equal small first-input sizes; use the ordinary executor or HVP")
        count = len(self.instances)
        with torch.enable_grad():
            if first_names:
                points = tuple(instance._input(name) for name in first_names for instance in self.instances)
                first_graph = self.create_graph or bool(second_names)
                gradients = self._gradients(torch.stack(self.energies).sum(), points, first_graph)
                for variable_index, name in enumerate(first_names):
                    for index, instance in enumerate(self.instances):
                        instance._cache[((name,), first_graph)] = gradients[variable_index * count + index]
            for first, targets in second_names.items():
                targets = tuple(targets)
                points = tuple(instance._input(name) for name in targets for instance in self.instances)
                rows = {name: [] for name in targets}
                first_gradients = tuple(instance._cache[((first,), True)] for instance in self.instances)
                for component in range(first_gradients[0].numel()):
                    scalar = torch.stack([gradient.reshape(-1)[component] for gradient in first_gradients]).sum()
                    gradients = self._gradients(scalar, points, self.create_graph)
                    for variable_index, name in enumerate(targets):
                        rows[name].append(gradients[variable_index * count:(variable_index + 1) * count])
                for name in targets:
                    for index, instance in enumerate(self.instances):
                        shape = instance.inputs[first].shape + instance.inputs[name].shape
                        value = (torch.stack([row[index] for row in rows[name]]).reshape(shape) if rows[name]
                                 else instance.inputs[first].new_zeros(shape))
                        if self.create_graph and not rows[name]:
                            value = value + instance.inputs[first].sum() * 0 + instance.inputs[name].sum() * 0
                        instance._cache[((first, name), self.create_graph)] = value
            # Reuse the ordinary executor's axis, factor, normalization and
            # detach semantics. All required derivative entries are cached.
            outputs = tuple(instance.evaluate(parsed) for instance in self.instances)
        return {spec.name: tuple(output[spec.name] for output in outputs) for spec in parsed}
