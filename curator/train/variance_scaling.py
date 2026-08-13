"""Model-independent fitting for hidden-feature variance scales."""

from __future__ import annotations

from collections import OrderedDict
from itertools import islice
from typing import Any, Callable, Iterable, Optional

from torch import nn

from curator.layer import VarianceScale


def find_variance_scales(model: nn.Module) -> "OrderedDict[str, VarianceScale]":
    """Return all variance scales in module traversal order."""
    return OrderedDict(
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, VarianceScale)
    )


def ensure_variance_scales_fitted(model: nn.Module) -> None:
    """Raise when a model contains an unfitted variance scale."""
    missing = [
        name
        for name, module in find_variance_scales(model).items()
        if not bool(module.fitted.item())
    ]
    if missing:
        raise RuntimeError(
            "Unfitted VarianceScale modules: " + ", ".join(missing)
        )


def fit_variance_scales(
    model: nn.Module,
    batches: Iterable[Any],
    *,
    num_batches: int = 16,
    forward: Optional[Callable[[Any], Any]] = None,
    reset: bool = True,
) -> "OrderedDict[str, float]":
    """Fit every variance scale in actual forward execution order.

    ``forward`` may close over any device transfer or input adaptation required by
    the caller. By default the model is called directly with each batch.
    """
    if num_batches <= 0:
        raise ValueError("num_batches must be positive.")
    scales = find_variance_scales(model)
    if not scales:
        return OrderedDict()

    observed_batches = list(islice(iter(batches), num_batches))
    if not observed_batches:
        raise ValueError("Cannot fit variance scales without batches.")
    run = model if forward is None else forward

    if reset:
        for module in scales.values():
            module.reset()
    targets = OrderedDict(
        (name, module)
        for name, module in scales.items()
        if not bool(module.fitted.item())
    )
    if not targets:
        return OrderedDict(
            (name, float(module.scale.item()))
            for name, module in scales.items()
        )

    execution_order = []
    handles = []
    for name, module in targets.items():
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, _output, name=name: (
                    execution_order.append(name)
                    if name not in execution_order
                    else None
                )
            )
        )

    was_training = model.training
    model.eval()
    try:
        run(observed_batches[0])
        for handle in handles:
            handle.remove()
        handles.clear()

        missing = set(targets) - set(execution_order)
        if missing:
            raise RuntimeError(
                "VarianceScale modules were not executed: "
                + ", ".join(sorted(missing))
            )

        for name in execution_order:
            module = targets[name]
            module.start_fitting()
            for batch in observed_batches:
                run(batch)
            module.fit()
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return OrderedDict(
        (name, float(module.scale.item()))
        for name, module in scales.items()
    )


__all__ = [
    "ensure_variance_scales_fitted",
    "find_variance_scales",
    "fit_variance_scales",
]
