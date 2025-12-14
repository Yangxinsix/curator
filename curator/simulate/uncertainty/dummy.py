from __future__ import annotations

from typing import Any, Dict

from ase import Atoms


class ConstantUncertainty:
    """
    Simple uncertainty backend for testing and demos.

    Returns a fixed value for a single key and optional warning/outlier flags.
    """

    def __init__(
        self,
        value: float = 0.1,
        key: str = "uncertainty",
        is_warning: bool = False,
        is_outlier: bool = False,
    ) -> None:
        self.value = float(value)
        self.key = key
        self.is_warning = is_warning
        self.is_outlier = is_outlier
        self.threshold_key = key
        self.uncertainty_keys = (key,)

    def __call__(self, atoms: Atoms) -> Dict[str, Any]:
        return {
            self.key: self.value,
            "is_warning": self.is_warning,
            "is_outlier": self.is_outlier,
        }
