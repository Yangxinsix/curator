"""Machine-readable statements of what a dataset can and cannot validate."""

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Tuple


class MissingDatasetCapabilityError(ValueError):
    """A requested training or validation claim is unsupported by the data."""


@dataclass(frozen=True)
class DatasetCapabilityManifest:
    name: str
    version: str
    source: str
    license: str
    labels: Tuple[str, ...]
    can_validate: Tuple[str, ...]
    cannot_validate: Tuple[str, ...]
    notes: Tuple[str, ...] = ()

    def require_labels(self, required: Iterable[str]) -> None:
        missing = sorted(set(required) - set(self.labels))
        if missing:
            raise MissingDatasetCapabilityError(
                "%s is missing required labels: %s"
                % (self.name, ", ".join(missing))
            )

    def require_claims(self, required: Iterable[str]) -> None:
        missing = sorted(set(required) - set(self.can_validate))
        if missing:
            raise MissingDatasetCapabilityError(
                "%s cannot validate: %s" % (self.name, ", ".join(missing))
            )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
