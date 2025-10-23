from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, List
from .context import SimContext

class BaseEngine(ABC):
    """
    Pluggable engine interface.
    In shell-based engines, attach() is usually a no-op (no per-step callbacks).
    """
    def __init__(self):
        self._attached: List[tuple[Callable[[], None], int]] = []
    def attach(self, fn: Callable[[], None], interval: int = 1) -> None:
        self._attached.append((fn, interval))
        self._attach_to_backend(fn, interval)
    @abstractmethod
    def setup(self, ctx: SimContext) -> None: ...
    @abstractmethod
    def run(self, **run_kwargs) -> None: ...
    @abstractmethod
    def _attach_to_backend(self, fn: Callable[[], None], interval: int) -> None: ...