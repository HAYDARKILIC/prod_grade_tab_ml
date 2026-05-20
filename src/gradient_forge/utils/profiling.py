"""High-resolution timing and peak-RSS profiling."""

from __future__ import annotations

import time
import tracemalloc
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from types import TracebackType


@dataclass
class Stopwatch(AbstractContextManager["Stopwatch"]):
    """Context manager wrapping ``time.perf_counter_ns`` for ns-precision timing."""

    label: str = "block"
    elapsed_ns: int = field(default=0, init=False)
    _start: int = field(default=0, init=False)

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter_ns()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.elapsed_ns = time.perf_counter_ns() - self._start

    @property
    def seconds(self) -> float:
        return self.elapsed_ns / 1e9


@dataclass
class MemoryProfiler(AbstractContextManager["MemoryProfiler"]):
    """Context manager wrapping ``tracemalloc`` for peak-memory measurement."""

    label: str = "block"
    peak_bytes: int = field(default=0, init=False)
    current_bytes: int = field(default=0, init=False)

    def __enter__(self) -> "MemoryProfiler":
        tracemalloc.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.current_bytes, self.peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / (1024 ** 2)
