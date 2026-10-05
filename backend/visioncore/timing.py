"""Per-stage millisecond timing that flows into the active RunRecord.

Two ways in, both thread-safe and both usable without a record present:

    with recording(record):
        with stage("decode"):
            ...

    @timed("detect")
    def detect(frame): ...

`stage`/`timed` always return the elapsed milliseconds; attaching to a record is
optional so library code can be timed in unit tests with no ceremony.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar

from .records import RunRecord

_active: ContextVar[RunRecord | None] = ContextVar("visioncore_active_record", default=None)

F = TypeVar("F", bound=Callable[..., Any])


def active_record() -> RunRecord | None:
    return _active.get()


@contextmanager
def recording(record: RunRecord) -> Iterator[RunRecord]:
    """Make `record` the stage sink for this context (and this thread/task only)."""
    token = _active.set(record)
    try:
        yield record
    finally:
        _active.reset(token)


class StageTimer:
    """Context manager returning elapsed ms in `.ms` and recording it if a record is active."""

    __slots__ = ("_start", "ms", "name", "record")

    def __init__(self, name: str, record: RunRecord | None = None) -> None:
        self.name = name
        self.record = record
        self._start = 0.0
        self.ms = 0.0

    def __enter__(self) -> StageTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.ms = (time.perf_counter() - self._start) * 1000.0
        sink = self.record or _active.get()
        if sink is not None:
            sink.add_stage(self.name, self.ms)


def stage(name: str, record: RunRecord | None = None) -> StageTimer:
    return StageTimer(name, record)


def timed(name: str | None = None) -> Callable[[F], F]:
    """Decorator recording wall-clock ms for each call under `name` (default: func name)."""

    def decorate(func: F) -> F:
        stage_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = (time.perf_counter() - start) * 1000.0
                wrapper.last_ms = elapsed  # type: ignore[attr-defined]
                sink = _active.get()
                if sink is not None:
                    sink.add_stage(stage_name, elapsed)

        wrapper.last_ms = 0.0  # type: ignore[attr-defined]
        wrapper.stage_name = stage_name  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorate
