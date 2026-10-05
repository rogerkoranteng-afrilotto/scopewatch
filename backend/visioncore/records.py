"""Run records: the evaluation evidence every product serialises.

A `RunRecord` is the single artefact a product writes per analysis. It carries
per-stage timings, the numbers the product produced, any refusals, and pointers
to the frames behind each result. The service returns it as JSON, the UI renders
it, and the technical report quotes it.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from .version import Environment, environment

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class Stage:
    """One timed stage of a pipeline."""

    name: str
    ms: float
    calls: int = 1

    @property
    def ms_per_call(self) -> float:
        return self.ms / self.calls if self.calls else 0.0

    def merged(self, other: Stage) -> Stage:
        return Stage(self.name, self.ms + other.ms, self.calls + other.calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ms": round(self.ms, 3),
            "calls": self.calls,
            "ms_per_call": round(self.ms_per_call, 3),
        }


@dataclass(frozen=True)
class Evidence:
    """A frame (or overlay, or chart) that a judge can look at to check a result."""

    label: str
    kind: str = "frame"  # frame | overlay | mask | chart | crop
    uri: str | None = None  # local path, s3:// or a service route
    frame_index: int | None = None
    timestamp_ms: float | None = None
    caption: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "kind": self.kind,
            "uri": self.uri,
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "caption": self.caption,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class Refusal:
    """A deliberate 'we will not answer' with a machine-readable reason.

    Refusals are first-class output, not errors. A measurement that declines to
    produce a number when the geometry is bad is the difference between an honest
    tool and one that reports 283 mm for a 0.75 mm crack.
    """

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": dict(self.details)}


@dataclass
class RunRecord:
    """Everything one analysis produced. Serialise this; do not invent another shape."""

    product: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_version: str = SCHEMA_VERSION
    input: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    stages: list[Stage] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    refusals: list[Refusal] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    env: Environment = field(default_factory=environment)

    # ---- mutation helpers -------------------------------------------------
    def add_stage(self, name: str, ms: float, calls: int = 1) -> Stage:
        for i, existing in enumerate(self.stages):
            if existing.name == name:
                merged = existing.merged(Stage(name, ms, calls))
                self.stages[i] = merged
                return merged
        stage = Stage(name, ms, calls)
        self.stages.append(stage)
        return stage

    def add_evidence(self, evidence: Evidence, **overrides: Any) -> Evidence:
        item = replace(evidence, **overrides) if overrides else evidence
        self.evidence.append(item)
        return item

    def refuse(self, code: str, message: str, **details: Any) -> Refusal:
        refusal = Refusal(code=code, message=message, details=details)
        self.refusals.append(refusal)
        return refusal

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    # ---- derived ----------------------------------------------------------
    @property
    def refused(self) -> bool:
        return bool(self.refusals)

    @property
    def total_ms(self) -> float:
        return sum(s.ms for s in self.stages)

    def stage(self, name: str) -> Stage | None:
        return next((s for s in self.stages if s.name == name), None)

    # ---- serialisation ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "product": self.product,
            "created_at": self.created_at,
            "input": dict(self.input),
            "params": dict(self.params),
            "timings": {
                "total_ms": round(self.total_ms, 3),
                "stages": [s.to_dict() for s in self.stages],
            },
            "metrics": dict(self.metrics),
            "results": list(self.results),
            "evidence": [e.to_dict() for e in self.evidence],
            "refusals": [r.to_dict() for r in self.refusals],
            "refused": self.refused,
            "warnings": list(self.warnings),
            "env": self.env.to_dict(),
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_fallback)


def _fallback(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def now_ms() -> float:
    return time.perf_counter() * 1000.0
