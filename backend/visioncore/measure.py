"""Stroke-width measurement done the way that does not produce 380x errors.

A failure we hit early: fill an open contour with
`drawContours(..., -1)` and take `max(distanceTransform)`. On an open polyline
the fill covers the hull, the distance transform peaks in the middle of that
hull, and a 0.75 mm crack reports as 283 mm. No exception, no warning.

The fix, implemented here:

1. Never fill contours to measure a thin feature. Run the distance transform on
   the **binary mask itself**.
2. Sample it only on the medial axis (local maxima of the distance field), which
   for a thin stroke is its centre line.
3. Report a **distribution** (p50, p90, p95, max, n) rather than a single max, so
   one bad pixel cannot become the headline number.
4. Refuse when the evidence is too thin to support a number.

Width convention: for a stroke of w pixels, the distance transform peaks at
about (w + 1) / 2 on the centre line, so width_px = 2 * d - 1. That is exact for
odd widths and within 1 px for even ones; `precision_px` records the floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .calibration import Calibration
from .records import Refusal

WIDTH_PRECISION_PX = 1.0


@dataclass(frozen=True)
class WidthProfile:
    """A distribution of stroke widths, in pixels and (if calibrated) millimetres."""

    ok: bool
    samples: int
    p50_px: float | None = None
    p90_px: float | None = None
    p95_px: float | None = None
    max_px: float | None = None
    mean_px: float | None = None
    length_px: float | None = None
    px_per_mm: float | None = None
    refusal: Refusal | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def _mm(self, px: float | None) -> float | None:
        if px is None or not self.px_per_mm:
            return None
        return px / self.px_per_mm

    @property
    def p50_mm(self) -> float | None:
        return self._mm(self.p50_px)

    @property
    def p95_mm(self) -> float | None:
        return self._mm(self.p95_px)

    @property
    def max_mm(self) -> float | None:
        return self._mm(self.max_px)

    @property
    def precision_mm(self) -> float | None:
        return self._mm(WIDTH_PRECISION_PX)

    def to_dict(self) -> dict[str, Any]:
        def r(v: float | None, n: int = 3) -> float | None:
            return None if v is None else round(v, n)

        return {
            "ok": self.ok,
            "samples": self.samples,
            "px": {
                "p50": r(self.p50_px),
                "p90": r(self.p90_px),
                "p95": r(self.p95_px),
                "max": r(self.max_px),
                "mean": r(self.mean_px),
                "length": r(self.length_px, 1),
                "precision": WIDTH_PRECISION_PX,
            },
            "mm": {
                "p50": r(self.p50_mm),
                "p95": r(self.p95_mm),
                "max": r(self.max_mm),
                "precision": r(self.precision_mm, 4),
            },
            "px_per_mm": r(self.px_per_mm, 4),
            "refusal": self.refusal.to_dict() if self.refusal else None,
            "diagnostics": dict(self.diagnostics),
        }


def _refuse(code: str, message: str, **details: Any) -> WidthProfile:
    return WidthProfile(
        ok=False, samples=0, refusal=Refusal(code, message, details), diagnostics=dict(details)
    )


def medial_axis(mask: np.ndarray, *, dist: np.ndarray | None = None) -> np.ndarray:
    """Boolean mask of local maxima of the distance transform (the ridge / centre line).

    A pixel is on the ridge when its distance value equals the 3x3 dilation of the
    distance field, i.e. nothing in its neighbourhood is further from the boundary.
    """
    binary = _as_binary(mask)
    if dist is None:
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dilated = cv2.dilate(dist, np.ones((3, 3), np.uint8))
    return (dist >= dilated - 1e-6) & (dist > 0)


def _as_binary(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def stroke_width_profile(
    mask: np.ndarray,
    *,
    calibration: Calibration | None = None,
    min_samples: int = 8,
    trim_fraction: float = 0.0,
) -> WidthProfile:
    """Width distribution of a thin feature, sampled on its medial axis.

    `mask`         non-zero where the feature is. Do NOT pass a filled contour.
    `calibration`  supplies px_per_mm; a refused calibration yields px only.
    `min_samples`  refuse below this many ridge pixels - too little evidence.
    `trim_fraction` drop this fraction from each end of the ridge sample, for
                   features whose ends flare (junctions, terminations).
    """
    binary = _as_binary(mask)
    if not np.any(binary):
        return _refuse("EMPTY_MASK", "mask is empty; nothing to measure")

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    ridge = medial_axis(binary, dist=dist)
    values = dist[ridge]
    if values.size == 0:
        return _refuse("NO_RIDGE", "no medial axis found in the mask")

    widths = np.sort(2.0 * values.astype(np.float64) - 1.0)
    widths = widths[widths > 0]
    if trim_fraction > 0 and widths.size > 4:
        cut = int(widths.size * trim_fraction)
        if cut > 0:
            widths = widths[cut:-cut] if widths.size - 2 * cut >= 1 else widths
    if widths.size < min_samples:
        return _refuse(
            "TOO_FEW_SAMPLES",
            f"only {int(widths.size)} medial-axis samples (need {min_samples}); "
            "the feature is too short or the mask too noisy to measure",
            samples=int(widths.size),
            min_samples=min_samples,
        )

    px_per_mm = calibration.px_per_mm if (calibration and calibration.ok) else None
    return WidthProfile(
        ok=True,
        samples=int(widths.size),
        p50_px=float(np.percentile(widths, 50)),
        p90_px=float(np.percentile(widths, 90)),
        p95_px=float(np.percentile(widths, 95)),
        max_px=float(widths.max()),
        mean_px=float(widths.mean()),
        length_px=float(np.count_nonzero(ridge)),
        px_per_mm=px_per_mm,
        diagnostics={
            "calibration": calibration.to_dict() if calibration else None,
            "mask_pixels": int(np.count_nonzero(binary)),
            "note": "widths sampled on the medial axis, never on a filled contour",
        },
    )


def elongated_components(
    mask: np.ndarray,
    *,
    min_area_px: int = 30,
    min_elongation: float = 3.0,
    connectivity: int = 8,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Keep only long, thin components. Returns (filtered mask, per-component stats).

    Elongation is the bounding-box aspect ratio, which is cheap and good enough to
    throw away blobs before the expensive width pass.
    """
    binary = _as_binary(mask)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity)
    keep = np.zeros_like(binary)
    kept: list[dict[str, Any]] = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < min_area_px:
            continue
        elongation = max(w, h) / max(1.0, min(w, h))
        if elongation < min_elongation:
            continue
        keep[labels == label] = 255
        kept.append(
            {
                "label": label,
                "bbox": [x, y, w, h],
                "area_px": area,
                "elongation": round(elongation, 3),
                "centroid": [round(float(centroids[label][0]), 2),
                             round(float(centroids[label][1]), 2)],
            }
        )
    return keep, kept


def filled_polyline_width_max(
    points: np.ndarray, shape: tuple[int, int]
) -> float:
    """The WRONG way, kept executable so a test can assert how wrong it is.

    This is the recorded bug verbatim: take the crack's path (an open polyline of
    points, which is what a contour or a thinned centre line gives you), fill it
    with `drawContours(..., thickness=-1)`, and read the distance transform's
    maximum. `drawContours` closes the polyline into a polygon, so the distance
    transform measures the *enclosed area*, not the stroke. On a 4K frame that is
    ~187 px where the true stroke is 5 px.

    Never call this from a product. It exists to keep the regression test honest.
    """
    pts = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
    filled = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(filled, [pts], -1, 255, thickness=-1)
    dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    return float(2.0 * dist.max() - 1.0)
