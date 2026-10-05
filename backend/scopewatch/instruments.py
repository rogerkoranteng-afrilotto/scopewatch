"""Instruments in the field, and the scale they give us for free.

Two channels, and they do different jobs.

**The shaft channel (classical, always on).** Laparoscopic instruments are steel:
low chroma and bright against tissue that is neither. They enter from outside the
projected circle, so a real shaft touches the edge of the field. And they are long
and straight, so the medial axis of the mask is a line whose distance-transform
values are half the shaft width. That last property is the useful one, because a
laparoscopic shaft is manufactured at a known outer diameter - 5 mm for the common
sizes - which makes every instrument in view a calibration target. This is how
Scopewatch gets millimetres per pixel out of a monocular video with no marker, no
chessboard and nothing added to the theatre.

**The DNN channel (YOLOX-tiny in `cv2.dnn`, Apache-2.0).** Run on a decimated
schedule and reported honestly: the official YOLOX-tiny weights are trained on
COCO, and COCO has no surgical instrument classes. It cannot name a Maryland
dissector. What it does contribute is a licence-clean, ONNX-only, OpenCV-5 DNN
path that flags generic elongated foreign objects in the field and gives the
technical report a measured `cv2.dnn` latency on the new engine. Where the DNN and
the shaft channel disagree, the shaft channel wins and the disagreement is logged.
We say this plainly rather than implying a surgical detector we do not have; the
weights for one exist only behind dataset registration forms (see README).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from visioncore import stage, stroke_width_profile

from .config import (
    INSTRUMENT_BORDER_MARGIN_PX,
    INSTRUMENT_MIN_AREA_PX,
    INSTRUMENT_MIN_ELONGATION,
    INSTRUMENT_S_MAX,
    INSTRUMENT_V_MIN,
    SHAFT_DIAMETERS_MM,
    SHAFT_TOLERANCE_MM,
)

# The largest relative width spread that is treated as measurement uncertainty
# rather than as evidence that the component is not a single shaft.
MAX_WIDTH_SPREAD = 0.15

# Every pixel constant in this module was set on 960-pixel frames. Real clips arrive
# at 320 x 240 as often as not, and an area floor of 900 pixels there is a shaft
# that fills a sixth of the frame: on four 320 x 240 TEP clips and the WSES ulcer
# clip, visible shafts were thrown away as speckle and the whole clip was refused
# for want of a scale. Lengths scale with the frame's long side, areas with its
# square.
REFERENCE_SIDE_PX = 960.0


def resolution_factor(shape: tuple[int, ...]) -> float:
    return max(shape[:2]) / REFERENCE_SIDE_PX


# The edge-profile width. Profiles perpendicular to the shaft axis, across its middle
# third; the shaft is the run of low saturation between two tissue shoulders.
EDGE_PROFILES = 9
EDGE_MIN_PROFILES = 5
EDGE_MAX_SPREAD = 0.2  # interquartile range over median, across the profiles
EDGE_MIN_CONTRAST = 25.0  # saturation levels between shaft core and tissue

# The span of the frame a laparoscope can plausibly see at working distance, used
# only to reject shaft widths that cannot be a shaft. A wide-angle laparoscope
# (roughly 70 to 85 degrees across) held 2 to 12 cm from tissue sees on the order of
# 3 to 18 cm; the bounds are set wider than that on purpose, because this is a
# sanity check, not a measurement.
FIELD_MM_MIN = 20.0

# How far inside the rim an entry is measured, as a share of the frame's long side,
# and how elongated that piece must be to be read as a shaft rather than a blob.
ENTRY_BAND_FRACTION = 0.22
ENTRY_MIN_ELONGATION = 2.0
FIELD_MM_MAX = 220.0


@dataclass(frozen=True)
class Shaft:
    """One instrument shaft: where it is, how wide it is, how sure we are."""

    label: int
    bbox: tuple[int, int, int, int]
    area_px: int
    elongation: float
    width_px: float
    width_p95_px: float
    width_iqr_px: float
    merged: bool
    touches_border: bool
    tip: tuple[int, int]
    angle_deg: float
    # Edge-to-edge widths measured near where the steel enters the field, one per
    # entry: (width px, profiles that found both edges, IQR over median).
    entry_widths: tuple[tuple[float, int, float], ...] = ()

    @property
    def scale_widths_px(self) -> list[float]:
        """The widths a scale may be taken from. Empty when there are none.

        Only edge-to-edge widths qualify, measured on enough profiles with a small
        enough spread. See `edge_width` for why the mask's medial-axis width does not.
        """
        return [
            w for (w, n, spread) in self.entry_widths
            if n >= EDGE_MIN_PROFILES and spread <= EDGE_MAX_SPREAD and w > 2.0
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "area_px": self.area_px,
            "elongation": round(self.elongation, 2),
            "width_px": round(self.width_px, 2),
            "width_p95_px": round(self.width_p95_px, 2),
            "width_iqr_px": round(self.width_iqr_px, 2),
            "merged": self.merged,
            "touches_border": self.touches_border,
            "tip": list(self.tip),
            "angle_deg": round(self.angle_deg, 1),
            "entry_widths_px": [
                {"width_px": round(w, 2), "profiles": n, "spread": round(sp, 3)}
                for (w, n, sp) in self.entry_widths
            ],
        }


@dataclass
class InstrumentReading:
    """What one frame said about the instruments and the scale."""

    shafts: list[Shaft] = field(default_factory=list)
    entries: int = 0
    mm_per_px: float | None = None
    mm_per_px_sigma: float = 0.0
    assumed_shaft_mm: float | None = None
    scale_source: str = "none"
    dnn_objects: list[dict[str, Any]] = field(default_factory=list)
    dnn_ran: bool = False

    @property
    def count(self) -> int:
        """How many instruments are in the field.

        Counted at the edge of the projected circle, not in the middle. Two
        instruments working on the same structure touch, and a component count in
        the middle of the field then says "one". The number of separate places the
        steel crosses into the field is stable under that.
        """
        return max(self.entries, len(self.shafts))

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "entries": self.entries,
            "shafts": [s.to_dict() for s in self.shafts],
            "mm_per_px": None if self.mm_per_px is None else round(self.mm_per_px, 5),
            "mm_per_px_sigma": round(self.mm_per_px_sigma, 5),
            "assumed_shaft_mm": self.assumed_shaft_mm,
            "scale_source": self.scale_source,
            "dnn_ran": self.dnn_ran,
            "dnn_objects": list(self.dnn_objects),
        }


# ---------------------------------------------------------------------------
# The metallic mask
# ---------------------------------------------------------------------------


def metallic_mask(image: np.ndarray, field_mask: np.ndarray | None = None) -> np.ndarray:
    """Low-saturation, high-value pixels inside the lit field: steel, not tissue."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((sat <= INSTRUMENT_S_MAX) & (val >= INSTRUMENT_V_MIN)).astype(np.uint8)
    if field_mask is not None:
        mask = cv2.bitwise_and(mask, field_mask.astype(np.uint8))
    k = max(3, round(7 * resolution_factor(image.shape)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def field_boundary(field_mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    """A one-pixel ring around the *lit* region, not around the image rectangle.

    This matters more than it sounds. A laparoscope projects a circle, so an
    instrument entering through a port crosses the edge of that circle several
    hundred pixels from the edge of the frame. Testing against the image rectangle
    finds nothing and the scale silently disappears - which is exactly the bug this
    function was written to fix.
    """
    height, width = shape
    if field_mask is None:
        ring = np.zeros((height, width), np.uint8)
        m = INSTRUMENT_BORDER_MARGIN_PX
        ring[:m, :] = 1
        ring[-m:, :] = 1
        ring[:, :m] = 1
        ring[:, -m:] = 1
        return ring
    mask = field_mask.astype(np.uint8)
    margin = max(2, round(INSTRUMENT_BORDER_MARGIN_PX * resolution_factor(shape)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2)
    return cv2.subtract(mask, cv2.erode(mask, kernel))


def _rect_elongation(component: np.ndarray) -> tuple[float, float]:
    """Aspect ratio and minor side of the minimum-area rectangle, in pixels.

    The bounding box is useless here: a shaft at 45 degrees has a nearly square
    bounding box and would be thrown away as a blob. The rotated rectangle is the
    shape the object actually has.
    """
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0
    biggest = max(contours, key=cv2.contourArea)
    if len(biggest) < 5:
        return 0.0, 0.0
    (_cx, _cy), (rw, rh), _angle = cv2.minAreaRect(biggest)
    long_side, short_side = max(rw, rh), min(rw, rh)
    if short_side <= 0:
        return 0.0, 0.0
    return long_side / short_side, short_side


def _tip_and_angle(
    component: np.ndarray,
    bbox: tuple[int, int, int, int],
    field_dist: np.ndarray | None = None,
) -> tuple[tuple[int, int], float]:
    """The working end of the shaft, and the shaft's angle in the frame.

    PCA on the component's pixels gives the axis. Of the two ends, the tip is the
    one further inside the lit field - measured on a distance transform of the
    field mask, not on distance to the image rectangle, because the scope's circle
    is where an instrument actually enters.
    """
    ys, xs = np.nonzero(component)
    if xs.size == 0:
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2), 0.0
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    centred = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axis = vt[0]
    proj = centred @ axis
    candidates = [pts[int(np.argmin(proj))], pts[int(np.argmax(proj))]]
    height, width = component.shape

    def inwardness(p: np.ndarray) -> float:
        col, row = round(p[0]), round(p[1])
        if field_dist is not None:
            return float(field_dist[min(max(row, 0), height - 1), min(max(col, 0), width - 1)])
        return float(min(col, row, width - 1 - col, height - 1 - row))

    tip = max(candidates, key=inwardness)
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
    return (round(tip[0]), round(tip[1])), angle


def edge_width(
    saturation: np.ndarray, component: np.ndarray, *, profiles: int = EDGE_PROFILES
) -> tuple[float | None, int, float | None]:
    """Edge-to-edge shaft width from saturation profiles across the shaft axis.

    Returns (median width px, profiles that found both edges, IQR over median).

    This replaced the medial-axis width of the steel mask as the scale reference,
    and real footage is why. The mask is "low saturation and bright", and a wet steel
    cylinder under a scope's light is bright only along the stripe facing the light;
    the shaded half is grey or dark and fails the brightness test. So the mask covers
    about half the shaft and its medial axis is half the width. Measured on nine
    shafts in seven real frames the edge-to-edge width was a median 2.2 times the
    mask width (1.1 to 2.4), which is exactly the factor by which the implied field
    of view came out too wide. Saturation does not have that problem: the whole
    cylinder is achromatic, lit side and shaded side, and the tissue on either side
    of it is not.
    """
    ys, xs = np.nonzero(component)
    if xs.size < 32:
        return None, 0, None
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    centre = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    axis = vt[0]
    normal = np.array([-axis[1], axis[0]], np.float32)
    along = (pts - centre) @ axis
    lo, hi = np.percentile(along, [33.0, 67.0])
    h, w = saturation.shape[:2]
    reach = max(40.0, float(max(h, w)) * 0.12)
    steps = np.arange(-reach, reach + 0.5, 0.5, dtype=np.float32)
    mid = steps.size // 2
    core_half = max(2, round(3 * resolution_factor(saturation.shape) * 2))
    widths: list[float] = []
    for u in np.linspace(lo, hi, profiles):
        origin = centre + float(u) * axis
        coords = origin[None, :] + steps[:, None] * normal[None, :]
        inside = (
            (coords[:, 0] >= 0) & (coords[:, 0] <= w - 1)
            & (coords[:, 1] >= 0) & (coords[:, 1] <= h - 1)
        )
        if inside.sum() < 0.6 * steps.size:
            continue
        prof = cv2.remap(
            saturation, coords[:, 0].reshape(-1, 1), coords[:, 1].reshape(-1, 1),
            cv2.INTER_LINEAR,
        ).ravel()
        prof = np.where(inside, prof, np.nan)
        core = np.nanmin(prof[mid - core_half: mid + core_half + 1])
        tissue = np.nanpercentile(prof, 85.0)
        if not np.isfinite(core) or tissue - core < EDGE_MIN_CONTRAST:
            continue
        threshold = (core + tissue) / 2.0
        filled = np.nan_to_num(prof, nan=255.0)
        below = filled < threshold
        if not below[mid]:
            continue
        a = mid
        while a > 0 and below[a - 1]:
            a -= 1
        b = mid
        while b < below.size - 1 and below[b + 1]:
            b += 1
        if a == 0 or b == below.size - 1 or not inside[a - 1] or not inside[b + 1]:
            continue  # an edge ran off the frame; this profile cannot see both sides
        # Sub-sample edges: where the profile crosses the threshold, by linear
        # interpolation. Counting whole samples adds half a sample to every width.
        left = (a - 1) + (filled[a - 1] - threshold) / max(1e-6, filled[a - 1] - filled[a])
        right = b + (threshold - filled[b]) / max(1e-6, filled[b + 1] - filled[b])
        widths.append(float(right - left) * 0.5)
    if not widths:
        return None, 0, None
    arr = np.asarray(widths)
    median = float(np.median(arr))
    q1, q3 = np.percentile(arr, [25.0, 75.0])
    return median, int(arr.size), float((q3 - q1) / median) if median > 0 else None


def _entry_widths(
    saturation: np.ndarray,
    component: np.ndarray,
    ring: np.ndarray,
    field_dist: np.ndarray,
    min_area_px: int,
) -> tuple[tuple[float, int, float], ...]:
    """Edge widths of each place this steel component enters the field.

    Measured on the part of the component within a band inside the rim, not on the
    whole component. Two instruments working on one structure merge near their
    tips, and a merged blob has no single axis to measure across; near the rim, where
    each comes through its own port, they are still apart.
    """
    band_px = ENTRY_BAND_FRACTION * float(max(component.shape[:2]))
    near = (component.astype(bool) & (field_dist < band_px)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(near, 8)
    out: list[tuple[float, int, float]] = []
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < max(24, min_area_px // 4):
            continue
        piece = (labels == i).astype(np.uint8)
        if not cv2.bitwise_and(piece, ring).any():
            continue
        elongation, _minor = _rect_elongation(piece)
        if elongation < ENTRY_MIN_ELONGATION:
            continue
        width, n, spread = edge_width(saturation, piece)
        if width is not None and spread is not None:
            out.append((width, n, spread))
    return tuple(out)


def find_shafts(
    image: np.ndarray,
    field_mask: np.ndarray | None = None,
    *,
    min_area_px: int = INSTRUMENT_MIN_AREA_PX,
    min_elongation: float = INSTRUMENT_MIN_ELONGATION,
) -> tuple[list[Shaft], np.ndarray]:
    """Find instrument shafts and measure each one's width on its medial axis."""
    with stage("instruments:shafts"):
        mask = metallic_mask(image, field_mask)
        ring = field_boundary(field_mask, mask.shape)
        factor = resolution_factor(image.shape)
        min_area_px = max(60, round(min_area_px * factor * factor))
        saturation = cv2.GaussianBlur(
            cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32), (3, 3), 0
        )
        lit = (field_mask.astype(np.uint8) if field_mask is not None
               else np.ones(mask.shape, np.uint8))
        field_dist = cv2.distanceTransform(lit, cv2.DIST_L2, 5)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        shafts: list[Shaft] = []
        kept = np.zeros(mask.shape, np.uint8)
        for label in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[label])
            if area < min_area_px:
                continue
            component = (labels == label).astype(np.uint8)
            elongation, _minor = _rect_elongation(component)
            on_border = bool(cv2.bitwise_and(component, ring).any())
            # Two instruments working on the same structure touch, and the union of
            # two crossed shafts is not elongated. Rejecting it loses both, and with
            # them the scale. So a large border-crossing blob is kept and flagged
            # `merged`: its median width is still the shaft width (most of the
            # medial axis runs along a shaft), and the evaluation reports the two
            # cases separately.
            merged = elongation < min_elongation
            if merged and not (on_border and area >= 3 * min_area_px):
                continue
            # trim_fraction drops the flared jaws at the working end and the
            # partial cross-section at the port, leaving the parallel shaft.
            profile = stroke_width_profile(component * 255, min_samples=12, trim_fraction=0.2)
            if not profile.ok or profile.p50_px is None:
                continue
            spread = float((profile.p90_px or profile.p50_px) - profile.p50_px)
            tip, angle = _tip_and_angle(component, (x, y, w, h), field_dist)
            entry_widths = (
                _entry_widths(saturation, component, ring, field_dist, min_area_px)
                if on_border else ()
            )
            shafts.append(
                Shaft(
                    label=label,
                    bbox=(x, y, w, h),
                    area_px=area,
                    elongation=elongation,
                    width_px=float(profile.p50_px),
                    width_p95_px=float(profile.p95_px or profile.p50_px),
                    width_iqr_px=abs(spread),
                    merged=merged,
                    touches_border=on_border,
                    tip=tip,
                    angle_deg=angle,
                    entry_widths=entry_widths,
                )
            )
            kept[labels == label] = 255
        shafts.sort(key=lambda s: -s.area_px)
        return shafts, kept


def count_entries(
    mask: np.ndarray,
    ring: np.ndarray,
    *,
    min_px: int = 24,
    min_component_px: int = INSTRUMENT_MIN_AREA_PX,
) -> int:
    """Distinct places where steel crosses into the lit field.

    The crossing has to belong to something substantial. A specular highlight sitting
    on the rim of the projected circle is low-saturation and bright, exactly like
    steel, and counting it puts a phantom instrument in an empty field - which then
    puts the phase machine into `exposure` for a case that has not started.
    """
    factor = resolution_factor(mask.shape)
    min_component_px = max(60, round(min_component_px * factor * factor))
    min_px = max(6, round(min_px * factor))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    substantial = np.zeros(mask.shape, np.uint8)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= min_component_px:
            substantial[labels == i] = 1
    crossings = cv2.bitwise_and(substantial, ring)
    n, _l, cstats, _c = cv2.connectedComponentsWithStats(crossings, 8)
    return sum(1 for i in range(1, n) if cstats[i, cv2.CC_STAT_AREA] >= min_px)


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------


def scale_from_shafts(
    shafts: list[Shaft],
    *,
    assumed_mm: float = 5.0,
    match_catalogue: bool = True,
    frame_long_side_px: int | None = None,
) -> tuple[float | None, float, float | None]:
    """Millimetres per pixel from a border-touching shaft's edge-to-edge width.

    Returns (mm_per_px, sigma, assumed_shaft_mm). `None` when nothing in the frame
    can carry a scale, which is a refusal the caller must surface, not a zero.

    Only single (not merged) shafts that touch the edge of the field and whose edge
    profile found both edges consistently are used. A shaft that does not reach the
    edge of the projected circle is more likely a highlight on a clip or tissue.

    What this scale is, and is not: the millimetres a pixel spans *at the depth of
    that shaft*. The blood is on tissue at some other depth, and on real footage two
    instruments in the same frame differ in apparent width by more than the
    manufacturing tolerance ever could. So the per-frame sigma here is only the
    measurement's own spread; whether the scale can be used at all is decided over
    the whole case by `pipeline.scale_gate`, which looks at how much it wanders.
    """
    widths = sorted(
        w for s in shafts if s.touches_border for w in s.scale_widths_px
    )
    if frame_long_side_px:
        # A shaft of the assumed diameter this narrow would put more than
        # FIELD_MM_MAX across the frame, and one this wide less than FIELD_MM_MIN.
        # Neither is a laparoscope at working distance; both are something else
        # (a specular streak on tissue, a smear on the lens) measured as a shaft.
        lo = assumed_mm * frame_long_side_px / FIELD_MM_MAX
        hi = assumed_mm * frame_long_side_px / FIELD_MM_MIN
        widths = [w for w in widths if lo <= w <= hi]
    if not widths:
        return None, 0.0, None
    diameter_mm = assumed_mm
    width = float(np.median(widths))
    if match_catalogue and len(widths) >= 2 and widths[-1] > 1.7 * widths[0]:
        # Two instruments of the same catalogue size should measure the same width.
        # If the widest is close to double the narrowest it may be a 10 mm device,
        # and the narrow one is the better reference for the assumed size.
        width = widths[0]
        diameter_mm = min(SHAFT_DIAMETERS_MM, key=lambda d: abs(d - assumed_mm))
    mm_per_px = diameter_mm / width
    rel_tolerance = SHAFT_TOLERANCE_MM / diameter_mm
    spreads = [
        sp for s in shafts if s.touches_border for (w, n, sp) in s.entry_widths
        if w in widths
    ]
    rel_measurement = min(float(np.median(spreads)) if spreads else 0.0, MAX_WIDTH_SPREAD)
    # Never tighter than a sample-and-a-half at the edges.
    rel_measurement = max(rel_measurement, 0.75 / width)
    sigma = mm_per_px * float(np.hypot(rel_tolerance, rel_measurement))
    return mm_per_px, sigma, diameter_mm


def frame_widths(shafts: list[Shaft]) -> list[float]:
    """Every qualified edge width in the frame, for the wide-device comparison."""
    return sorted(w for s in shafts if s.touches_border for w in s.scale_widths_px)


# ---------------------------------------------------------------------------
# The DNN channel
# ---------------------------------------------------------------------------


def dnn_objects(detector: Any, image: np.ndarray, *, score: float = 0.25) -> list[dict[str, Any]]:
    """Run YOLOX-tiny and report what it saw, without pretending it saw instruments."""
    detections = detector.detect(image)
    return [
        {
            "bbox": [round(v, 1) for v in d.bbox],
            "score": round(float(d.score), 3),
            "class_name": d.class_name,
            "note": "COCO class; not a surgical instrument taxonomy",
        }
        for d in detections
        if d.score >= score
    ]


def find_shafts_and_entries(
    image: np.ndarray, field_mask: np.ndarray | None = None
) -> tuple[list[Shaft], int, np.ndarray]:
    shafts, kept = find_shafts(image, field_mask)
    mask = metallic_mask(image, field_mask)
    ring = field_boundary(field_mask, mask.shape)
    return shafts, count_entries(mask, ring), kept


def read_frame(
    image: np.ndarray,
    field_mask: np.ndarray | None = None,
    *,
    assumed_shaft_mm: float = 5.0,
    mm_per_px_override: float | None = None,
    detector: Any | None = None,
    dnn_score: float = 0.25,
) -> tuple[InstrumentReading, np.ndarray]:
    """Everything one frame says about instruments and scale."""
    shafts, entries, mask = find_shafts_and_entries(image, field_mask)
    if mm_per_px_override is not None and mm_per_px_override > 0:
        mm_per_px, sigma, assumed = mm_per_px_override, 0.0, None
        source = "operator"
    else:
        mm_per_px, sigma, assumed = scale_from_shafts(
            shafts, assumed_mm=assumed_shaft_mm, frame_long_side_px=max(image.shape[:2])
        )
        source = "instrument_shaft" if mm_per_px else "none"

    reading = InstrumentReading(
        shafts=shafts,
        entries=entries,
        mm_per_px=mm_per_px,
        mm_per_px_sigma=sigma,
        assumed_shaft_mm=assumed,
        scale_source=source,
    )
    if detector is not None:
        reading.dnn_objects = dnn_objects(detector, image, score=dnn_score)
        reading.dnn_ran = True
    return reading, mask
