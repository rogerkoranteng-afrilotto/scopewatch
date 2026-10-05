"""Can this frame be measured at all?

This module exists because the alternative is a tool that produces a confident
number from a fogged lens. Every gate here is a reason to say "I cannot measure
this", and each one is a named refusal that reaches the UI, the run record and the
technical report. A refusal is the tool working, not the tool failing.

The four states a laparoscope actually gets into, and what each looks like in
pixels:

* **Out of focus** - the scope is too close to the tissue, or the optic is wet.
  The Laplacian carries no variance.
* **Fogged** - condensation on the cold lens entering a warm abdomen, or
  electrocautery smoke filling the cavity. Haze lifts the per-pixel minimum across
  colour channels (the dark channel prior) and flattens global contrast.
* **Occluded** - a swab, a retracted lobe or an instrument sits across the optic.
  Most of the circular field is one flat region.
* **Clipped** - the automatic gain has blown the highlights or crushed the blacks,
  so hue and chroma are not recoverable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from visioncore import stage

from .config import REFUSAL_CODES, Thresholds


@dataclass(frozen=True)
class FrameQuality:
    """The four statistics, the verdict, and why."""

    focus: float
    dark_channel: float
    contrast: float
    occlusion: float
    clipped: float
    usable: bool
    reasons: tuple[str, ...] = ()
    field_mask_fraction: float = 1.0
    cool_share: float = 0.0
    white_share: float = 0.0
    rest_saturation: float = 0.0

    @property
    def reason(self) -> str | None:
        return self.reasons[0] if self.reasons else None

    @property
    def message(self) -> str:
        if not self.reasons:
            return "Measurable."
        return " ".join(REFUSAL_CODES.get(r, r) for r in self.reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "focus": round(self.focus, 2),
            "dark_channel": round(self.dark_channel, 2),
            "contrast": round(self.contrast, 2),
            "occlusion": round(self.occlusion, 4),
            "clipped": round(self.clipped, 4),
            "usable": self.usable,
            "reasons": list(self.reasons),
            "message": self.message,
            "field_mask_fraction": round(self.field_mask_fraction, 4),
            "cool_share": round(self.cool_share, 4),
            "white_share": round(self.white_share, 4),
            "rest_saturation": round(self.rest_saturation, 1),
        }


@dataclass
class QualityLedger:
    """How many frames each gate rejected across a whole clip."""

    total: int = 0
    usable: int = 0
    by_reason: dict[str, int] = field(default_factory=dict)

    def add(self, q: FrameQuality) -> None:
        self.total += 1
        if q.usable:
            self.usable += 1
        for reason in q.reasons:
            self.by_reason[reason] = self.by_reason.get(reason, 0) + 1

    @property
    def usable_fraction(self) -> float:
        return self.usable / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": self.total,
            "usable": self.usable,
            "usable_fraction": round(self.usable_fraction, 4),
            "rejected_by": dict(sorted(self.by_reason.items(), key=lambda kv: -kv[1])),
        }


# ---------------------------------------------------------------------------
# The laparoscope aperture
# ---------------------------------------------------------------------------


def erode_disc(mask: np.ndarray, radius: int) -> np.ndarray:
    """Erode a binary mask by a disc, via the distance transform.

    Equivalent to `cv2.erode` with an elliptical kernel of that radius, and several
    times cheaper at the radii used here (15 to 20 px), where the kernel erosion was
    the single most expensive call in the frame.
    """
    binary = (mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    return (dist > radius).astype(np.uint8)


def field_mask(image: np.ndarray, *, min_fraction: float = 0.25) -> np.ndarray:
    """The circular image the scope actually projects, excluding the black surround.

    A laparoscope projects a circle into a rectangular sensor. The corners are not
    dark tissue, they are no data at all, and counting them as "not blood" quietly
    deflates every area fraction by the same 20%. So we find the lit region once and
    every fraction in this product is taken against it.

    Falls back to the whole frame when the lit region is implausibly small, which is
    what happens on an already-cropped clip.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 16 is well below any lit tissue and above sensor noise in the black surround.
    mask = (grey > 16).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return np.ones(grey.shape, np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    lit = (labels == largest).astype(np.uint8)
    if lit.sum() < min_fraction * grey.size:
        return np.ones(grey.shape, np.uint8)
    return lit


# ---------------------------------------------------------------------------
# The four statistics
# ---------------------------------------------------------------------------


def focus_score(grey: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Variance of the Laplacian inside the lit field."""
    lap = cv2.Laplacian(grey, cv2.CV_32F, ksize=3)
    if mask is None:
        return float(lap.var())
    sel = lap[mask.astype(bool)]
    return float(sel.var()) if sel.size else 0.0


def dark_channel(image: np.ndarray, mask: np.ndarray | None = None, patch: int = 15) -> float:
    """Mean dark-channel value: the haze indicator from He et al.'s dark channel prior.

    A clear frame of an abdominal cavity has genuine shadow somewhere in every
    neighbourhood, so the per-pixel minimum over channels, eroded over a patch, is
    low. Fog and cautery smoke are additive white light, which lifts that floor.
    """
    mins = image.min(axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    dark = cv2.erode(mins, kernel)
    if mask is None:
        return float(dark.mean())
    sel = dark[mask.astype(bool)]
    return float(sel.mean()) if sel.size else 255.0


def contrast_score(grey: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Standard deviation of luminance inside the lit field."""
    if mask is None:
        return float(grey.std())
    sel = grey[mask.astype(bool)]
    return float(sel.std()) if sel.size else 0.0


def occlusion_fraction(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Fraction of the lit field covered by something that is not the operative field.

    The first version of this tested for a large flat low-gradient region, and it
    rejected 39% of a perfectly measurable clip - because a settled pool of blood is
    exactly that: large, flat and low-gradient. Calling the thing we are trying to
    measure an occlusion is the worst failure a gate like this can have.

    What an actual occluder has that a pool does not is the absence of colour. Gauze,
    a swab, a glove and a lens cap are all achromatic; blood and tissue are not. So
    an occluder is flat *and* desaturated, or simply dark enough to be no image at
    all. Both conditions are counted, and the largest connected region wins.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]

    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    achromatic_flat = (mag < 12.0) & (sat < 55) & (val > 90)
    blacked_out = val < 28
    covered = (achromatic_flat | blacked_out).astype(np.uint8)
    if mask is not None:
        covered = cv2.bitwise_and(covered, mask.astype(np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    covered = cv2.morphologyEx(covered, cv2.MORPH_OPEN, kernel)

    count, _labels, stats, _ = cv2.connectedComponentsWithStats(covered, connectivity=8)
    if count <= 1:
        return 0.0
    largest = float(stats[1:, cv2.CC_STAT_AREA].max())
    denom = float(mask.sum()) if mask is not None else float(grey.size)
    return largest / denom if denom else 0.0


def clipped_fraction(grey: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Share of the lit field at either end of the 8-bit range."""
    sel = grey[mask.astype(bool)] if mask is not None else grey.reshape(-1)
    if not sel.size:
        return 1.0
    return float(((sel <= 2) | (sel >= 253)).mean())


def domain_cues(image: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float, float]:
    """(cool-hue share, large near-white share, mean saturation of the rest).

    The three numbers the out-of-domain gate reads; config.py says what each one
    separated on real footage and what it did not.
    """
    lit = np.ones(image.shape[:2], bool) if mask is None else mask.astype(bool)
    total = int(lit.sum())
    if total == 0:
        return 0.0, 0.0, 0.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    cool = (hue >= 35) & (hue <= 130) & (sat >= 40) & (val >= 40) & lit
    white = ((sat < 40) & (val > 225) & lit).astype(np.uint8)
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    min_blob = max(20, int(0.002 * total))
    big = sum(int(a) for a in stats[1:, cv2.CC_STAT_AREA] if a >= min_blob) if count > 1 else 0
    rest = lit & ~white.astype(bool)
    rest_sat = float(sat[rest].mean()) if rest.any() else 0.0
    return float(cool.sum()) / total, big / total, rest_sat


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


def assess(
    image: np.ndarray,
    *,
    thresholds: Thresholds | None = None,
    mask: np.ndarray | None = None,
) -> FrameQuality:
    """Run all four gates and decide whether this frame may produce a number."""
    t = thresholds or Thresholds()
    with stage("quality"):
        lit = field_mask(image) if mask is None else mask
        # The edge of the projected circle is the hardest edge in a laparoscopic
        # frame: a step from tissue to nothing. Left in, it dominates the variance
        # of the Laplacian and the focus score barely moves as the image goes soft -
        # measured at 1337 on a frame blurred with a 9 pixel sigma, against a
        # threshold of 55. So the focus and contrast statistics are taken on an
        # eroded core, away from the rim. The blood and occlusion statistics use
        # the full lit region, because a pool at the edge of the field is still a
        # pool.
        core = erode_disc(lit, 15)
        if core.sum() < 0.2 * max(1, lit.sum()):
            core = lit.astype(np.uint8)
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        focus = focus_score(grey, core)
        dc = dark_channel(image, lit)
        contrast = contrast_score(grey, core)
        occ = occlusion_fraction(image, lit)
        clip = clipped_fraction(grey, lit)

        cool, white, rest_sat = domain_cues(image, lit)

        reasons: list[str] = []
        # Domain first: a frame of drapes or gloved hands is not a laparoscopic frame
        # that happens to be hard to measure, and no other refusal describes it.
        if cool >= t.domain_cool_max or (
            white >= t.domain_white_min and rest_sat >= t.domain_rest_saturation_min
        ):
            reasons.append("OUT_OF_DOMAIN")
        # Fog is checked before focus: a fogged frame is also out of focus, and the
        # useful thing to tell a scrub nurse is "wipe the lens", not "it is blurry".
        if dc > t.fog_dark_channel_max and contrast < t.fog_contrast_min:
            reasons.append("LENS_FOGGED")
        elif focus < t.focus_min:
            reasons.append("OUT_OF_FOCUS")
        if occ > t.occlusion_max:
            reasons.append("OCCLUDED")
        if clip > t.exposure_clip_max:
            reasons.append("EXPOSURE_CLIPPED")

        return FrameQuality(
            focus=focus,
            dark_channel=dc,
            contrast=contrast,
            occlusion=occ,
            clipped=clip,
            usable=not reasons,
            reasons=tuple(reasons),
            field_mask_fraction=float(lit.mean()),
            cool_share=cool,
            white_share=white,
            rest_saturation=rest_sat,
        )
