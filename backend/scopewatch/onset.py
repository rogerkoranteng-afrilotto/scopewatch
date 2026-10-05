"""The time series: smoothing, camera motion, rate of change, and bleeding onset.

The question "when did this start bleeding" is a change-point question on a noisy
series, and the two ways to get it wrong are both common.

The first is to threshold the volume itself. That answers "when was there a lot of
blood", which is minutes later than the event you wanted and is exactly the delay a
surgeon already has by eye.

The second is to threshold the raw first difference. Laparoscopic video is a
handheld camera in a moving cavity, so the frame-to-frame difference is dominated
by the scope panning across a pool that was already there. A pan produces the same
positive slope as a bleed, and a tool that cannot tell them apart will timestamp
every camera movement as a haemorrhage.

So: fit a slope by least squares over a window, in millilitres per minute, and
require a one-sided CUSUM on the same series to agree. Gate both on a global motion
estimate from phase correlation, so frames where the field moved do not contribute
a rate. The reported onset carries the window length as its uncertainty, because
that is genuinely how well a windowed estimator can localise a step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .config import (
    MEDIAN_WINDOW,
    MOTION_SUSPECT_PX,
    ONSET_CUSUM_H_SIGMA,
    ONSET_CUSUM_K_SIGMA,
    ONSET_CUSUM_MIN_SIGMA_ML,
    ONSET_RATE_ML_PER_MIN,
    ONSET_WINDOW_S,
    SMOOTHING_ALPHA,
)

# ---------------------------------------------------------------------------
# Camera motion
# ---------------------------------------------------------------------------


class MotionEstimator:
    """Global translation between consecutive kept frames, by phase correlation.

    Phase correlation on a Hanning-windowed, downscaled grey frame costs under a
    millisecond and gives a sub-pixel global shift. It is not optical flow and does
    not pretend to be: it answers "did the whole field move", which is the only
    question this pipeline needs it to answer.
    """

    def __init__(self, size: int = 256) -> None:
        self.size = size
        self._previous: np.ndarray | None = None
        self._window: np.ndarray | None = None

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        small = cv2.resize(grey, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return small.astype(np.float32)

    def update(self, image: np.ndarray) -> float:
        """Pixels of global shift since the previous call, scaled to the input frame."""
        current = self._prepare(image)
        if self._window is None:
            self._window = cv2.createHanningWindow((self.size, self.size), cv2.CV_32F)
        if self._previous is None:
            self._previous = current
            return 0.0
        (dx, dy), _response = cv2.phaseCorrelate(self._previous, current, self._window)
        self._previous = current
        scale = max(image.shape[:2]) / float(self.size)
        return float(np.hypot(dx, dy) * scale)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


def median_filter(values: list[float], window: int = MEDIAN_WINDOW) -> list[float]:
    """Running median. Rejects a single mis-segmented frame without lagging a trend."""
    if window < 2 or len(values) < 2:
        return list(values)
    half = window // 2
    out: list[float] = []
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        out.append(float(np.median(values[lo:hi])))
    return out


def ema(values: list[float], alpha: float = SMOOTHING_ALPHA) -> list[float]:
    """Exponential moving average, applied after the median."""
    out: list[float] = []
    acc: float | None = None
    for v in values:
        acc = v if acc is None else alpha * v + (1.0 - alpha) * acc
        out.append(float(acc))
    return out


def smooth(
    values: list[float], *, window: int = MEDIAN_WINDOW, alpha: float = SMOOTHING_ALPHA
) -> list[float]:
    return ema(median_filter(values, window), alpha)


# ---------------------------------------------------------------------------
# Rate
# ---------------------------------------------------------------------------


def windowed_rate(
    times_s: list[float],
    values: list[float],
    *,
    window_s: float = ONSET_WINDOW_S,
) -> list[float]:
    """Least-squares slope over a trailing window, in units per minute.

    Trailing rather than centred, because a theatre instrument may not look into
    the future. Windows with fewer than three samples or no time span report zero
    rather than a slope fitted through noise.
    """
    rates: list[float] = []
    for i, t in enumerate(times_s):
        lo = i
        while lo > 0 and t - times_s[lo - 1] <= window_s:
            lo -= 1
        xs = np.asarray(times_s[lo : i + 1], dtype=np.float64)
        ys = np.asarray(values[lo : i + 1], dtype=np.float64)
        if xs.size < 3 or (xs[-1] - xs[0]) <= 1e-6:
            rates.append(0.0)
            continue
        xs = xs - xs.mean()
        denom = float((xs * xs).sum())
        slope = float((xs * (ys - ys.mean())).sum() / denom) if denom > 0 else 0.0
        rates.append(slope * 60.0)
    return rates


def difference_sigma(values: list[float], *, floor: float = ONSET_CUSUM_MIN_SIGMA_ML) -> float:
    """Robust noise scale of the series, from the median absolute first difference.

    The median is used rather than the standard deviation precisely because the
    change we are hunting is in the tail. A standard deviation computed over a
    series that contains the bleed is inflated by the bleed, and the detector then
    needs a bigger bleed to notice - which is the wrong way round.
    """
    if len(values) < 3:
        return floor
    diffs = np.abs(np.diff(np.asarray(values, dtype=np.float64)))
    mad = float(np.median(diffs))
    return max(floor, 1.4826 * mad)


def cusum(
    values: list[float],
    *,
    k: float | None = None,
    h: float | None = None,
    min_sigma: float = ONSET_CUSUM_MIN_SIGMA_ML,
) -> tuple[list[float], int | None]:
    """One-sided upward CUSUM on the first differences. Returns (statistic, first alarm).

    Page's classic detector. `k` is the slack: increments smaller than it accumulate
    nothing, so ordinary segmentation jitter never drifts the statistic upward. `h`
    is the decision interval. Left unset, both are scaled from this series' own
    noise, because a millilitre figure that is right for one scope at one working
    distance is wrong for the next.
    """
    sigma = difference_sigma(values, floor=min_sigma)
    k = ONSET_CUSUM_K_SIGMA * sigma if k is None else k
    h = ONSET_CUSUM_H_SIGMA * sigma if h is None else h
    s = 0.0
    out: list[float] = []
    alarm: int | None = None
    previous = values[0] if values else 0.0
    for i, v in enumerate(values):
        delta = v - previous
        previous = v
        s = max(0.0, s + delta - k)
        out.append(s)
        if alarm is None and s > h:
            alarm = i
    return out, alarm


# ---------------------------------------------------------------------------
# Onset
# ---------------------------------------------------------------------------


# The gates, in the order they are checked. A candidate frame is one whose fitted
# rate reached the threshold; each gate below can still stop it, and the reason
# string counts how many candidates each one stopped. The first version of this
# module reported "peak slope X never reached the threshold" whenever no onset was
# declared, computed over frames that had passed only some of the gates. On three
# real clips the rate had reached the threshold and a different gate had stopped the
# alarm, so the message said the opposite of what had happened.
GATE_UNMEASURABLE = "unmeasurable"
GATE_MOTION = "camera_motion"
GATE_UNRELIABLE = "unreliable_measurement"
GATE_INSTRUMENTS = "instrument_count_changed"
GATE_CUSUM = "cusum_not_confirmed"
GATES: tuple[str, ...] = (
    GATE_UNMEASURABLE, GATE_MOTION, GATE_UNRELIABLE, GATE_INSTRUMENTS, GATE_CUSUM,
)
GATE_WORDS = {
    GATE_UNMEASURABLE: "the frame was refused by a quality or domain gate",
    GATE_MOTION: "the camera was moving",
    GATE_UNRELIABLE: "the blood area was below the reliable size",
    GATE_INSTRUMENTS: "an instrument entered or left inside the fit window",
    GATE_CUSUM: "the CUSUM had not confirmed a sustained step",
}


@dataclass
class Onset:
    """When the rate of change crossed the threshold, and how sure we are of when."""

    detected: bool
    index: int | None = None
    timestamp_ms: float | None = None
    uncertainty_ms: float = 0.0
    rate_per_min: float | None = None
    threshold_per_min: float = ONSET_RATE_ML_PER_MIN
    unit: str = "ml/min"
    cusum_index: int | None = None
    reason: str = ""
    peak_rate_any_per_min: float | None = None
    candidates: int = 0
    blocked_by: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def r(v: float | None, n: int = 3) -> float | None:
            return None if v is None else round(float(v), n)

        return {
            "detected": self.detected,
            "index": self.index,
            "timestamp_ms": r(self.timestamp_ms, 1),
            "uncertainty_ms": round(self.uncertainty_ms, 1),
            "rate_per_min": r(self.rate_per_min),
            "threshold_per_min": self.threshold_per_min,
            "unit": self.unit,
            "cusum_index": self.cusum_index,
            "reason": self.reason,
            "peak_rate_any_per_min": r(self.peak_rate_any_per_min),
            "candidates": self.candidates,
            "blocked_by": dict(self.blocked_by),
        }


@dataclass
class Series:
    """The whole case as numbers, which is what the trace draws and the report quotes."""

    times_ms: list[float] = field(default_factory=list)
    raw: list[float] = field(default_factory=list)
    smoothed: list[float] = field(default_factory=list)
    rate: list[float] = field(default_factory=list)
    cusum: list[float] = field(default_factory=list)
    motion_px: list[float] = field(default_factory=list)
    measurable: list[bool] = field(default_factory=list)
    instruments: list[int] = field(default_factory=list)
    reliable: list[bool] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "times_ms": [round(t, 1) for t in self.times_ms],
            "raw": [round(v, 4) for v in self.raw],
            "smoothed": [round(v, 4) for v in self.smoothed],
            "rate_per_min": [round(v, 4) for v in self.rate],
            "cusum": [round(v, 4) for v in self.cusum],
            "motion_px": [round(v, 2) for v in self.motion_px],
            "measurable": list(self.measurable),
            "instruments": list(self.instruments),
            "reliable": list(self.reliable),
        }


def analyse(
    times_ms: list[float],
    values: list[float],
    *,
    motion_px: list[float] | None = None,
    measurable: list[bool] | None = None,
    instruments: list[int] | None = None,
    reliable: list[bool] | None = None,
    window_s: float = ONSET_WINDOW_S,
    threshold: float = ONSET_RATE_ML_PER_MIN,
    motion_limit: float = MOTION_SUSPECT_PX,
    unit: str = "ml/min",
    cusum_min_sigma: float = ONSET_CUSUM_MIN_SIGMA_ML,
) -> tuple[Series, Onset]:
    """Smooth the series, fit the rate, run the CUSUM, and pick the onset.

    Every frame whose fitted rate reached the threshold is a candidate, and the first
    candidate that clears every gate is the onset. When none clears them, the reason
    names the gates that stopped the candidates and how many each stopped, so a
    reader can tell "nothing rose" from "something rose and the camera was moving".
    """
    n = len(times_ms)
    motion_px = motion_px or [0.0] * n
    measurable = measurable if measurable is not None else [True] * n
    instruments = instruments if instruments is not None else [0] * n
    reliable = reliable if reliable is not None else [True] * n
    times_s = [t / 1000.0 for t in times_ms]

    # Unmeasurable frames carry no value. Filling them with zero, which is what the
    # series used to do, turns every refusal gap into a fall and every return into a
    # rise, and the rise is a slope that has nothing to do with blood. The last
    # measured value is held instead; the gap is still marked unmeasurable.
    held: list[float] = []
    last = next((v for v, m in zip(values, measurable, strict=False) if m), 0.0)
    for v, m in zip(values, measurable, strict=False):
        if m:
            last = v
        held.append(last)

    smoothed = smooth(held)
    rate = windowed_rate(times_s, smoothed, window_s=window_s)
    stat, alarm = cusum(smoothed, min_sigma=cusum_min_sigma)

    series = Series(
        times_ms=list(times_ms),
        raw=list(values),
        smoothed=smoothed,
        rate=rate,
        cusum=stat,
        motion_px=list(motion_px),
        measurable=list(measurable),
        instruments=list(instruments),
        reliable=list(reliable),
    )

    if n < 4:
        return series, Onset(
            False, threshold_per_min=threshold, unit=unit,
            reason="too few measurable frames to fit a rate",
        )

    blocked: dict[str, int] = {}
    candidates = 0
    for i in range(n):
        if rate[i] < threshold:
            continue
        candidates += 1
        gate = _first_blocking_gate(
            i, measurable, motion_px, reliable, times_s, instruments, window_s,
            motion_limit, alarm,
        )
        if gate is not None:
            blocked[gate] = blocked.get(gate, 0) + 1
            continue
        return series, Onset(
            detected=True,
            index=i,
            timestamp_ms=times_ms[i],
            uncertainty_ms=window_s * 1000.0,
            rate_per_min=rate[i],
            threshold_per_min=threshold,
            unit=unit,
            cusum_index=alarm,
            reason=(
                f"slope {rate[i]:.2f} {unit} over a {window_s:.0f} s window, "
                f"confirmed by CUSUM at sample {alarm}"
            ),
            peak_rate_any_per_min=max(rate),
            candidates=candidates,
            blocked_by=blocked,
        )

    peak_any = max(rate) if rate else 0.0
    if candidates == 0:
        reason = (
            f"the fitted rate never reached {threshold:.2f} {unit} "
            f"(peak {peak_any:.2f} {unit} over the whole clip)"
        )
    else:
        top = max((g for g in GATES if blocked.get(g)),
                   key=lambda g: blocked[g], default=None)
        reason = (
            f"the rate crossed the threshold, but {GATE_WORDS[top]} each time"
            if top else "the rate crossed the threshold, but every crossing was gated"
        )
    return series, Onset(
        False,
        rate_per_min=peak_any,
        threshold_per_min=threshold,
        unit=unit,
        cusum_index=alarm,
        reason=reason,
        peak_rate_any_per_min=peak_any,
        candidates=candidates,
        blocked_by=blocked,
    )


def _first_blocking_gate(
    i: int,
    measurable: list[bool],
    motion_px: list[float],
    reliable: list[bool],
    times_s: list[float],
    instruments: list[int],
    window_s: float,
    motion_limit: float,
    alarm: int | None,
) -> str | None:
    """The first gate, in `GATES` order, that stops frame `i` from being the onset."""
    if not measurable[i]:
        return GATE_UNMEASURABLE
    if motion_px[i] > motion_limit:
        return GATE_MOTION  # the field moved; this slope is not about bleeding
    if not reliable[i]:
        # Below the pool size where the false-positive floor stops mattering, the
        # estimator's own jitter is the same order as the rate being looked for.
        return GATE_UNRELIABLE
    if _instruments_changed(times_s, instruments, i, window_s):
        # An instrument entering or leaving uncovers or hides part of the pool, and
        # the measured area steps with the same shape as a bleed.
        return GATE_INSTRUMENTS
    if alarm is None or i < alarm - 2:
        return GATE_CUSUM
    return None


def _instruments_changed(
    times_s: list[float], instruments: list[int], index: int, window_s: float
) -> bool:
    """True when the instrument count moved anywhere inside the rate window."""
    if not instruments:
        return False
    lo = index
    while lo > 0 and times_s[index] - times_s[lo - 1] <= window_s:
        lo -= 1
    window = instruments[lo : index + 1]
    return len(set(window)) > 1


def cumulative_observed_loss(times_ms: list[float], values: list[float]) -> float:
    """Sum of the positive increases in on-field volume across the case.

    This is a **lower bound** on blood loss and is labelled as one everywhere it is
    shown. Blood that is suctioned away between two frames leaves the field and is
    never counted again; blood that soaks into a swab is not on the field at all.
    What this number is good for is comparison - between two halves of one case,
    or between the same operation done twice - not for a transfusion decision.
    """
    smoothed = smooth(values)
    total = 0.0
    for i in range(1, len(smoothed)):
        delta = smoothed[i] - smoothed[i - 1]
        if delta > 0:
            total += delta
    del times_ms
    return float(total)
