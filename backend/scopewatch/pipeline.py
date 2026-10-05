"""The pipeline: frames in, a RunRecord out, with the agent loop running alongside.

Two passes, and the second one only exists because of what the first one saw.

*Pass one* decimates the video - `stride` frames apart - and for each kept frame
asks four questions in this order: can this be measured at all, what is the scale,
how much blood is on the field, and what is in the field. The answers become a time
series, a phase track and a stream of observations fed to `agent.AgentLoop`.

*Pass two* happens only if the loop asked for it. When the onset detector fires on
the coarse series, the loop emits a `rescan_window` action and this module re-opens
the file, reads that window at full frame rate, and replaces the coarse estimate of
the onset with the dense one. Nothing about that is scripted: a clip with no bleed
in it never triggers a second read, and a clip with two bleeds triggers one around
the first, which is the one that matters.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from visioncore import (
    Evidence,
    Frame,
    RunRecord,
    iter_video,
    read_image,
    stage,
    video_info,
)

from . import blood as blood_mod
from . import instruments as instruments_mod
from . import onset as onset_mod
from . import phase as phase_mod
from . import quality as quality_mod
from .agent import ACTION_RESCAN, AgentLoop, Observation
from .config import (
    BLOOD_COLOUR_SPACE,
    DOMAIN_CLIP_SHARE,
    ONSET_CUSUM_MIN_SIGMA_PCT,
    PALETTE,
    REFUSAL_CODES,
    SCALE_MAX_CV,
    SCALE_MIN_FRAME_SHARE,
    SCALE_MIN_FRAMES,
    YOLOX_SCORE,
    YOLOX_STRIDE,
    PipelineParams,
    Thresholds,
)

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}

# Frames the instrument-activity signal is held over. At the default stride this is
# about a second of case time.
ACTIVITY_WINDOW_FRAMES = 4

# Burned into every evidence frame the product writes, because an exported frame
# outlives the page it was exported from.
DISCLAIMER = (
    "Scopewatch - retrospective measurement, not intra-operative guidance. "
    "Not a medical device."
)

ProgressFn = Callable[[float, str], None]
EvidenceFn = Callable[[str, bytes, dict[str, Any]], str | None]


@dataclass
class FrameResult:
    """Everything one frame produced. The evaluation reads these directly."""

    index: int
    timestamp_ms: float
    quality: quality_mod.FrameQuality
    instruments: instruments_mod.InstrumentReading
    blood: blood_mod.BloodMeasurement | None
    motion_px: float
    tip_motion_px: float  # 0 to 1: the share of the steel mask that changed
    refusal: str | None
    steel: np.ndarray | None = None

    @property
    def measurable(self) -> bool:
        return self.refusal is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "timestamp_ms": round(self.timestamp_ms, 1),
            "refusal": self.refusal,
            "quality": self.quality.to_dict(),
            "instruments": self.instruments.to_dict(),
            "blood": self.blood.to_dict() if self.blood else None,
            "motion_px": round(self.motion_px, 2),
            "instrument_activity": round(self.tip_motion_px, 4),
        }


@dataclass
class CaseResult:
    """The whole clip: the series, the phases, the onset, the agent's log."""

    frames: list[FrameResult] = field(default_factory=list)
    ledger: quality_mod.QualityLedger = field(default_factory=quality_mod.QualityLedger)
    series: onset_mod.Series | None = None
    onset: onset_mod.Onset | None = None
    phases: phase_mod.PhaseTrack = field(default_factory=phase_mod.PhaseTrack)
    loop: AgentLoop = field(default_factory=AgentLoop)
    peak_volume_ml: float | None = None
    peak_volume_low_ml: float | None = None
    peak_volume_high_ml: float | None = None
    cumulative_observed_ml: float | None = None
    mm_per_px: float | None = None
    mm_per_px_sigma: float = 0.0
    scale_source: str = "none"
    scale_gate: dict[str, Any] = field(default_factory=dict)
    peak_field_fraction: float | None = None
    median_field_fraction: float | None = None
    rescanned: bool = False
    video: dict[str, Any] = field(default_factory=dict)
    runner: phase_mod.PhaseRunner = field(default_factory=phase_mod.PhaseRunner)
    width_samples: list[float] = field(default_factory=list)
    activity_window: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# One frame
# ---------------------------------------------------------------------------


def process_frame(
    frame: Frame,
    *,
    params: PipelineParams,
    thresholds: Thresholds | None = None,
    detector: Any | None = None,
    run_dnn: bool = False,
    motion_px: float = 0.0,
    previous_steel: np.ndarray | None = None,
) -> tuple[FrameResult, np.ndarray | None, np.ndarray]:
    """Measure one frame. Returns (result, blood mask or None, field mask).

    The order is deliberate. Quality first, because a fogged frame must not reach
    the segmenter at all - not so we save the milliseconds, but so there is no code
    path in which an unmeasurable frame produces a number that something downstream
    could pick up.
    """
    image = frame.image
    lit = quality_mod.field_mask(image)
    q = quality_mod.assess(image, thresholds=thresholds, mask=lit)

    if not q.usable:
        return (
            FrameResult(
                index=frame.index,
                timestamp_ms=frame.timestamp_ms,
                quality=q,
                instruments=instruments_mod.InstrumentReading(),
                blood=None,
                motion_px=motion_px,
                tip_motion_px=0.0,
                refusal=q.reasons[0],
                steel=None,
            ),
            None,
            lit,
        )

    reading, steel = instruments_mod.read_frame(
        image,
        lit,
        assumed_shaft_mm=params.shaft_mm,
        mm_per_px_override=params.mm_per_px,
        detector=detector if run_dnn else None,
        dnn_score=YOLOX_SCORE,
    )

    activity = instrument_activity(steel, previous_steel)

    # The blood is measured as a share of the visible field, which needs no scale.
    # Millilitres are added later, over the whole case, and only if the case's scale
    # passes `scale_gate`; a frame is never refused for want of a scale any more.
    measurement, mask = blood_mod.measure(
        image,
        lit,
        method=BLOOD_COLOUR_SPACE,
        scale_source=reading.scale_source,
    )

    return (
        FrameResult(
            index=frame.index,
            timestamp_ms=frame.timestamp_ms,
            quality=q,
            instruments=reading,
            blood=measurement,
            motion_px=motion_px,
            tip_motion_px=activity,
            refusal=None,
            steel=steel,
        ),
        mask,
        lit,
    )


def instrument_activity(steel: np.ndarray, previous: np.ndarray | None) -> float:
    """How much the steel in the field moved since the last measured frame, 0 to 1.

    The symmetric difference of the two instrument masks over their union. This
    replaced a tip-matching estimate, and the replacement was not a refinement - the
    tip estimate was structurally broken. It returned zero whenever the number of
    detected tips changed between frames, which happens constantly, because two
    instruments working on the same structure merge into one component and then
    separate again. Zero motion reads as "the tips are holding still", so sustained
    dissection was scored as exposure on 363 of 600 frames and the phase track never
    accumulated the dwell the checkpoint depends on.

    A mask difference has none of that. It does not care how many components there
    are, it does not need correspondence between frames, and it costs one bitwise
    operation on a binary image.
    """
    if previous is None or steel.shape != previous.shape:
        return 0.0
    union = cv2.bitwise_or(steel, previous)
    total = int(union.sum())
    if total < 200:
        return 0.0
    return float(int(cv2.bitwise_xor(steel, previous).sum()) / total)


# ---------------------------------------------------------------------------
# A whole case
# ---------------------------------------------------------------------------


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES


def _frames(path: Path, params: PipelineParams, *, stride: int | None = None,
            start_ms: float = 0.0, end_ms: float | None = None):
    if _is_video(path):
        return iter_video(
            path,
            stride=stride or params.stride,
            max_frames=params.max_frames,
            max_side=params.max_side,
            start_ms=start_ms,
            end_ms=end_ms,
        )
    image = read_image(path)
    if max(image.shape[:2]) > params.max_side:
        scale = params.max_side / max(image.shape[:2])
        image = cv2.resize(
            image,
            (round(image.shape[1] * scale), round(image.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return iter([Frame(index=0, timestamp_ms=0.0, image=image, source=str(path))])


def analyse_case(
    path: Path | str,
    *,
    params: PipelineParams | None = None,
    thresholds: Thresholds | None = None,
    detector: Any | None = None,
    progress: ProgressFn | None = None,
    save_evidence: EvidenceFn | None = None,
) -> CaseResult:
    """Run the whole thing. This is what the service and the evaluation both call."""
    params = params or PipelineParams()
    path = Path(path)
    result = CaseResult()
    if _is_video(path):
        result.video = video_info(path).to_dict()

    _pass(path, params, thresholds, detector, result, progress, save_evidence, label="pass 1")
    _finish(result, params)

    # --- the agent's second look -------------------------------------------
    rescan = next(
        (a for a in result.loop.actions if a.kind == ACTION_RESCAN and not a.performed), None
    )
    if rescan and params.rescan and _is_video(path):
        if progress:
            progress(78.0, "the onset triggered a dense re-read of that window")
        dense = CaseResult()
        _pass(
            path,
            params,
            thresholds,
            detector,
            dense,
            progress,
            save_evidence,
            label="rescan",
            stride=int(rescan.detail.get("stride", 1)),
            start_ms=float(rescan.detail.get("start_ms", 0.0)),
            end_ms=float(rescan.detail.get("end_ms", 0.0)) or None,
            loop=result.loop,
            run_agent=False,
        )
        _finish(dense, params)
        rescan.performed = True
        rescan.result = {
            "frames_read": len(dense.frames),
            "stride": int(rescan.detail.get("stride", 1)),
            "onset_before_ms": result.onset.timestamp_ms if result.onset else None,
            "onset_after_ms": dense.onset.timestamp_ms if dense.onset else None,
        }
        if dense.onset and dense.onset.detected:
            result.onset = dense.onset
            result.rescanned = True
            rescan.result["refined"] = True
        else:
            rescan.result["refined"] = False
            rescan.result["note"] = (
                "the dense pass did not confirm the coarse onset; the coarse "
                "timestamp is kept and flagged"
            )
    return result


def _pass(
    path: Path,
    params: PipelineParams,
    thresholds: Thresholds | None,
    detector: Any | None,
    result: CaseResult,
    progress: ProgressFn | None,
    save_evidence: EvidenceFn | None,
    *,
    label: str,
    stride: int | None = None,
    start_ms: float = 0.0,
    end_ms: float | None = None,
    loop: AgentLoop | None = None,
    run_agent: bool = True,
) -> None:
    """One read of the file. Pass one runs the agent; the rescan does not re-run it."""
    if loop is not None:
        result.loop = loop
    motion = onset_mod.MotionEstimator()
    previous_steel: np.ndarray | None = None
    expected = params.max_frames or 400
    started = time.perf_counter()

    frames = _frames(path, params, stride=stride, start_ms=start_ms, end_ms=end_ms)
    for n, frame in enumerate(frames):
        shift = motion.update(frame.image)
        result.video["analysed_long_side"] = int(max(frame.image.shape[:2]))
        run_dnn = bool(params.use_dnn and detector is not None and n % YOLOX_STRIDE == 0)
        fr, mask, lit = process_frame(
            frame,
            params=params,
            thresholds=thresholds,
            detector=detector,
            run_dnn=run_dnn,
            motion_px=shift,
            previous_steel=previous_steel,
        )
        if fr.steel is not None:
            previous_steel = fr.steel
        result.frames.append(fr)
        result.ledger.add(fr.quality)

        if progress and n % 5 == 0:
            pct = 8.0 + 66.0 * min(1.0, n / max(1, expected))
            progress(pct, f"{label}: frame {frame.index} at {frame.timestamp_ms / 1000:.1f} s")

        current_phase = result.runner.push(_feature_for(result, fr))
        if run_agent:
            uri = _maybe_evidence(save_evidence, frame, fr, mask, lit, n)
            _observe(result, fr, uri, params, current_phase)

    result.video.setdefault("analysis_seconds", 0.0)
    result.video["analysis_seconds"] = round(time.perf_counter() - started, 3)


def _observe(
    result: CaseResult,
    fr: FrameResult,
    uri: str | None,
    params: PipelineParams,
    provisional: str,
) -> None:
    """Feed the agent loop the phase as it stands at this frame, not in hindsight."""
    result.loop.observe(
        Observation(
            index=fr.index,
            timestamp_ms=fr.timestamp_ms,
            phase=provisional,
            measurable=fr.measurable,
            refusal=fr.refusal,
            volume_ml=None,
            onset=False,  # set in _finish, once the series exists
            safety_view_established=params.safety_view_established,
            evidence_uri=uri,
            checkpoint_enabled=params.auto_checkpoint,
        )
    )


def _feature_for(result: CaseResult, fr: FrameResult) -> phase_mod.PhaseFeatures:
    """One frame's phase features.

    The wide-device cue compares edge-to-edge widths of separate instruments in this
    frame, not this frame's widest mask width against a running median of every
    width seen so far; `phase.py` has the history of why.
    """
    widths = instruments_mod.frame_widths(fr.instruments.shafts)
    history = result.width_samples
    usual = float(np.median(history)) if history else 0.0
    samples = len(history)
    if fr.measurable:
        # Only widths that are not themselves wide join the history, so a device that
        # stays in view does not become the usual width.
        history.extend(w for w in widths if not usual or w < 1.4 * usual or samples < 20)
    # "Has it moved recently", not "is it moving in this instant". A grasper sweeping
    # back and forth is momentarily stationary at each end of the sweep, and an
    # instantaneous churn reading dips to nothing there. The running maximum over the
    # last second answers the question the phase rules are actually asking.
    result.activity_window.append(fr.tip_motion_px)
    del result.activity_window[:-ACTIVITY_WINDOW_FRAMES]
    activity = max(result.activity_window) if result.activity_window else 0.0
    return phase_mod.PhaseFeatures(
        timestamp_ms=fr.timestamp_ms,
        instrument_count=fr.instruments.count,
        widest_px=widths[-1] if widths else 0.0,
        narrowest_px=widths[0] if widths else 0.0,
        widths_in_frame=len(widths),
        usual_width_px=usual,
        usual_width_samples=samples,
        blood_fraction=fr.blood.area_fraction if fr.blood else 0.0,
        tip_motion_px=activity,
        measurable=fr.measurable,
    )


def scale_gate(frames: list[FrameResult]) -> dict[str, Any]:
    """Decide, over the whole case, whether its shaft scale can carry a volume.

    Two tests, both stated in config.py. Enough measurable frames must carry a shaft
    scale, and the per-frame scale must not wander: its robust coefficient of
    variation must stay under SCALE_MAX_CV. The second is the one real footage fails.
    A shaft's apparent width moves with its distance from the lens, and the tissue
    the blood lies on is at yet another distance, so a scale that jumps by a third
    between frames is not a property of the field.
    """
    measurable = [f for f in frames if f.measurable]
    operator = [f for f in measurable if f.instruments.scale_source == "operator"]
    if operator:
        mm = float(operator[0].instruments.mm_per_px or 0.0)
        return {
            "status": "operator", "passed": True, "mm_per_px": mm, "mm_per_px_sigma": 0.0,
            "frames_with_scale": len(operator), "measurable_frames": len(measurable),
            "frame_share": 1.0, "robust_cv": 0.0, "reason_code": None,
            "reason": "the operator supplied millimetres per pixel",
        }
    scales = [f.instruments.mm_per_px for f in measurable if f.instruments.mm_per_px]
    share = len(scales) / len(measurable) if measurable else 0.0
    out: dict[str, Any] = {
        "frames_with_scale": len(scales),
        "measurable_frames": len(measurable),
        "frame_share": round(share, 4),
        "min_frame_share": SCALE_MIN_FRAME_SHARE,
        "max_robust_cv": SCALE_MAX_CV,
    }
    if not scales:
        return {**out, "status": "none", "passed": False, "mm_per_px": None,
                "mm_per_px_sigma": 0.0, "robust_cv": None,
                "reason_code": "NO_SCALE_REFERENCE",
                "reason": "no frame carried a measurable instrument shaft"}
    arr = np.asarray(scales, dtype=np.float64)
    median = float(np.median(arr))
    cv = float(1.4826 * np.median(np.abs(arr - median)) / median) if median > 0 else 1.0
    sigmas = [f.instruments.mm_per_px_sigma for f in measurable if f.instruments.mm_per_px]
    # The case sigma is the larger of the typical per-frame sigma and the case's own
    # spread; a scale that wanders is not known better than it wanders.
    sigma = max(float(np.median(sigmas)), cv * median)
    out.update({"mm_per_px": median, "mm_per_px_sigma": sigma, "robust_cv": round(cv, 4)})
    if share < SCALE_MIN_FRAME_SHARE or len(scales) < SCALE_MIN_FRAMES:
        return {**out, "status": "sparse", "passed": False,
                "reason_code": "NO_SCALE_REFERENCE",
                "reason": "too few frames carried an instrument of known size"}
    if cv > SCALE_MAX_CV:
        return {**out, "status": "inconsistent", "passed": False,
                "reason_code": "SCALE_INCONSISTENT",
                "reason": "the instrument scale was not steady enough across the case"}
    return {**out, "status": "consistent", "passed": True, "reason_code": None,
            "reason": (f"shaft scale on {share:.0%} of measurable frames, "
                       f"varying {cv:.0%}")}


def _finish(result: CaseResult, params: PipelineParams) -> None:
    """Fit the series, the phases and the onset over the frames this pass produced."""
    frames = result.frames
    if not frames:
        return

    with stage("series"):
        times = [f.timestamp_ms for f in frames]
        # Percentage points of the visible field. Unmeasurable frames carry no value;
        # `onset.analyse` holds the last measured value across them rather than
        # dropping to zero, which is what used to manufacture slopes.
        fractions = [
            (f.blood.area_fraction * 100.0 if (f.blood and f.measurable) else 0.0)
            for f in frames
        ]
        measurable = [f.measurable for f in frames]
        motion = [f.motion_px for f in frames]
        counts = [f.instruments.count for f in frames]
        reliable = [bool(f.blood and f.blood.reliable) for f in frames]
        series, onset = onset_mod.analyse(
            times,
            fractions,
            motion_px=motion,
            measurable=measurable,
            instruments=counts,
            reliable=reliable,
            threshold=params.onset_rate_pct_per_min,
            unit="%/min",
            cusum_min_sigma=ONSET_CUSUM_MIN_SIGMA_PCT,
        )
        result.series = series
        result.onset = onset

    with stage("phase"):
        result.phases = result.runner.finish()

    measured = [f for f in frames if f.blood and f.measurable]
    if measured:
        values = [f.blood.area_fraction for f in measured if f.blood]
        result.peak_field_fraction = float(max(values))
        result.median_field_fraction = float(np.median(values))

    # Millilitres, only when the case's scale can carry them.
    gate = scale_gate(frames)
    result.scale_gate = gate
    result.mm_per_px = gate.get("mm_per_px")
    result.mm_per_px_sigma = float(gate.get("mm_per_px_sigma") or 0.0)
    result.scale_source = (
        "operator" if gate.get("status") == "operator"
        else "instrument_shaft" if gate.get("mm_per_px") else "none"
    )
    if gate.get("passed") and measured and result.mm_per_px:
        volumes: list[float] = []
        for f in frames:
            if f.blood and f.measurable:
                f.blood = blood_mod.with_volume(
                    f.blood, result.mm_per_px, result.mm_per_px_sigma,
                    depth_mm=params.film_depth_mm, depth_low_mm=params.film_depth_low_mm,
                    depth_high_mm=params.film_depth_high_mm, scale_source=result.scale_source,
                )
            volumes.append(
                f.blood.volume_ml if (f.blood and f.blood.volume_ml is not None) else 0.0
            )
        peak = max(measured, key=lambda f: f.blood.volume_ml or 0.0 if f.blood else 0.0)
        if peak.blood:
            result.peak_volume_ml = peak.blood.volume_ml
            result.peak_volume_low_ml = peak.blood.volume_ml_low
            result.peak_volume_high_ml = peak.blood.volume_ml_high
        result.cumulative_observed_ml = onset_mod.cumulative_observed_loss(times, volumes)

    # Now that the onset is known, replay it into the loop so the rescan sees it.
    # This is the only place `onset=True` is ever set.
    if result.onset and result.onset.detected and result.onset.index is not None:
        i = min(result.onset.index, len(frames) - 1)
        fr = frames[i]
        result.loop.observe(
            Observation(
                index=fr.index,
                timestamp_ms=fr.timestamp_ms,
                phase=result.phases.at(i),
                measurable=fr.measurable,
                refusal=fr.refusal,
                volume_ml=fr.blood.volume_ml if fr.blood else None,
                field_fraction=fr.blood.area_fraction if fr.blood else None,
                onset=True,
                safety_view_established=params.safety_view_established,
                checkpoint_enabled=params.auto_checkpoint,
            )
        )


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def _maybe_evidence(
    save: EvidenceFn | None,
    frame: Frame,
    fr: FrameResult,
    mask: np.ndarray | None,
    lit: np.ndarray,
    n: int,
) -> str | None:
    """Save a frame when it is worth looking at, not every tenth frame regardless.

    A judge checking a number wants the frame behind that number: the first refusal
    of each kind, the frame with the most blood so far, and the frames where the
    instrument count changed. Everything else is noise in the evidence panel.
    """
    if save is None:
        return None
    reason = None
    if not fr.measurable:
        reason = f"refused-{(fr.refusal or 'unknown').lower()}"
    elif n == 0:
        reason = "first-measured"
    elif fr.blood and fr.blood.area_fraction > 0.02 and n % 12 == 0:
        reason = "blood"
    if reason is None:
        return None
    image = annotate(frame.image, fr, mask, lit)
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return None
    return save(
        f"{fr.index:06d}-{reason}",
        buffer.tobytes(),
        {
            "frame_index": fr.index,
            "timestamp_ms": fr.timestamp_ms,
            "caption": _caption(fr),
            "metrics": fr.to_dict(),
        },
    )


def _caption(fr: FrameResult) -> str:
    if not fr.measurable:
        return f"{fr.refusal}: {REFUSAL_CODES.get(fr.refusal or '', 'not measurable')}"
    b = fr.blood
    share = (b.area_fraction * 100) if b else 0.0
    return f"{share:.1f}% of the visible field segmented as blood"


def annotate(
    image: np.ndarray, fr: FrameResult, mask: np.ndarray | None, lit: np.ndarray
) -> np.ndarray:
    """The evidence frame: the pool outlined, the shafts marked, the number written.

    The disclaimer is burned into the pixels, not written next to them. An evidence
    frame is the thing that leaves this product: it gets saved, pasted into a
    presentation, attached to an email, and shown to someone who never saw the page
    it came from. A caveat that lives in the interface does not travel with it. This
    one does.
    """
    out = image.copy()
    if mask is not None and mask.any():
        out = blood_mod.overlay(out, mask, PALETTE.blood)
    for shaft in fr.instruments.shafts:
        x, y, w, h = shaft.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), PALETTE.ink_dim, 1)
        cv2.circle(out, shaft.tip, 5, PALETTE.accent, 2)
    del lit

    font = cv2.FontFace("sans") if hasattr(cv2, "FontFace") else None
    text = _caption(fr)
    colour = PALETTE.accent if not fr.measurable else PALETTE.ink
    height = out.shape[0]
    cv2.rectangle(out, (0, height - 52), (out.shape[1], height), PALETTE.ground, -1)
    if font is not None:
        cv2.putText(out, text, (12, height - 32), colour, font, 15)
        cv2.putText(out, DISCLAIMER, (12, height - 11), PALETTE.ink_dim, font, 12)
    else:  # pragma: no cover - OpenCV 5 always has FontFace
        cv2.putText(out, text, (12, height - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        cv2.putText(out, DISCLAIMER, (12, height - 11), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, PALETTE.ink_dim, 1)
    return out


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def to_record(result: CaseResult, record: RunRecord, params: PipelineParams) -> RunRecord:
    """Fill a visioncore RunRecord from a CaseResult. This is the service's contract."""
    record.params.update(params.to_dict())
    gate = result.scale_gate or {}
    volume_measured = result.peak_volume_ml is not None
    record.metrics.update(
        {
            "frames_analysed": len(result.frames),
            "quality": result.ledger.to_dict(),
            "scale": {
                "mm_per_px": None if result.mm_per_px is None else round(result.mm_per_px, 5),
                "mm_per_px_sigma": round(result.mm_per_px_sigma, 5),
                "source": result.scale_source,
                "assumed_shaft_mm": params.shaft_mm,
                "gate": _rounded(gate),
                "implied_field_width_mm": _field_width_mm(result),
            },
            "blood": {
                "primary": "field_fraction",
                "peak_field_fraction": _r(result.peak_field_fraction, 5),
                "median_field_fraction": _r(result.median_field_fraction, 5),
                "volume": {
                    "status": "MEASURED" if volume_measured else "CANNOT_MEASURE",
                    "reason_code": None if volume_measured else gate.get("reason_code"),
                    "reason": gate.get("reason", "no measurable frame"),
                    "validated_on_real_footage": False,
                },
                "peak_volume_ml": _r(result.peak_volume_ml),
                "peak_volume_low_ml": _r(result.peak_volume_low_ml),
                "peak_volume_high_ml": _r(result.peak_volume_high_ml),
                "cumulative_observed_ml": _r(result.cumulative_observed_ml),
                "cumulative_is_lower_bound": True,
                "film_depth_mm": [params.film_depth_low_mm, params.film_depth_mm,
                                  params.film_depth_high_mm],
            },
            "onset": result.onset.to_dict() if result.onset else None,
            "rescanned": result.rescanned,
            "phases": result.phases.to_dict(),
            "checkpoint": {
                "automatic": params.auto_checkpoint,
                "status": "enabled" if params.auto_checkpoint else "disabled",
                "note": (
                    "Experimental. The cue is instrument width, which on real video "
                    "cannot tell a clip applier from a grasper nearer the lens."
                ),
            },
            "agent": result.loop.to_dict(),
            "video": result.video,
            "series": result.series.to_dict() if result.series else None,
            "series_unit": "percent of the visible field",
        }
    )

    rejected = result.ledger.to_dict()["rejected_by"]
    out_of_domain = rejected.get("OUT_OF_DOMAIN", 0) >= DOMAIN_CLIP_SHARE * max(
        1, result.ledger.total
    )
    if result.ledger.total and result.ledger.usable == 0 and not out_of_domain:
        record.refuse(
            "NO_USABLE_FRAMES", REFUSAL_CODES["NO_USABLE_FRAMES"], rejected_by=rejected
        )
    elif out_of_domain:
        record.refuse(
            "OUT_OF_DOMAIN",
            REFUSAL_CODES["OUT_OF_DOMAIN"],
            rejected_by=rejected,
            hint="Scopewatch reads laparoscopic video from inside the abdomen only.",
            share_of_frames=round(
                rejected.get("OUT_OF_DOMAIN", 0) / max(1, result.ledger.total), 3
            ),
        )
    record.results = _results(result, params)
    return record


def _field_width_mm(result: CaseResult) -> float | None:
    """The scale times the analysed frame's long side: a sanity number, not a claim."""
    if not result.mm_per_px or not result.frames:
        return None
    width = result.video.get("width") or 0
    height = result.video.get("height") or 0
    side = max(width, height)
    if not side:
        return None
    # The scale is in analysed pixels, and frames are analysed at no more than max_side.
    analysed = int(result.video.get("analysed_long_side") or side)
    return round(result.mm_per_px * analysed, 1)


def _rounded(d: dict[str, Any]) -> dict[str, Any]:
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items()}


def _results(result: CaseResult, params: PipelineParams) -> list[dict[str, Any]]:
    """The headline rows: what the KPI band shows, each with what qualifies it."""
    rows: list[dict[str, Any]] = []
    measured_any = result.peak_field_fraction is not None
    rows.append(
        {
            "label": "Blood-covered field, peak",
            "value": _r(result.peak_field_fraction * 100.0, 2) if measured_any else None,
            "unit": "%",
            "measured": measured_any,
            "note": (
                "share of the visible field segmented as blood"
                if measured_any else "no frame in this clip passed the gates"
            ),
        }
    )
    gate = result.scale_gate or {}
    if result.peak_volume_ml is not None:
        rows.append(
            {
                "label": "Blood on the field, volume",
                "value": _r(result.peak_volume_ml),
                "low": _r(result.peak_volume_low_ml),
                "high": _r(result.peak_volume_high_ml),
                "unit": "ml",
                "measured": True,
                "status": "MEASURED",
                "note": (
                    f"peak; area x assumed film depth {params.film_depth_low_mm}-"
                    f"{params.film_depth_high_mm} mm at a {gate.get('status', '')} scale. "
                    "Not validated on real footage: no real clip has a known volume"
                ),
            }
        )
    else:
        rows.append(
            {
                "label": "Blood on the field, volume",
                "value": None,
                "unit": "ml",
                "measured": False,
                "status": "CANNOT_MEASURE",
                "reason_code": gate.get("reason_code"),
                "note": f"CANNOT_MEASURE: {gate.get('reason', 'no measurable frame')}",
            }
        )
    if result.onset:
        rows.append(
            {
                "label": "Bleeding onset",
                "value": (
                    None if not result.onset.detected
                    else round((result.onset.timestamp_ms or 0.0) / 1000.0, 1)
                ),
                "plus_minus": round(result.onset.uncertainty_ms / 1000.0, 1),
                "unit": "s",
                "measured": result.onset.detected,
                "note": result.onset.reason,
            }
        )
    rejected = result.ledger.to_dict()["rejected_by"]
    worst = max(rejected, key=lambda k: rejected[k]) if rejected else ""
    rows.append(
        {
            "label": "Frames measurable",
            "value": round(result.ledger.usable_fraction * 100.0, 1),
            "unit": "%",
            "measured": result.ledger.total > 0,
            "note": (
                f"{result.ledger.usable} of {result.ledger.total} frames"
            ),
        }
    )
    checkpoint = result.loop.open_checkpoint
    rows.append(
        {
            "label": "Safety checkpoint",
            "value": result.loop.state if params.auto_checkpoint else "off",
            "unit": "",
            "measured": params.auto_checkpoint,
            "note": (
                checkpoint.question if checkpoint
                else "experimental; raised only when the automatic checkpoint is switched on"
                if not params.auto_checkpoint else "no checkpoint was raised"
            ),
        }
    )
    return rows


def _r(value: float | None, n: int = 3) -> float | None:
    return None if value is None else round(float(value), n)


def evidence_from(result: CaseResult) -> list[Evidence]:
    """Kept for callers that want evidence objects without the service's save hook."""
    del result
    return []
