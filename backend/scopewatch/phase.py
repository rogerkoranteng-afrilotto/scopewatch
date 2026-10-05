"""Operative phase, inferred from what is in the field and what it has been doing.

The only thing the checkpoint needs to know is whether the team is approaching the
irreversible step - in a laparoscopic cholecystectomy, applying clips to what the
surgeon believes is the cystic duct and artery. Everything in this module exists to
answer that one question early enough to be worth asking.

It is a transparent rule-based temporal model, not a learned classifier, and that
is a deliberate choice rather than a shortcut. A learned phase model would need the
Cholec80 or CholecT50 annotations, which sit behind a registration form and a
CC BY-NC-SA licence (see README). More importantly, a checkpoint that stops a
surgeon has to be explainable in one sentence at the time it fires - "two
instruments, sustained dissection for ninety seconds, and a ten-millimetre device
just entered the field" is a sentence. A softmax is not.

What this buys in honesty it costs in validated accuracy, and the evaluation says
so: the confusion matrix in docs/evaluation.md is computed on scripted synthetic
sequences where the phase is known by construction, and it is not evidence about
real operative video. That work is named as the next step, not claimed as done.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import PHASE_MIN_DWELL_S, PHASES

# A wide device entering a field that has been dissecting is the strongest single
# cue this module has that clips are about to be applied: a 10 mm clip applier
# against 5 mm working instruments is twice the shaft width.
#
# What real footage showed about this cue, and why it is now built the way it is.
# The first version compared the 95th-percentile mask width of any shaft in the frame
# against the running median of every shaft's 50th-percentile width. Those are
# different statistics, and a single ordinary shaft's p95 sits well above the median
# of p50s: on three real clips the cue was true on 70, 72 and 89 per cent of
# measurable frames *before* the dwell had armed it. So the checkpoint fired on the
# first frame after 20 s of "dissection", on five clips between 26.4 and 27.8 s. The
# timer was the trigger and the picture was not. The mask widths were also half-shaft
# stripes (see `instruments.edge_width`), and on the Barroso hernia clip the "wide
# device" was pale peritoneum that passed the steel colour test.
#
# Now: the cue compares edge-to-edge widths, the same statistic on both sides: either
# two separate instruments in the same frame, or one instrument against the running
# median of the edge widths this case has already seen. It must hold for
# WIDE_PERSIST_S of frames that show instruments before it counts. The dwell only
# arms it; the dwell can never set it. Even so, two 5 mm instruments at different
# distances from the lens differ in apparent width by more than 1.7x, so on real
# video this cue cannot tell a clip applier from a near grasper. That is why the
# automatic checkpoint is off by default (`PipelineParams.auto_checkpoint`) and the
# interface says it is experimental.
WIDE_DEVICE_RATIO = 1.7
WIDE_PERSIST_S = 1.5
# Edge widths the case must have seen before "wider than usual" means anything.
WIDE_HISTORY_MIN = 20

# How long sustained two-instrument work must run before the approach to the
# irreversible step is plausible. Below this we are still exposing. This arms the
# wide-device cue; on its own it triggers nothing.
DISSECTION_DWELL_S = 20.0

# The share of the steel mask that changes between frames. Above this the
# instruments are working; below it they are being held. The units changed when the
# activity signal moved from tip matching to a mask difference; see
# `pipeline.instrument_activity` for why. Set from the scripted sequences, where
# holding still renders as a fixed shaft and working renders as a swept one.
ACTIVE_TIP_MOTION_PX = 0.15

# How long the approach to the irreversible step lasts before we call it division.
# This is the window the checkpoint has to be useful in: after a clip applier enters
# a field that has been dissecting, and before the clip goes on. Ten seconds is
# short, and deliberately so - a checkpoint that fires two minutes early is a
# checkpoint that gets dismissed out of habit.
CRITICAL_APPROACH_S = 10.0


@dataclass
class PhaseFeatures:
    """The numbers the phase rules read. All of them are already measured."""

    timestamp_ms: float
    instrument_count: int
    widest_px: float  # the widest qualified edge width in this frame
    narrowest_px: float  # the narrowest, from a separate entry in the same frame
    blood_fraction: float
    tip_motion_px: float
    measurable: bool = True
    widths_in_frame: int = 0
    usual_width_px: float = 0.0  # running median of earlier edge widths in this case
    usual_width_samples: int = 0

    @property
    def wide_device(self) -> bool:
        if (
            self.widths_in_frame >= 2
            and self.narrowest_px > 0
            and self.widest_px >= WIDE_DEVICE_RATIO * self.narrowest_px
        ):
            return True
        if self.widths_in_frame >= 1 and self.usual_width_samples >= WIDE_HISTORY_MIN:
            return self.widest_px >= WIDE_DEVICE_RATIO * self.usual_width_px
        return False


@dataclass
class PhaseSpan:
    """One contiguous run of a phase."""

    phase: str
    start_ms: float
    end_ms: float
    frames: int = 0

    @property
    def duration_ms(self) -> float:
        return max(0.0, self.end_ms - self.start_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "start_ms": round(self.start_ms, 1),
            "end_ms": round(self.end_ms, 1),
            "duration_ms": round(self.duration_ms, 1),
            "frames": self.frames,
        }


@dataclass
class PhaseTrack:
    """Per-frame phase labels plus the spans they collapse into."""

    labels: list[str] = field(default_factory=list)
    raw_labels: list[str] = field(default_factory=list)
    spans: list[PhaseSpan] = field(default_factory=list)
    wide_cue: dict[str, Any] = field(default_factory=dict)

    def at(self, index: int) -> str:
        if not self.labels:
            return "preparation"
        return self.labels[min(max(index, 0), len(self.labels) - 1)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "spans": [s.to_dict() for s in self.spans],
            "order": list(PHASES),
            "wide_cue": dict(self.wide_cue),
            "validated_on": "scripted synthetic sequences only; not on real operative video",
        }


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------


def classify(
    features: PhaseFeatures, *, dissection_s: float, seen_wide: bool, wide_s: float
) -> str:
    """The instantaneous label for one frame, before any temporal smoothing.

    Three pieces of history reach this function and nothing else does: how long
    two-instrument dissection has run, whether a wide device has arrived in a field
    that had been dissecting, and how long ago it arrived. Everything else in the
    decision is about this frame alone, which is what makes the label explainable
    at the moment it fires.
    """
    if not features.measurable:
        return "unmeasurable"

    n = features.instrument_count

    if seen_wide:
        # The clip applier is in. The first stretch after it arrives is the approach
        # to the irreversible step, whatever the tips are doing - lining a clip up
        # involves holding still, so tip motion cannot distinguish "about to clip"
        # from "clipping". Time since arrival can.
        if wide_s < CRITICAL_APPROACH_S:
            return "critical_approach"
        if n == 0:
            return "extraction"
        return "division" if features.tip_motion_px < ACTIVE_TIP_MOTION_PX else "extraction"

    if n == 0:
        return "preparation"
    if n == 1:
        return "exposure"
    if n >= 2:
        return "dissection" if features.tip_motion_px >= ACTIVE_TIP_MOTION_PX else "exposure"
    return "exposure"


class PhaseRunner:
    """The state machine, one frame at a time.

    Incremental because the pipeline needs the current phase *during* the pass -
    the agent loop reacts to it as frames arrive - and re-fitting the whole track
    on every frame would make the cost quadratic in clip length. On a 40 minute
    case at 5 frames per second that is the difference between a few milliseconds
    and a few minutes.
    """

    def __init__(self, *, min_dwell_s: float = PHASE_MIN_DWELL_S) -> None:
        self.min_dwell_s = min_dwell_s
        self.track = PhaseTrack()
        self.dissection_s = 0.0
        self.seen_wide = False
        self.wide_s = 0.0
        self.wide_run_s = 0.0
        self.wide_cue_frames = 0
        self.wide_cue_frames_unarmed = 0
        self.measurable_frames_unarmed = 0
        self.accepted = "preparation"
        self._pending: str | None = None
        self._pending_since = 0.0
        self._previous_ms: float | None = None
        self._times: list[float] = []

    def push(self, f: PhaseFeatures) -> str:
        """Feed one frame's features. Returns the accepted phase after this frame."""
        if self._previous_ms is None:
            self._previous_ms = f.timestamp_ms
        dt = max(0.0, (f.timestamp_ms - self._previous_ms) / 1000.0)
        self._previous_ms = f.timestamp_ms
        armed = self.dissection_s >= DISSECTION_DWELL_S
        cue = bool(f.measurable and f.wide_device)
        if f.measurable and not armed:
            self.measurable_frames_unarmed += 1
            self.wide_cue_frames_unarmed += int(cue)
        self.wide_cue_frames += int(cue)
        # The cue must persist in the picture; a frame without it resets the run.
        # Unmeasurable frames neither extend nor break it.
        if cue:
            self.wide_run_s += dt
        elif f.measurable and f.widths_in_frame:
            self.wide_run_s = 0.0  # instruments measured, and none of them wide
        # A wide device only counts once the field has actually been dissecting. A
        # 10 mm port trocar in view during access is not a clip applier. The dwell
        # arms the cue; only the cue, held for WIDE_PERSIST_S, sets it.
        if armed and cue and self.wide_run_s >= WIDE_PERSIST_S:
            self.seen_wide = True
        if self.seen_wide:
            self.wide_s += dt

        raw = classify(
            f,
            dissection_s=self.dissection_s,
            seen_wide=self.seen_wide,
            wide_s=self.wide_s,
        )
        self.track.raw_labels.append(raw)
        if raw == "unmeasurable":
            pass  # a fogged lens never changes the phase
        elif raw == self.accepted:
            self._pending = None
        elif raw == self._pending:
            if (f.timestamp_ms - self._pending_since) / 1000.0 >= self.min_dwell_s:
                self.accepted = raw
                self._pending = None
        else:
            self._pending = raw
            self._pending_since = f.timestamp_ms

        # Dwell accumulates on the *accepted* phase, not the raw label. Tip motion
        # dips below the threshold on individual frames while a surgeon is plainly
        # still dissecting, so counting raw labels halves the measured dwell and the
        # wide-device cue never arms. The hysteresis exists precisely to smooth that
        # out, so the dwell should read what it produced.
        if self.accepted == "dissection":
            self.dissection_s += dt

        self.track.labels.append(self.accepted)
        self._times.append(f.timestamp_ms)
        return self.accepted

    def finish(self) -> PhaseTrack:
        self.track.spans = spans_from(self.track.labels, self._times)
        self.track.wide_cue = {
            "frames_with_cue": self.wide_cue_frames,
            "frames_with_cue_before_armed": self.wide_cue_frames_unarmed,
            "measurable_frames_before_armed": self.measurable_frames_unarmed,
            "seen_wide": self.seen_wide,
        }
        return self.track


def infer(
    features: list[PhaseFeatures],
    *,
    min_dwell_s: float = PHASE_MIN_DWELL_S,
) -> PhaseTrack:
    """Label every frame, with hysteresis so the ribbon does not flicker.

    Hysteresis rule: a new label must hold for `min_dwell_s` before it is accepted.
    Until then the previous accepted label stands. Unmeasurable frames inherit the
    last accepted label, so a fogged lens never changes the phase.
    """
    runner = PhaseRunner(min_dwell_s=min_dwell_s)
    for f in features:
        runner.push(f)
    return runner.finish()


def spans_from(labels: list[str], times_ms: list[float]) -> list[PhaseSpan]:
    """Collapse per-frame labels into contiguous spans for the phase ribbon."""
    spans: list[PhaseSpan] = []
    for i, label in enumerate(labels):
        t = times_ms[i]
        if spans and spans[-1].phase == label:
            spans[-1].end_ms = t
            spans[-1].frames += 1
        else:
            spans.append(PhaseSpan(phase=label, start_ms=t, end_ms=t, frames=1))
    return spans


def first_entry(track: PhaseTrack, phase: str, times_ms: list[float]) -> float | None:
    """When the track first entered a phase, in milliseconds."""
    for i, label in enumerate(track.labels):
        if label == phase:
            return times_ms[i] if i < len(times_ms) else None
    return None
