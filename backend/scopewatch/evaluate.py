"""Evaluation against ground truth that is known by construction (synthetic scenes).

The evaluation on real laparoscopic footage is `scopewatch.realeval`; this module
folds its summary (docs/real-evaluation.json) into evaluation.json under "real" so
the service serves both. Synthetic numbers say the arithmetic is right. Only the
real ones say anything about tissue.

Run it:

    python -m scopewatch.evaluate --out docs

It writes `docs/evaluation.json` (the numbers, machine-readable, served by the app
at /api/evaluation) and the plots under `docs/plots/`. `docs/evaluation.md` quotes
what comes out of here; if the two ever disagree, this file is right.

Seven experiments, each answering one question the product's claims depend on:

1. **Colour space.** Which of five candidate segmentations recovers a blood mask
   that is known exactly? Scored with and without an inflamed-serosa distractor,
   because the distractor is what separates a real answer from a red-pixel counter.
2. **Area.** How far off is the measured area, in percent, across three decades of
   pool size?
3. **Scale.** How far off is millimetres-per-pixel, recovered from an instrument
   shaft of known width?
4. **Volume.** Does the reported interval contain the truth, and how often?
5. **Refusal.** Swept across blur, fog and occlusion: where does the tool stop
   answering, and what error would it have reported had it not stopped? The second
   half is the part that matters - a refusal is only worth having if the answer it
   withheld would have been wrong.
6. **Onset.** On scripted cases with a bleed that starts on a known frame, how many
   seconds out is the timestamp, and how often does a case with no bleed produce a
   false onset?
7. **Phase and checkpoint.** A confusion matrix over the scripted phases, and
   whether the checkpoint fired in the window where it would have been useful.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from visioncore import RunRecord, environment, recording

from . import blood as blood_mod
from . import instruments as instruments_mod
from . import quality as quality_mod
from .config import BLOOD_COLOUR_SPACE, PHASES, PipelineParams
from .pipeline import analyse_case
from .synth import CaseScript, SceneSpec, render, write_case_video

SEEDS = (3, 11, 29, 47, 61, 83, 97, 113)
POOL_SIZES = (0, 400, 1_200, 4_000, 12_000, 36_000, 72_000)
DISTRACTOR_PX = 20_000


def dice(truth: np.ndarray, predicted: np.ndarray) -> float:
    t, p = truth.astype(bool), predicted.astype(bool)
    total = int(t.sum()) + int(p.sum())
    if total == 0:
        return 1.0
    return 2.0 * float((t & p).sum()) / total


def _pct(errors: list[float]) -> dict[str, float]:
    if not errors:
        return {"n": 0}
    arr = np.asarray(errors, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 3),
        "median": round(float(np.median(arr)), 3),
        "p95": round(float(np.percentile(np.abs(arr), 95)), 3),
        "max_abs": round(float(np.abs(arr).max()), 3),
        "sd": round(float(arr.std(ddof=1)) if arr.size > 1 else 0.0, 3),
    }


# ---------------------------------------------------------------------------
# 1. Colour space
# ---------------------------------------------------------------------------


def colour_space_trial(seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    """Score every candidate segmentation on masks known by construction."""
    scores: dict[str, dict[str, list[float]]] = {
        name: {"clean": [], "distractor": []} for name in blood_mod.SEGMENTERS
    }
    false_positive_px: dict[str, list[int]] = {name: [] for name in blood_mod.SEGMENTERS}

    for seed in seeds:
        for distractor in (0, DISTRACTOR_PX):
            key = "clean" if distractor == 0 else "distractor"
            for area in POOL_SIZES:
                scene = render(
                    SceneSpec(pool_area_px=area, blush_area_px=distractor, seed=seed + area)
                )
                lit = quality_mod.field_mask(scene.image)
                for name in blood_mod.SEGMENTERS:
                    mask, _ = blood_mod.blood_mask(scene.image, lit, method=name)
                    mask, _, _ = blood_mod.drop_small_pools(mask)
                    scores[name][key].append(dice(scene.blood_mask, mask))
                    if area == 0:
                        false_positive_px[name].append(int(mask.sum()))

    table = {}
    for name in blood_mod.SEGMENTERS:
        table[name] = {
            "dice_clean": round(float(np.mean(scores[name]["clean"])), 4),
            "dice_with_distractor": round(float(np.mean(scores[name]["distractor"])), 4),
            "false_positive_px_on_empty_field": round(
                float(np.mean(false_positive_px[name])), 1
            ),
        }
    winner = max(table, key=lambda n: table[n]["dice_with_distractor"])
    return {
        "candidates": table,
        "winner": winner,
        "in_use": BLOOD_COLOUR_SPACE,
        "agrees": winner == BLOOD_COLOUR_SPACE,
        "chosen_on": (
            "hand-labelled real frames (dev clips), not these synthetic scenes; the "
            "synthetic winner scored precision 0 and recall 0 on the held-out real clips"
        ),
        "seeds": list(seeds),
        "pool_sizes_px": list(POOL_SIZES),
        "distractor_px": DISTRACTOR_PX,
        "note": (
            "The distractor is a patch of inflamed serosa: redder than the tissue "
            "around it and no darker. It is the case that separates a measurement "
            "from a red-pixel counter, and it is why the decision is two-dimensional."
        ),
    }


# ---------------------------------------------------------------------------
# 2-4. Area, scale, volume
# ---------------------------------------------------------------------------


@dataclass
class AreaSample:
    seed: int
    truth_px: int
    measured_px: int
    error_pct: float
    dice: float
    mm_per_px_true: float
    mm_per_px_measured: float | None
    scale_error_pct: float | None
    volume_true_ml: float
    volume_ml: float | None
    volume_low_ml: float | None
    volume_high_ml: float | None
    covered: bool | None


def area_and_scale_trial(seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    """Area error, scale error, and whether the reported volume interval covers truth."""
    samples: list[AreaSample] = []
    for seed in seeds:
        for area in POOL_SIZES:
            if area == 0:
                continue
            spec = SceneSpec(pool_area_px=area, blush_area_px=DISTRACTOR_PX, seed=seed + area)
            scene = render(spec)
            lit = quality_mod.field_mask(scene.image)
            reading, _ = instruments_mod.read_frame(
                scene.image, lit, assumed_shaft_mm=spec.shaft_mm
            )
            measurement, mask = blood_mod.measure(
                scene.image,
                lit,
                method=BLOOD_COLOUR_SPACE,
                mm_per_px=reading.mm_per_px,
                mm_per_px_sigma=reading.mm_per_px_sigma,
                scale_source=reading.scale_source,
            )
            truth_px = scene.area_px
            err = 100.0 * (measurement.area_px - truth_px) / truth_px if truth_px else 0.0
            scale_err = (
                100.0 * (reading.mm_per_px - spec.mm_per_px) / spec.mm_per_px
                if reading.mm_per_px
                else None
            )
            covered = None
            if measurement.volume_ml is not None:
                covered = bool(
                    measurement.volume_ml_low <= scene.volume_ml <= measurement.volume_ml_high
                )
            samples.append(
                AreaSample(
                    seed=seed,
                    truth_px=truth_px,
                    measured_px=measurement.area_px,
                    error_pct=err,
                    dice=dice(scene.blood_mask, mask),
                    mm_per_px_true=spec.mm_per_px,
                    mm_per_px_measured=reading.mm_per_px,
                    scale_error_pct=scale_err,
                    volume_true_ml=scene.volume_ml,
                    volume_ml=measurement.volume_ml,
                    volume_low_ml=measurement.volume_ml_low,
                    volume_high_ml=measurement.volume_ml_high,
                    covered=covered,
                )
            )

    area_errors = [s.error_pct for s in samples]
    scale_errors = [s.scale_error_pct for s in samples if s.scale_error_pct is not None]
    covered = [s.covered for s in samples if s.covered is not None]
    volume_errors = [
        100.0 * (s.volume_ml - s.volume_true_ml) / s.volume_true_ml
        for s in samples
        if s.volume_ml and s.volume_true_ml
    ]
    by_size: dict[str, dict[str, float]] = {}
    for size in POOL_SIZES:
        if size == 0:
            continue
        subset = [s.error_pct for s in samples if abs(s.truth_px - size) < size * 0.5]
        if subset:
            by_size[str(size)] = _pct(subset)

    return {
        "area_error_pct": _pct(area_errors),
        "area_error_pct_by_pool_size": by_size,
        "dice": _pct([s.dice for s in samples]),
        "scale_error_pct": _pct(scale_errors),
        "volume_point_error_pct": _pct(volume_errors),
        "interval_coverage": {
            "n": len(covered),
            "covered": int(sum(covered)),
            "fraction": round(sum(covered) / len(covered), 4) if covered else None,
        },
        "samples": [asdict(s) for s in samples],
        "note": (
            "The point volume error is dominated by the film-depth assumption, not by "
            "the segmentation. The interval coverage is the number that matters: it "
            "says how often the range we print contains the truth."
        ),
    }


# ---------------------------------------------------------------------------
# 5. Refusal
# ---------------------------------------------------------------------------


def refusal_sweep(seeds: tuple[int, ...] = SEEDS[:4]) -> dict[str, Any]:
    """Blur, fog and occlusion from harmless to impossible.

    For each step we record whether the tool refused, and - separately - the area
    error it *would* have produced had the gate not been there. That second column
    is the whole argument for having the gates at all.
    """
    sweeps: dict[str, list[dict[str, Any]]] = {"blur": [], "fog": [], "occlusion": []}
    grids = {
        "blur": np.arange(0.0, 9.1, 1.0),
        "fog": np.arange(0.0, 0.91, 0.1),
        "occlusion": np.arange(0.0, 0.81, 0.1),
    }
    for kind, grid in grids.items():
        for level in grid:
            refused, errors, reasons = [], [], []
            for seed in seeds:
                kwargs: dict[str, Any] = {"pool_area_px": 12_000, "blush_area_px": DISTRACTOR_PX,
                                          "seed": seed}
                field_name = {"blur": "blur_sigma", "fog": "fog", "occlusion": "occlusion"}[kind]
                kwargs[field_name] = float(level)
                scene = render(SceneSpec(**kwargs))
                lit = quality_mod.field_mask(scene.image)
                q = quality_mod.assess(scene.image, mask=lit)
                refused.append(not q.usable)
                reasons.extend(q.reasons)
                mask, _ = blood_mod.blood_mask(scene.image, lit, method=BLOOD_COLOUR_SPACE)
                mask, _, _ = blood_mod.drop_small_pools(mask)
                truth = scene.area_px
                if truth:
                    errors.append(100.0 * (int(mask.sum()) - truth) / truth)
            sweeps[kind].append(
                {
                    "level": round(float(level), 3),
                    "refused_fraction": round(float(np.mean(refused)), 3),
                    "area_error_pct_if_forced": _pct(errors),
                    "reasons": sorted(set(reasons)),
                }
            )
    return {
        "sweeps": sweeps,
        "note": (
            "`area_error_pct_if_forced` is the error the segmenter returns when the "
            "gate is bypassed. Where it is large and the refusal fraction is 1.0, the "
            "gate is doing its job. Where both are small the gate is costing us "
            "measurable frames for nothing, and that is a defect."
        ),
    }


# ---------------------------------------------------------------------------
# 6-7. Onset, phase, checkpoint
# ---------------------------------------------------------------------------


def case_trial(tmpdir: Path, *, seeds: tuple[int, ...] = (7, 23, 41)) -> dict[str, Any]:
    """Whole scripted cases: onset timing, phase confusion, checkpoint behaviour."""
    tmpdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    confusion: dict[str, dict[str, int]] = {}
    # (truth, predicted, seconds to the nearest true transition) per frame, so the
    # score can be reported both strictly and with the tolerance that surgical
    # phase-recognition work conventionally allows around a boundary.
    labelled: list[tuple[str, str, float]] = []

    scripts = []
    for seed in seeds:
        scripts.append(("bleed", CaseScript(seed=seed, bleed_start_s=26.0,
                                            bleed_rate_ml_per_min=1.6)))
        scripts.append(("no-bleed", CaseScript(seed=seed + 1, bleed_start_s=None)))

    for kind, script in scripts:
        path = tmpdir / f"case-{kind}-{script.seed}.mp4"
        if not path.is_file():
            write_case_video(script, path)
        record = RunRecord(product="scopewatch-eval")
        with recording(record):
            # The automatic checkpoint is off by default; it is switched on here so the
            # synthetic trial can still say what it does when it is used.
            result = analyse_case(
                path,
                params=PipelineParams(
                    stride=3, use_dnn=False, max_frames=400, auto_checkpoint=True
                ),
            )

        truth_onset = script.bleed_start_s
        detected = bool(result.onset and result.onset.detected)
        measured_onset = (
            result.onset.timestamp_ms / 1000.0 if (detected and result.onset) else None
        )
        rows.append(
            {
                "case": path.name,
                "kind": kind,
                "truth_onset_s": truth_onset,
                "detected": detected,
                "onset_s": None if measured_onset is None else round(measured_onset, 2),
                "error_s": (
                    None
                    if (measured_onset is None or truth_onset is None)
                    else round(measured_onset - truth_onset, 2)
                ),
                "false_positive": bool(detected and truth_onset is None),
                "rescanned": result.rescanned,
                "usable_fraction": round(result.ledger.usable_fraction, 4),
                "checkpoint_raised": bool(result.loop.checkpoints),
                "checkpoint_at_s": (
                    round(result.loop.checkpoints[0].at_ms / 1000.0, 2)
                    if result.loop.checkpoints
                    else None
                ),
                "truth_wide_device_at_s": 34.0,
                "peak_volume_ml": (
                    round(result.peak_volume_ml, 4) if result.peak_volume_ml else None
                ),
                "mm_per_px": round(result.mm_per_px, 5) if result.mm_per_px else None,
                "mm_per_px_true": script.mm_per_px,
            }
        )

        boundaries = [entry[0] for entry in script.phase_plan]
        for i, frame in enumerate(result.frames):
            t_s = frame.timestamp_ms / 1000.0
            truth_label = script.label_at(t_s)
            predicted = result.phases.at(i)
            confusion.setdefault(truth_label, {})
            confusion[truth_label][predicted] = confusion[truth_label].get(predicted, 0) + 1
            to_boundary = min(abs(t_s - b) for b in boundaries)
            labelled.append((truth_label, predicted, to_boundary))

    errors = [r["error_s"] for r in rows if r["error_s"] is not None]
    bleeds = [r for r in rows if r["kind"] == "bleed"]
    quiet = [r for r in rows if r["kind"] == "no-bleed"]

    total = sum(sum(v.values()) for v in confusion.values())
    correct = sum(v.get(k, 0) for k, v in confusion.items())

    # Away from a boundary the label is either right or wrong on its own merits.
    # Within a few seconds of a transition, a state machine with hysteresis is by
    # construction still on the previous label, and scoring those frames as errors
    # measures the hysteresis rather than the model. Both numbers are published.
    tolerance_s = 6.0
    settled = [(t, p) for t, p, d in labelled if d > tolerance_s]
    settled_correct = sum(1 for t, p in settled if t == p)
    per_phase: dict[str, dict[str, Any]] = {}
    for name in PHASES:
        truth_n = sum(1 for t, _p, _d in labelled if t == name)
        hit = sum(1 for t, p, _d in labelled if t == name and p == name)
        pred_n = sum(1 for _t, p, _d in labelled if p == name)
        per_phase[name] = {
            "frames_true": truth_n,
            "recall": round(hit / truth_n, 4) if truth_n else None,
            "precision": round(hit / pred_n, 4) if pred_n else None,
        }
    return {
        "cases": rows,
        "onset": {
            "detected_on_bleeding_cases": f"{sum(r['detected'] for r in bleeds)}/{len(bleeds)}",
            "false_positives_on_quiet_cases": f"{sum(r['detected'] for r in quiet)}/{len(quiet)}",
            "error_s": _pct(errors),
        },
        "phase": {
            "confusion": confusion,
            "frames": total,
            "accuracy": round(correct / total, 4) if total else None,
            "accuracy_away_from_boundaries": (
                round(settled_correct / len(settled), 4) if settled else None
            ),
            "boundary_tolerance_s": tolerance_s,
            "frames_away_from_boundaries": len(settled),
            "per_phase": per_phase,
            "order": list(PHASES),
            "note": (
                "Scored on scripted synthetic sequences where the phase is known by "
                "construction. This is evidence about the state machine, not about "
                "operative video, and it must not be read as the latter."
            ),
        },
        "checkpoint": {
            "raised": f"{sum(r['checkpoint_raised'] for r in rows)}/{len(rows)}",
            "delay_s": _pct(
                [
                    r["checkpoint_at_s"] - r["truth_wide_device_at_s"]
                    for r in rows
                    if r["checkpoint_at_s"] is not None
                ]
            ),
        },
        "rescan": {
            "performed": f"{sum(r['rescanned'] for r in rows)}/{len(rows)}",
            "note": (
                "The second dense read happens only when the coarse pass found an "
                "onset. A quiet case never triggers one, which is the loop doing "
                "what it claims."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def timing_trial(repeats: int = 12) -> dict[str, Any]:
    """Per-stage cost on this machine, so the report quotes a measured number."""
    scene = render(SceneSpec(pool_area_px=12_000, blush_area_px=DISTRACTOR_PX, seed=5))
    image = scene.image
    out: dict[str, float] = {}

    def timeit(name: str, fn) -> None:
        fn()  # warm
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        out[name] = round((time.perf_counter() - start) * 1000.0 / repeats, 3)

    lit = quality_mod.field_mask(image)
    timeit("field_mask", lambda: quality_mod.field_mask(image))
    timeit("quality.assess", lambda: quality_mod.assess(image, mask=lit))
    timeit("blood.segment", lambda: blood_mod.blood_mask(image, lit, method=BLOOD_COLOUR_SPACE))
    timeit("instruments.find_shafts", lambda: instruments_mod.find_shafts(image, lit))
    estimator_image = image.copy()
    timeit(
        "motion.phaseCorrelate",
        lambda: cv2.phaseCorrelate(
            cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (256, 256)).astype(np.float32),
            cv2.resize(
                cv2.cvtColor(estimator_image, cv2.COLOR_BGR2GRAY), (256, 256)
            ).astype(np.float32),
        ),
    )
    return {
        "ms_per_frame_at_960x540": out,
        "total_ms": round(sum(out.values()), 3),
        "repeats": repeats,
        "machine": platform.machine(),
        "threads": cv2.getNumThreads(),
        "opencv": cv2.__version__,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def write_plots(results: dict[str, Any], out_dir: Path) -> list[str]:
    """Plots in the product's own palette. Skipped cleanly if matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    ground, surface, ink, accent, ok = "#0B1210", "#141F1B", "#E9F1EC", "#FF8A3D", "#5FD3A6"
    written: list[str] = []

    def style(ax, title: str, xlabel: str, ylabel: str) -> None:
        ax.set_facecolor(surface)
        ax.set_title(title, color=ink, fontsize=11, loc="left", pad=12)
        ax.set_xlabel(xlabel, color="#9DB3AC", fontsize=9)
        ax.set_ylabel(ylabel, color="#9DB3AC", fontsize=9)
        ax.tick_params(colors="#7F968F", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#26362F")
        ax.grid(True, color="#1D2C27", linewidth=0.8)
        ax.set_axisbelow(True)

    # --- area error against pool size ---
    samples = results["area_and_scale"]["samples"]
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=ground)
    ax.scatter([s["truth_px"] for s in samples], [s["error_pct"] for s in samples],
               s=16, color=accent, alpha=0.75, edgecolors="none")
    ax.axhline(0, color=ok, linewidth=1)
    ax.set_xscale("log")
    style(ax, "Area error against true pool area", "true pool area, pixels (log)", "error, %")
    fig.tight_layout()
    path = out_dir / "area-error.png"
    fig.savefig(path, dpi=160, facecolor=ground)
    plt.close(fig)
    written.append(path.name)

    # --- refusal sweeps ---
    # The refusal fraction and the error share an x-axis and nothing else. Past the
    # occlusion gate the error reaches 150,000%, and on a shared linear y-axis that
    # flattens the refusal curve - the thing the plot is about - onto the baseline.
    # The error gets its own log axis on the right.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor=ground)
    for ax, (kind, rows) in zip(axes, results["refusal"]["sweeps"].items(), strict=False):
        levels = [r["level"] for r in rows]
        refused = [100.0 * r["refused_fraction"] for r in rows]
        errs = [max(0.05, abs(r["area_error_pct_if_forced"].get("median", 0.0)))
                for r in rows]
        ax.plot(levels, refused, color=accent, linewidth=2.2, marker="o", markersize=4,
                label="refused, %", zorder=3)
        ax.set_ylim(-4, 104)
        style(ax, kind, kind, "refused, %")

        twin = ax.twinx()
        twin.plot(levels, errs, color=ok, linewidth=1.6, linestyle="--", marker="s",
                  markersize=3, label="|error| if forced, %")
        twin.set_yscale("log")
        twin.set_ylabel("|error| if forced, % (log)", color="#9DB3AC", fontsize=8)
        twin.tick_params(colors="#7F968F", labelsize=8)
        twin.set_facecolor("none")
        for spine in twin.spines.values():
            spine.set_color("#26362F")

        handles = ax.get_lines() + twin.get_lines()
        leg = ax.legend(handles, [h.get_label() for h in handles], fontsize=8,
                        facecolor=surface, edgecolor="#26362F", loc="center left")
        for text in leg.get_texts():
            text.set_color(ink)
    fig.tight_layout()
    path = out_dir / "refusal-sweeps.png"
    fig.savefig(path, dpi=160, facecolor=ground)
    plt.close(fig)
    written.append(path.name)

    # --- colour space comparison ---
    table = results["colour_space"]["candidates"]
    names = list(table)
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=ground)
    x = np.arange(len(names))
    ax.bar(x - 0.2, [table[n]["dice_clean"] for n in names], 0.4, color="#5FD3A6",
           label="clean field")
    ax.bar(x + 0.2, [table[n]["dice_with_distractor"] for n in names], 0.4, color=accent,
           label="with inflamed serosa")
    ax.set_xticks(x)
    ax.set_xticklabels(names, color="#9DB3AC", fontsize=9)
    ax.set_ylim(0, 1.05)
    style(ax, "Segmentation Dice by colour-space decision", "", "Dice")
    leg = ax.legend(fontsize=8, facecolor=surface, edgecolor="#26362F")
    for text in leg.get_texts():
        text.set_color(ink)
    fig.tight_layout()
    path = out_dir / "colour-spaces.png"
    fig.savefig(path, dpi=160, facecolor=ground)
    plt.close(fig)
    written.append(path.name)

    # --- one case's field trace ---
    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor=ground)
    trace = results.get("trace")
    if trace:
        t = np.asarray(trace["times_ms"]) / 1000.0
        ax.plot(t, trace["smoothed"], color=accent, linewidth=1.8)
        ax.plot(t, trace["raw"], color="#7F968F", linewidth=0.8, alpha=0.7)
        if trace.get("onset_s") is not None:
            ax.axvline(trace["onset_s"], color=ok, linewidth=1.4, linestyle="--")
            ax.text(trace["onset_s"] + 0.6, max(trace["smoothed"]) * 0.9,
                    f"measured onset {trace['onset_s']:.1f} s", color=ok, fontsize=8)
        if trace.get("truth_onset_s") is not None:
            ax.axvline(trace["truth_onset_s"], color=ink, linewidth=1.0, linestyle=":")
            ax.text(trace["truth_onset_s"] - 12.0, max(trace["smoothed"]) * 0.75,
                    f"true onset {trace['truth_onset_s']:.1f} s", color=ink, fontsize=8)
    style(ax, "Blood-covered field over one scripted case", "seconds",
          "% of the visible field")
    fig.tight_layout()
    path = out_dir / "field-trace.png"
    fig.savefig(path, dpi=160, facecolor=ground)
    plt.close(fig)
    written.append(path.name)
    return written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_all(out_dir: Path, *, quick: bool = False) -> dict[str, Any]:
    seeds = SEEDS[:3] if quick else SEEDS
    started = time.time()
    results: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": environment().to_dict(),
        "colour_space": colour_space_trial(seeds),
        "area_and_scale": area_and_scale_trial(seeds),
        "refusal": refusal_sweep(seeds[:4]),
        "timing": timing_trial(),
    }
    cases = case_trial(out_dir.parent / "media" / "eval", seeds=(7,) if quick else (7, 23, 41))
    results["cases"] = cases

    # One case's series, for the trace plot and the report's figure.
    trace_path = out_dir.parent / "media" / "eval" / f"case-bleed-{7}.mp4"
    if trace_path.is_file():
        record = RunRecord(product="scopewatch-eval")
        with recording(record):
            r = analyse_case(trace_path, params=PipelineParams(stride=3, use_dnn=False))
        if r.series:
            results["trace"] = {
                "times_ms": r.series.times_ms,
                "raw": r.series.raw,
                "smoothed": r.series.smoothed,
                "onset_s": (r.onset.timestamp_ms / 1000.0
                            if (r.onset and r.onset.detected) else None),
                "truth_onset_s": 26.0,
            }

    real = out_dir / "real-evaluation.json"
    if real.is_file():
        results["real"] = json.loads(real.read_text(encoding="utf-8"))

    results["plots"] = write_plots(results, out_dir / "plots")
    results["elapsed_s"] = round(time.time() - started, 1)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs", help="directory for evaluation.json and plots")
    parser.add_argument("--quick", action="store_true", help="fewer seeds, for a smoke run")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = run_all(out_dir, quick=args.quick)

    # The trace is large and only the plot needs it; keep the JSON readable.
    published = {k: v for k, v in results.items() if k != "trace"}
    (out_dir / "evaluation.json").write_text(
        json.dumps(published, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in published.items()
                      if k in ("colour_space", "cases", "elapsed_s")},
                     indent=2, default=str)[:4000])
    print(f"\nwrote {out_dir / 'evaluation.json'} and {len(results['plots'])} plots")


if __name__ == "__main__":
    main()
