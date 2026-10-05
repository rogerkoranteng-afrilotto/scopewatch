"""The evaluation on real laparoscopic footage, built around what real footage can check.

There is no true blood volume for any of these clips, so nothing here scores a
millilitre. What it does score:

1. **Segmentation** against hand-drawn blood polygons on a fixed sample of frames
   (`eval/real/blood-labels.json`): pixel precision and recall, split into the clips
   thresholds were allowed to be chosen on (`dev`) and the clips that were not (`test`).
2. **Refusals**: which frames each gate refused, and whether the out-of-domain clip
   and the draped tail were refused.
3. **Onset**: whether and when it fired, and which gate stopped it when it did not,
   next to the visible-bleeding notes in `eval/real/onset-truth.json`.
4. **Checkpoint**: whether the automatic checkpoint would be raised, and whether the
   image cue was present or the dwell timer did the work.
5. **Scale**: whether the case scale passed its gate, and the field width it implies.

    python -m scopewatch.realeval --media /path/to/clips --out runs/ [--labels ...]

The clips are not in this repository; `realdata.CLIPS` says where each one is from.
The same module scores the baseline commit, which is how the before-and-after tables
in docs/evaluation.md were made: it reads only fields both versions write.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from visioncore import RunRecord, recording

from . import blood as blood_mod
from . import pipeline as pipeline_mod
from . import quality as quality_mod
from .config import BLOOD_COLOUR_SPACE, PipelineParams
from .realdata import CLIPS, Tally, load_labels, rasterise, read_frames

PRODUCT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = PRODUCT_ROOT / "eval" / "real" / "blood-labels.json"


def _analysis_size(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = max_side / max(h, w)
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def segmentation(media: Path, labels_path: Path, *, max_side: int = 960) -> dict[str, Any]:
    """Precision and recall of the blood mask on the hand-labelled frames."""
    labels = load_labels(labels_path)["frames"]
    by_clip: dict[str, list[tuple[str, int]]] = {}
    for key in labels:
        clip, index = key.rsplit("@", 1)
        by_clip.setdefault(clip, []).append((key, int(index)))

    tallies = {name: Tally() for name in ("dev", "test", "all_laparoscopic", "open")}
    tallies_gated = {name: Tally() for name in ("dev", "test", "all_laparoscopic")}
    frames: list[dict[str, Any]] = []
    for clip in CLIPS:
        wanted = by_clip.get(clip.name)
        if not wanted:
            continue
        images = read_frames(media / clip.filename, [i for _, i in wanted])
        for key, index in wanted:
            image = _analysis_size(images[index], max_side)
            lit = quality_mod.field_mask(image)
            q = quality_mod.assess(image, mask=lit)
            m, mask = blood_mod.measure(image, lit, method=BLOOD_COLOUR_SPACE)
            entry = labels[key]
            truth, ignore = rasterise(entry, image.shape[:2])
            view = entry.get("view", "laparoscopic")
            one = Tally()
            one.add(mask, truth, ignore)
            if clip.domain == "open" or view != "laparoscopic":
                tallies["open"].add(mask, truth, ignore)
            else:
                tallies[clip.split].add(mask, truth, ignore)
                tallies["all_laparoscopic"].add(mask, truth, ignore)
                if q.usable:
                    tallies_gated[clip.split].add(mask, truth, ignore)
                    tallies_gated["all_laparoscopic"].add(mask, truth, ignore)
            frames.append({
                "key": key, "clip": clip.name, "split": clip.split, "view": view,
                "usable": q.usable, "refusal": q.reason,
                "true_blood_fraction": round(float(truth[ignore == 0].mean()), 4)
                if (ignore == 0).any() else None,
                "predicted_fraction": round(m.area_fraction, 4),
                **one.to_dict(),
            })
    return {
        "labels": str(labels_path.name),
        "frames": len(frames),
        "pixelwise": {k: v.to_dict() for k, v in tallies.items()},
        "pixelwise_on_frames_the_gates_passed": {k: v.to_dict() for k, v in tallies_gated.items()},
        "per_frame": frames,
    }


def run_clip(path: Path, params: PipelineParams) -> dict[str, Any]:
    """One clip through the same call path as the service, summarised."""
    record = RunRecord(product="scopewatch")
    started = time.perf_counter()
    with recording(record):
        result = pipeline_mod.analyse_case(path, params=params)
    pipeline_mod.to_record(result, record, params)
    data = json.loads(record.to_json())
    m = data["metrics"]
    onset = m.get("onset") or {}
    agent = m.get("agent") or {}
    held = [c for c in agent.get("checkpoints", [])]
    phases = m.get("phases") or {}
    approach = next(
        (s["start_ms"] for s in phases.get("spans", []) if s["phase"] == "critical_approach"),
        None,
    )
    blood = m.get("blood") or {}
    scale = m.get("scale") or {}
    width = (m.get("video") or {}).get("analysed_long_side") or max(
        (m.get("video") or {}).get("width") or 0, (m.get("video") or {}).get("height") or 0
    )
    width = min(width, params.max_side) if width else width
    return {
        "clip": path.stem,
        "seconds": round(time.perf_counter() - started, 1),
        "frames": m["quality"]["frames"],
        "usable": m["quality"]["usable"],
        "rejected_by": m["quality"]["rejected_by"],
        "record_refusals": [r["code"] for r in data.get("refusals", [])],
        "scale": {
            "mm_per_px": scale.get("mm_per_px"),
            "gate": scale.get("gate"),
            "implied_field_width_mm": scale.get("implied_field_width_mm")
            or (round(scale["mm_per_px"] * width, 1) if scale.get("mm_per_px") and width else None),
            "frames_with_scale": (scale.get("gate") or {}).get("frames_with_scale"),
        },
        "blood": {
            "peak_field_fraction": blood.get("peak_field_fraction"),
            "median_field_fraction": blood.get("median_field_fraction"),
            "volume_status": (blood.get("volume") or {}).get("status"),
            "volume_reason": (blood.get("volume") or {}).get("reason"),
            "peak_volume_ml": blood.get("peak_volume_ml"),
        },
        "onset": {
            "detected": onset.get("detected"),
            "timestamp_s": None if onset.get("timestamp_ms") is None
            else round(onset["timestamp_ms"] / 1000.0, 1),
            "reason": onset.get("reason"),
            "blocked_by": onset.get("blocked_by"),
            "unit": onset.get("unit", "ml/min"),
        },
        "checkpoint": {
            "state": agent.get("state"),
            "held_at_s": round(held[0]["at_ms"] / 1000.0, 1) if held else None,
            "critical_approach_from_s": None if approach is None else round(approach / 1000.0, 1),
            "wide_cue": phases.get("wide_cue"),
        },
        "series": {
            "times_ms": (m.get("series") or {}).get("times_ms"),
            "smoothed": (m.get("series") or {}).get("smoothed"),
            "measurable": (m.get("series") or {}).get("measurable"),
            "unit": m.get("series_unit", "ml"),
        },
    }


def _visible_bleeding_match(onset_s: float | None, truth: dict[str, Any], slack_s: float) -> str:
    """Does a detected onset fall inside a span where the labeller saw new bleeding?"""
    spans = truth.get("new_bleeding") or []
    if onset_s is None:
        return "no onset; visible new bleeding noted" if spans else "no onset; none noted"
    for span in spans:
        if span["from_s"] - slack_s <= onset_s <= span["to_s"] + slack_s:
            return f"inside a noted span ({span['from_s']}-{span['to_s']} s, {span['confidence']})"
    return "not near any noted new bleeding" if spans else "fired on a clip with none noted"


def summarise(before: Path, after: Path, truth_path: Path) -> dict[str, Any]:
    """The before-and-after tables in docs/evaluation.md, from two run directories."""
    truth = json.loads(truth_path.read_text(encoding="utf-8")) if truth_path.is_file() else {}
    rows: list[dict[str, Any]] = []
    for clip in CLIPS:
        b_path, a_path = before / f"{clip.name}.json", after / f"{clip.name}.json"
        if not (b_path.is_file() and a_path.is_file()):
            continue
        b = json.loads(b_path.read_text(encoding="utf-8"))
        a = json.loads(a_path.read_text(encoding="utf-8"))
        t = truth.get(clip.name, {})
        a_gate = (a["scale"].get("gate") or {})
        rows.append({
            "clip": clip.name,
            "split": clip.split,
            "domain": clip.domain,
            "frames": a["frames"],
            "before": {
                "usable": b["usable"],
                "record_refusals": b["record_refusals"],
                "mm_per_px": b["scale"]["mm_per_px"],
                "implied_field_width_mm": b["scale"]["implied_field_width_mm"],
                "peak_volume_ml": b["blood"]["peak_volume_ml"],
                "onset_detected": b["onset"]["detected"],
                "onset_s": b["onset"]["timestamp_s"],
                "onset_reason": b["onset"]["reason"],
                "checkpoint_state": b["checkpoint"]["state"],
                "checkpoint_held_at_s": b["checkpoint"]["held_at_s"],
                "critical_approach_from_s": b["checkpoint"]["critical_approach_from_s"],
            },
            "after": {
                "usable": a["usable"],
                "rejected_by": a["rejected_by"],
                "record_refusals": a["record_refusals"],
                "scale_status": a_gate.get("status"),
                "scale_frames": (
                    f"{a_gate.get('frames_with_scale')}/{a_gate.get('measurable_frames')}"
                ),
                "scale_robust_cv": a_gate.get("robust_cv"),
                "mm_per_px": a["scale"]["mm_per_px"],
                "implied_field_width_mm": a["scale"]["implied_field_width_mm"],
                "peak_field_fraction": a["blood"]["peak_field_fraction"],
                "median_field_fraction": a["blood"]["median_field_fraction"],
                "volume_status": a["blood"]["volume_status"],
                "volume_reason": a["blood"]["volume_reason"],
                "peak_volume_ml": a["blood"]["peak_volume_ml"],
                "onset_detected": a["onset"]["detected"],
                "onset_s": a["onset"]["timestamp_s"],
                "onset_reason": a["onset"]["reason"],
                "onset_blocked_by": a["onset"]["blocked_by"],
                "onset_vs_visible_bleeding": _visible_bleeding_match(
                    a["onset"]["timestamp_s"], t, slack_s=4.0
                ),
                "checkpoint_if_enabled": a["checkpoint"]["state"],
                "checkpoint_held_at_s": a["checkpoint"]["held_at_s"],
                "critical_approach_from_s": a["checkpoint"]["critical_approach_from_s"],
                "wide_cue": a["checkpoint"]["wide_cue"],
            },
            "visible_bleeding": {
                "new_bleeding": t.get("new_bleeding"),
                "non_laparoscopic_spans": t.get("non_laparoscopic_spans"),
                "summary": t.get("summary"),
            },
        })
    seg_before = json.loads((before / "segmentation.json").read_text(encoding="utf-8"))
    seg_after = json.loads((after / "segmentation.json").read_text(encoding="utf-8"))
    return {
        "note": (
            "No real clip has a known blood volume, so no millilitre is scored here. "
            "Segmentation is scored against hand-drawn polygons on 64 fixed frames; "
            "thresholds were chosen on the dev clips only."
        ),
        "segmentation": {
            "before": seg_before["pixelwise"],
            "after": seg_after["pixelwise"],
            "before_gated": seg_before["pixelwise_on_frames_the_gates_passed"],
            "after_gated": seg_after["pixelwise_on_frames_the_gates_passed"],
            "frames": seg_after["frames"],
        },
        "clips": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--media", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--clips", nargs="*", default=None)
    parser.add_argument("--skip-cases", action="store_true")
    parser.add_argument("--auto-checkpoint", action="store_true",
                        help="switch the automatic checkpoint on, to evaluate it")
    parser.add_argument("--summarise", nargs=2, type=Path, metavar=("BEFORE", "AFTER"),
                        help="write the before-and-after summary to --out and stop")
    args = parser.parse_args()
    if args.summarise:
        truth = args.labels.parent / "onset-truth.json"
        summary = summarise(args.summarise[0], args.summarise[1], truth)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
        return
    args.out.mkdir(parents=True, exist_ok=True)

    if args.labels.is_file():
        seg = segmentation(args.media, args.labels)
        (args.out / "segmentation.json").write_text(json.dumps(seg, indent=1))
        print("segmentation", json.dumps(seg["pixelwise"]))

    if args.skip_cases:
        return
    params = PipelineParams()
    params.use_dnn = False  # the DNN channel reports; it changes no number here
    if hasattr(params, "auto_checkpoint"):
        params.auto_checkpoint = bool(args.auto_checkpoint)
    names = args.clips or [c.name for c in CLIPS]
    for name in names:
        summary = run_clip(args.media / f"{name}.mp4", params)
        (args.out / f"{name}.json").write_text(json.dumps(summary, indent=1))
        short = {k: v for k, v in summary.items() if k != "series"}
        print(json.dumps(short)[:900])


if __name__ == "__main__":
    main()
