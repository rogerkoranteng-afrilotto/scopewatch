"""The reasoning layer: NVIDIA Nemotron on Nebius Token Factory reads what OpenCV measured.

Nothing here looks at a pixel. The model is handed a digest of the engine's output (the
measured series, the phase track, the refusals, the gates that stopped the onset
detector and the volume figure, and how well the segmentation has scored on held-out
real frames) and it does three jobs a threshold cannot do:

  1. decide whether to raise a safety checkpoint before the irreversible step, and say why;
  2. explain, in plain words, every refusal or suspension the engine produced;
  3. write a short surgeon-facing summary of the measured case.

Every number the model cites is checked against the digest afterwards (`ground`). A
citation that does not match what OpenCV measured is flagged on screen, not hidden.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import nemotron

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache" / "reasoning"
MODEL = nemotron.SUPER
MODEL_LABEL = "NVIDIA Nemotron 3 Super 120B (A12B)"

# What the validation of the measurement says, supplied to the model verbatim so it can
# weigh the series by how much the series deserves to be trusted.
VALIDATION = {
    "blood_segmentation_vs_hand_drawn_masks": {
        "held_out_test_clips": {"precision": 0.636, "recall": 0.077},
        "dev_clips": {"precision": 0.822, "recall": 0.401},
        "pooled_15_laparoscopic_clips": {"precision": 0.782, "recall": 0.232},
        "meaning": ("The colour rule is tuned for precision (F-0.5, leave-one-clip-out): on "
                    "held-out real clips about 64% of the pixels it marks as blood are blood, "
                    "but it finds only about 8% of the blood a person drew. So a marked region "
                    "is usually real, and the blood share is a conservative under-count; absolute "
                    "levels read low and movement over time matters more than the level. It "
                    "separates blood from shadow and from a port sleeve."),
    },
    "millilitres": "never validated against a real volume; refused when the shaft scale wanders",
    "phase_track": ("scripted synthetic sequences only; on real video the instrument-width cue "
                    "cannot tell a clip applier from a grasper nearer the lens, so the "
                    "phase track is a weak signal"),
    "bleeding_onset": "has not fired on any of 16 real clips; the camera is rarely still long enough",
}

_lock = threading.Lock()


# ─────────────────────────────── the digest ───────────────────────────────

def _spans(flags: list[tuple[float, str | None]], gap_ms: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t, code in flags:
        if code is None:
            continue
        if out and out[-1]["code"] == code and t - out[-1]["end_ms"] <= gap_ms:
            out[-1]["end_ms"] = t
            out[-1]["frames"] += 1
        else:
            out.append({"code": code, "start_ms": t, "end_ms": t, "frames": 1})
    return out


def refusal_spans(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Runs of consecutive refused frames, by code. Codes come from the engine's evidence."""
    s = (record.get("metrics") or {}).get("series") or {}
    times, meas = s.get("times_ms") or [], s.get("measurable") or []
    code_at: dict[float, str] = {}
    for e in record.get("evidence") or []:
        code = (e.get("metrics") or {}).get("refusal")
        if code and e.get("timestamp_ms") is not None:
            code_at[round(e["timestamp_ms"], 0)] = code
    flags = []
    for t, ok in zip(times, meas):
        if ok:
            continue
        flags.append((t, code_at.get(round(t, 0)) or _nearest(code_at, t)))
    step = (times[1] - times[0]) if len(times) > 1 else 200
    spans = _spans(flags, gap_ms=max(step * 2.5, 600))
    for i, sp in enumerate(spans):
        sp["id"] = f"R{i + 1}"
    return spans


def _nearest(code_at: dict[float, str], t: float) -> str | None:
    if not code_at:
        return "REFUSED"
    k = min(code_at, key=lambda x: abs(x - t))
    return code_at[k] if abs(k - t) < 1500 else "REFUSED"


def digest(record: dict[str, Any], procedure: str = "") -> dict[str, Any]:
    """Everything the model may know about the case, and nothing it did not measure."""
    m = record.get("metrics") or {}
    s = m.get("series") or {}
    times = s.get("times_ms") or []
    dur_s = (m.get("video") or {}).get("duration_ms", 0) / 1000.0
    rows = []
    last = -1e9
    for i, t in enumerate(times):
        if t - last < 1000:           # one row per second keeps the prompt small and citeable
            continue
        last = t
        rows.append({
            "t_s": round(t / 1000, 1),
            "blood_pct": round(s["smoothed"][i], 2),
            "rate_pct_per_min": round(s["rate_per_min"][i], 1),
            "measurable": bool(s["measurable"][i]),
            "reliable": bool(s["reliable"][i]),
            "instruments": s["instruments"][i],
            "camera_motion_px": round(s["motion_px"][i], 1),
        })
    meas = [(r["blood_pct"], r["t_s"]) for r in rows if r["measurable"]]
    peak = max(meas) if meas else (None, None)
    raw = [(v, times[i] / 1000.0) for i, v in enumerate(s.get("raw") or []) if s["measurable"][i]]
    raw_peak = max(raw) if raw else (None, None)
    phases = [{"phase": p["phase"], "start_s": round(p["start_ms"] / 1000, 1),
               "end_s": round(p["end_ms"] / 1000, 1)}
              for p in (m.get("phases") or {}).get("spans", [])]
    on = m.get("onset") or {}
    q = m.get("quality") or {}
    scale = (m.get("scale") or {}).get("gate") or {}
    refs = refusal_spans(record)
    clip_refusal = (record.get("refusals") or [{}])[0] if record.get("refusals") else None
    return {
        "clip": {"duration_s": round(dur_s, 1), "frames_read": q.get("frames"),
                 "frames_usable": q.get("usable"), "refused_by": q.get("rejected_by") or {}},
        "procedure_stated_by_operator": procedure or "not stated",
        "clip_level_refusal": ({"code": clip_refusal.get("code"), "message": clip_refusal.get("message")}
                               if clip_refusal else None),
        "blood_covered_field_pct": {
            "peak": round((m.get("blood") or {}).get("peak_field_fraction", 0) * 100, 2),
            "peak_at_s": round(raw_peak[1], 1) if raw_peak[1] is not None else None,
            "peak_smoothed": peak[0], "peak_smoothed_at_s": peak[1],
            "median": round((m.get("blood") or {}).get("median_field_fraction", 0) * 100, 2),
            "note": ("percent of the visible field. `peak` and `median` are the per-frame figures "
                     "shown to the surgeon; the series below is smoothed, one row per second, so its "
                     "maximum is a little lower than `peak`"),
        },
        "series": rows,
        "phase_track": {"spans": phases,
                        "reached_critical_approach": any(p["phase"] == "critical_approach" for p in phases),
                        "reliability": VALIDATION["phase_track"]},
        "bleeding_onset": {"fired": bool(on.get("detected")), "timestamp_s": (on.get("timestamp_ms") or 0) / 1000
                           if on.get("detected") else None,
                           "peak_rate_pct_per_min": on.get("peak_rate_any_per_min"),
                           "threshold_pct_per_min": on.get("threshold_per_min"),
                           "candidate_frames": on.get("candidates"),
                           "blocked_by": on.get("blocked_by") or {}, "engine_reason": on.get("reason")},
        "volume_ml": {"status": (m.get("blood") or {}).get("volume", {}).get("status"),
                      "reason": (m.get("blood") or {}).get("volume", {}).get("reason"),
                      "scale_gate": {k: scale.get(k) for k in ("status", "robust_cv", "frame_share")}},
        "refusal_events": [{"id": r["id"], "code": r["code"], "start_s": round(r["start_ms"] / 1000, 1),
                            "end_s": round(r["end_ms"] / 1000, 1), "frames": r["frames"]} for r in refs],
        "engine_actions": [{"kind": a["kind"], "at_s": round(a["at_ms"] / 1000, 1)}
                           for a in (m.get("agent") or {}).get("actions", [])],
        "safety_view_recorded_by_operator": bool((record.get("params") or {}).get("safety_view_established")),
        "validation_of_the_measurement": VALIDATION,
    }


# ─────────────────────────────── the three calls ───────────────────────────────

RULES = (
    "You are the reasoning layer of Scopewatch, a retrospective measurement instrument for "
    "laparoscopic video. OpenCV measured everything in the JSON you are given; you did not "
    "see the video and you must not claim to. Use only the figures in the JSON. Never state "
    "a volume in millilitres. Never invent an anatomy, an instrument or an event the JSON does "
    "not contain. In particular, never say or imply that a rise in blood followed, caused or "
    "marks any operative step (clipping, division, dissection finishing): the series does not "
    "show that. Treat a rise only as more of the field being called blood from that time. "
    "You advise a named human; you take no clinical action and block nothing. "
    "Write plainly, as to a surgeon: short sentences, no hype, no hedging boilerplate. "
    "Reply with one JSON object and nothing else."
)

CHECKPOINT_TASK = """TASK: decide whether Scopewatch should raise a safety checkpoint, and where.

You are reading a finished recording. There is no "now": the whole case is in front of you.
Decide whether a checkpoint belonged anywhere in it and, if so, the second at which it should
have been raised (before the irreversible step, not after).

A checkpoint pauses the review record and asks a named person to confirm that the critical
view of safety (CVS) is established, or to proceed with a stated reason, before the
irreversible step of the operation (clipping or dividing a duct or vessel). It costs the
surgeon a few seconds. Raising one needlessly erodes trust in it; failing to raise one when
the irreversible step is near is the failure that matters. The engine's own automatic cue
is switched off because it fires on a timer; you are the replacement judgement, and you
must justify yours from the measured evidence.

Two facts about the evidence. The phase track is a weak signal in both directions: on real
video its cue is instrument width, so reaching critical_approach proves little and not reaching
it proves little. And the blood series is a coarse indicator: movement matters more than level.
The operator's statement of what is happening is therefore a primary input, not a footnote.

Weigh: what the operator says the procedure is; whether the procedure has an irreversible
step a CVS-type check applies to; what the phase track and the blood series show about how
close that step may be; whether the measurement is trustworthy enough to rely on (see the
validation block and the refusal events); and whether the operator has already recorded the
safety view. The blood series says how much of the field was called blood and when; it does not say what
the surgeon was doing at that moment. Do not infer which operative step happened when from it.
Where the procedure has no such step, say so and do not raise. If the clip is refused
as not a laparoscopic view, answer "cannot_assess": Scopewatch has nothing measured to reason
from, which is different from the step being safe.

Return JSON:
{"decision": "raise" | "do_not_raise" | "cannot_assess",
 "at_s": <seconds into the clip where the checkpoint belongs, or null>,
 "headline": "<one sentence, at most 22 words, the decision and its main reason>",
 "reasoning": "<4 to 7 plain sentences: the chain from evidence to decision>",
 "evidence": [ 3 to 5 items {"t_s": <number>, "metric": "blood_pct" | "rate_pct_per_min" | "phase" | "refusal" | "instruments", "value": <number or phase name or refusal code>, "reading": "<what it tells you, one sentence>"} ],
 "limits": "<one or two sentences: what about the measurement makes this decision uncertain>",
 "would_change_if": "<one sentence: what evidence would flip the decision>"}
Each evidence value must be copied from the series, phase track or refusal events."""

EXPLAIN_TASK = """TASK: explain each refusal or suspension in plain language for a surgeon.

You are given the list of events below. For every one, say what it means in this clip, what
Scopewatch did because of it (it never guesses a number through a refusal), and what a person
could do about it. Next steps must be things an operator can do with a clip: record again,
hold the camera still, bring an instrument shaft into view, enter millimetres per pixel,
or read the number with its limits. Do not suggest changing hardware or the algorithm. Do
not confuse "candidate frames" (moments the rate crossed the threshold) with refused frames. Include the clip-level refusal if present, the reason the bleeding-onset
detector did not fire (its blocked_by gates) and the reason no millilitre figure was given.

Events to explain (use these ids): {ids}

Return JSON:
{{"explanations": [ {{"id": "<event id>", "title": "<4 to 8 words>", "plain": "<2 to 3 sentences: what it means here>", "did": "<one sentence: what Scopewatch did>", "next": "<one sentence: what a person can do>"}} ]}}"""

SUMMARY_TASK = """TASK: write the case summary a surgeon would read after the case.

Three to five sentences over the measured trace: how much of the field the engine called
blood and when it peaked, whether the level moved or stayed flat, what the engine refused
and for how long, what it could not tell, and whether the checkpoint question was
raised. Quote only figures present in the JSON; "refused" means the refused_by counts only, never
candidate_frames. State the checkpoint decision (given in the JSON) in one clause. End with the limit that matters most for
reading these numbers. No headings, no lists.

Return JSON:
{"summary": "<the paragraph>",
 "figures": [ up to 4 items {"t_s": <number or null>, "metric": "blood_pct" | "peak_blood_pct" | "median_blood_pct" | "refused_frames" | "usable_frames", "value": <number>} naming the figures you quoted ]}"""


def _call(task: str, dig: dict[str, Any], *, effort: str, max_tokens: int) -> tuple[dict[str, Any], dict[str, Any]]:
    msgs = [{"role": "system", "content": RULES},
            {"role": "user", "content": task + "\n\nMEASURED EVIDENCE (JSON):\n" + json.dumps(dig, separators=(",", ":"))}]
    t0 = time.time()
    try:
        out = nemotron.think(msgs, model=MODEL, json_out=True, temperature=0.2,
                             max_tokens=max_tokens, reasoning_effort=effort)
    except SystemExit as e:                       # nemotron.py exits when no key is set
        raise RuntimeError(str(e)) from e
    return out, {"model": MODEL, "model_label": MODEL_LABEL, "reasoning_effort": effort,
                 "seconds": round(time.time() - t0, 1)}


# ─────────────────────────────── grounding ───────────────────────────────

def _row_at(dig: dict[str, Any], t: float | None) -> dict[str, Any] | None:
    rows = dig["series"]
    if t is None or not rows:
        return None
    r = min(rows, key=lambda x: abs(x["t_s"] - t))
    return r if abs(r["t_s"] - t) <= 2.0 else None


def _phase_at(dig: dict[str, Any], t: float) -> str | None:
    for p in dig["phase_track"]["spans"]:
        if p["start_s"] - 1.0 <= t <= p["end_s"] + 1.0:
            return p["phase"]
    return None


def ground_item(dig: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    """Check one cited figure against the digest. Returns the item with `verified` and `measured`."""
    metric, value, t = item.get("metric"), item.get("value"), item.get("t_s")
    ok, measured = False, None
    try:
        if metric in ("blood_pct", "rate_pct_per_min", "instruments"):
            row = _row_at(dig, t)
            key = {"blood_pct": "blood_pct", "rate_pct_per_min": "rate_pct_per_min",
                   "instruments": "instruments"}[metric]
            if row is not None:
                measured = row[key]
                ok = abs(float(value) - float(measured)) <= max(0.35, 0.12 * abs(float(measured)))
        elif metric in ("peak_blood_pct", "median_blood_pct"):
            measured = dig["blood_covered_field_pct"]["peak" if metric == "peak_blood_pct" else "median"]
            ok = measured is not None and abs(float(value) - measured) <= max(0.2, 0.08 * measured)
            if not ok and metric == "peak_blood_pct":   # the smoothed maximum is also on screen
                alt = dig["blood_covered_field_pct"]["peak_smoothed"]
                if alt is not None and abs(float(value) - alt) <= max(0.1, 0.03 * alt):
                    ok, measured = True, alt
        elif metric == "phase":
            near = [p["phase"] for p in dig["phase_track"]["spans"]
                    if p["start_s"] - 1.0 <= float(t) <= p["end_s"] + 1.0]
            ok = str(value).lower() in near
            measured = str(value).lower() if ok else (near[0] if near else None)
        elif metric == "refusal":
            for r in dig["refusal_events"]:
                if r["start_s"] - 1.5 <= float(t) <= r["end_s"] + 1.5 and str(value).upper() == r["code"]:
                    ok, measured = True, r["code"]
                    break
            measured = measured or "no such refusal at that time"
        elif metric in ("refused_frames", "usable_frames"):
            measured = (dig["clip"]["frames_usable"] if metric == "usable_frames"
                        else (dig["clip"]["frames_read"] or 0) - (dig["clip"]["frames_usable"] or 0))
            ok = measured is not None and abs(float(value) - measured) <= max(2, 0.03 * measured)
    except (TypeError, ValueError):
        ok = False
    return {**item, "verified": bool(ok), "measured": measured}


# ─────────────────────────────── public API ───────────────────────────────

def reason(record: dict[str, Any], procedure: str = "", *, which: tuple[str, ...] = ("checkpoint", "explain", "summary")) -> dict[str, Any]:
    """Run the Nemotron calls for one measured case. Live: this spends tokens."""
    dig = digest(record, procedure)
    out: dict[str, Any] = {"digest_rows": len(dig["series"]), "model": MODEL, "model_label": MODEL_LABEL,
                           "made_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "calls": {}}
    with _lock:   # one live run at a time: the key's balance is the shared resource
        if "checkpoint" in which:
            for attempt in range(3):      # a reasoning model occasionally drops a key; ask again
                cp, meta = _call(CHECKPOINT_TASK, dig, effort="high", max_tokens=14000)
                if all(cp.get(k) for k in ("decision", "headline", "reasoning", "limits", "would_change_if")):
                    break
            allowed = {"blood_pct", "rate_pct_per_min", "phase", "refusal", "instruments"}
            cp["evidence"] = [ground_item(dig, e) for e in (cp.get("evidence") or [])
                              if e.get("metric") in allowed]
            if cp.get("decision") == "raise" and cp.get("at_s") is None:
                cp["at_s"] = dig["blood_covered_field_pct"]["peak_smoothed_at_s"]
            out["checkpoint"] = cp
            out["calls"]["checkpoint"] = meta
        if "explain" in which:
            ids = (["CLIP"] if dig["clip_level_refusal"] else []) + [r["id"] for r in dig["refusal_events"]]
            legend = {"CLIP": "the clip-level refusal", "ONSET": "why bleeding onset did not fire",
                      "VOLUME": "why there is no millilitre figure"}
            if not dig["bleeding_onset"]["fired"]:
                ids.append("ONSET")
            if dig["volume_ml"]["status"] != "MEASURED":
                ids.append("VOLUME")
            task = EXPLAIN_TASK.format(ids="; ".join(f"{i} = {legend.get(i, 'a run of refused frames, see refusal_events')}" for i in ids))
            ex, meta = _call(task, dig, effort="low", max_tokens=9000)
            out["explanations"] = ex.get("explanations", [])
            out["calls"]["explain"] = meta
        if "summary" in which:
            cpd = out.get("checkpoint") or {}
            sm, meta = _call(SUMMARY_TASK, {**dig, "scopewatch_checkpoint_decision":
                                            {"decision": cpd.get("decision"), "headline": cpd.get("headline")}},
                             effort="low", max_tokens=6000)
            sm["figures"] = [ground_item(dig, f) for f in (sm.get("figures") or [])]
            out["summary"] = sm
            out["calls"]["summary"] = meta
    out["refusal_events"] = dig["refusal_events"]
    return out


def cached(slug: str) -> dict[str, Any] | None:
    p = CACHE / f"{slug}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def save(slug: str, result: dict[str, Any]) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"{slug}.json").write_text(json.dumps(result, indent=1))


