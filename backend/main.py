"""Scopewatch on Nebius: the unchanged OpenCV engine, with Nemotron reasoning over what it measured.

One FastAPI app on one port serves the web UI and the API.

  * The engine (`scopewatch`, `visioncore`, `servicekit`) is the shipped OpenCV 5
    measurement code, copied in without edits. Uploads still go through it live.
  * The five bundled cases were measured once by that engine (`build_cases.py`) and are
    served from `cache/cases/`. Nemotron's reasoning over them is in `cache/reasoning/`.
  * A live Nemotron call happens only when someone presses "Run it" (`POST .../reason`).
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI
from fastapi.responses import FileResponse
from servicekit import ProductInfo, ServiceConfig, ServiceError, create_app
from servicekit.jobs import DONE, Job

import cases as case_meta
import reasoning
from scopewatch import __version__
from scopewatch.agent import ACTION_HOLD, HELD, Checkpoint, Observation
from scopewatch.service import PARAMS_SCHEMA, REGISTRY, analyze

HERE = Path(__file__).resolve().parent
WEB_DIR = HERE.parent / "web"
CASES_DIR = HERE / "cache" / "cases"
LIVE_DIR = Path(os.environ.get("SCOPEWATCH_LIVE_DIR", "/tmp/scopewatch-live"))
DATA_DIR = HERE / "data"
UPLOAD_DIR = Path(os.environ.get("OPENCV26_UPLOAD_DIR", "/tmp/scopewatch-uploads"))
os.environ.setdefault("OPENCV26_YOLOX_URI", str(HERE / "models" / "yolox_tiny.onnx"))
os.environ.setdefault("SCOPEWATCH_MODEL_DIR", str(HERE / "models"))

# Live Nemotron calls spend a shared balance. Cap them.
LIVE_PER_HOUR = int(os.environ.get("SCOPEWATCH_LIVE_PER_HOUR", 12))
_live_times: list[float] = []
_live_guard = threading.Lock()

# What the Nemotron layer last said about each job (cached or live).
REASONING: dict[str, dict[str, Any]] = {}
PROCEDURE: dict[str, str] = {}

PRODUCT = ProductInfo(
    slug="scopewatch",
    title="Scopewatch",
    tagline="Measures the operating field instead of leaving it to the eye.",
    description=("OpenCV measures how much of the laparoscopic field is covered in blood. "
                 "NVIDIA Nemotron reads those measurements and decides, with its reasons, "
                 "whether a safety checkpoint belongs before the irreversible step."),
    accent="#C2410C",
    version=__version__,
    repo_url="",
)


# ───────────────────────── bundled cases ─────────────────────────

def _clean_record(rec: dict[str, Any], title: str) -> dict[str, Any]:
    v = (rec.get("metrics") or {}).get("video") or {}
    v.pop("path", None)                       # an absolute path on the build machine
    rec.setdefault("input", {})["title"] = title
    return rec


def _apply_nemotron_checkpoint(slug: str, reasoned: dict[str, Any]) -> None:
    """Turn Nemotron's 'raise' into a held checkpoint in the engine's own agent loop.

    The engine's automatic cue is off. This is the replacement: the model decides, the loop
    records it as a transition by actor `nemotron`, and only a named person can resolve it.
    """
    loop = REGISTRY.get(slug)
    cp = reasoned.get("checkpoint") or {}
    if cp.get("decision") != "raise" or cp.get("at_s") is None:
        return
    rec = JOBS_RECORD[slug]
    times = rec["metrics"]["series"]["times_ms"]
    at_ms = float(cp["at_s"]) * 1000
    idx = min(range(len(times)), key=lambda i: abs(times[i] - at_ms)) if times else 0
    ev = min((e for e in rec["evidence"] if e.get("timestamp_ms") is not None),
             key=lambda e: abs(e["timestamp_ms"] - at_ms), default=None)
    phase = "dissection"
    for sp in rec["metrics"]["phases"].get("spans", []):
        if sp["start_ms"] - 400 <= at_ms <= sp["end_ms"] + 400:
            phase = sp["phase"]
    obs = Observation(index=idx, timestamp_ms=at_ms, phase=phase, measurable=True,
                      evidence_uri=ev["uri"] if ev else None)
    loop.checkpoints = [c for c in loop.checkpoints if not c.checkpoint_id.startswith("nm")]
    chk = Checkpoint(
        checkpoint_id="nm" + uuid.uuid4().hex[:8], at_ms=at_ms, frame_index=idx, phase=phase,
        question=(cp.get("headline") or "Nemotron advises a safety checkpoint here.")
        + " Confirm the critical view of safety is established, or proceed with a reason.",
        evidence_uri=obs.evidence_uri)
    loop.checkpoints.append(chk)
    loop._transition(HELD, obs, trigger="Nemotron decision: raise a checkpoint", actor="nemotron",
                     reason=cp.get("headline", ""))
    loop._act(ACTION_HOLD, obs, checkpoint_id=chk.checkpoint_id, decided_by="nvidia/nemotron-3-super")


JOBS_RECORD: dict[str, dict[str, Any]] = {}


def _restore_loop(slug: str) -> None:
    REGISTRY.put(slug, pickle.loads((CASES_DIR / slug / "loop.pkl").read_bytes()))


def _register_cases(app: FastAPI) -> None:
    store = app.state.store
    for meta in case_meta.CASES:
        slug = meta["slug"]
        folder = CASES_DIR / slug
        if not (folder / "record.json").is_file():
            continue
        rec = _clean_record(json.loads((folder / "record.json").read_text()), meta["title"])
        JOBS_RECORD[slug] = rec
        dest = UPLOAD_DIR / slug
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(folder / "evidence", dest)
        job = Job(job_id=slug, filename=rec["input"]["filename"], params={}, input_path=folder / "clip.mp4",
                  status=DONE, result=rec, started_at=time.time(), finished_at=time.time())
        store._jobs[slug] = job
        PROCEDURE[slug] = meta["procedure"]
        _restore_loop(slug)
        cached = reasoning.cached(slug)
        if cached:
            REASONING[slug] = cached
            _apply_nemotron_checkpoint(slug, cached)
    pinned = {m["slug"] for m in case_meta.CASES}
    original_drop = store._drop
    store._drop = lambda job: None if job.job_id in pinned else original_drop(job)   # never evict a bundled case


# ───────────────────────── app ─────────────────────────

def build_app() -> FastAPI:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    config = ServiceConfig(
        product=PRODUCT, upload_dir=UPLOAD_DIR, static_dir=WEB_DIR, params_schema=PARAMS_SCHEMA,
        job_ttl_seconds=10**9, max_jobs=200,
        max_concurrent_jobs=int(os.environ.get("OPENCV26_MAX_CONCURRENT_JOBS", 2)))
    app = create_app(config, analyze)
    _register_cases(app)
    _routes(app)
    return app


def _job_record(app: FastAPI, job_id: str) -> dict[str, Any]:
    job = app.state.store.get(job_id)
    if job.result is None:
        raise ServiceError("NOT_FOUND", "that job has no result yet")
    return job.result


def _routes(app: FastAPI) -> None:
    @app.get("/api/cases")
    async def list_cases() -> dict[str, Any]:
        out = []
        for meta in case_meta.CASES:
            slug = meta["slug"]
            rec = JOBS_RECORD.get(slug)
            if not rec:
                continue
            v = rec["metrics"]["video"]
            cp = (REASONING.get(slug) or {}).get("checkpoint") or {}
            out.append({**{k: meta[k] for k in ("slug", "title", "blurb", "credit", "licence", "kind")},
                        "duration_ms": v.get("duration_ms"), "refused": bool(rec.get("refused")),
                        "decision": cp.get("decision"), "decision_at_s": cp.get("at_s")})
        return {"cases": out, "live_available": bool(_have_key())}

    @app.get("/api/cases/{slug}/clip.mp4", include_in_schema=False)
    async def clip(slug: str) -> FileResponse:
        path = CASES_DIR / Path(slug).name / "clip.mp4"
        if not path.is_file():
            raise ServiceError("NOT_FOUND", f"no clip for {slug}")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/jobs/{job_id}/checkpoints")
    async def checkpoints(job_id: str) -> dict[str, Any]:
        return REGISTRY.get(job_id).to_dict()

    @app.post("/api/jobs/{job_id}/checkpoints/{checkpoint_id}/confirm")
    async def confirm(job_id: str, checkpoint_id: str, payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
        actor = str(payload.get("actor") or "").strip()
        if not actor:
            raise ServiceError("BAD_REQUEST", "a checkpoint is confirmed by a named person, so `actor` is required")
        loop = REGISTRY.get(job_id)
        try:
            cp = loop.confirm(checkpoint_id, actor, str(payload.get("note") or ""))
        except KeyError as exc:
            raise ServiceError("NOT_FOUND", str(exc)) from exc
        return {"checkpoint": cp.to_dict(), "agent": loop.to_dict()}

    @app.post("/api/jobs/{job_id}/checkpoints/{checkpoint_id}/dismiss")
    async def dismiss(job_id: str, checkpoint_id: str, payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
        from scopewatch.config import DISMISSAL_REASONS
        actor = str(payload.get("actor") or "").strip()
        reason = str(payload.get("reason") or "").strip()
        if not actor:
            raise ServiceError("BAD_REQUEST", "`actor` is required")
        if not reason:
            raise ServiceError("BAD_REQUEST", "a checkpoint may only be dismissed with a reason",
                               reasons_offered=list(DISMISSAL_REASONS))
        loop = REGISTRY.get(job_id)
        try:
            cp = loop.dismiss(checkpoint_id, actor, reason, str(payload.get("note") or ""))
        except KeyError as exc:
            raise ServiceError("NOT_FOUND", str(exc)) from exc
        return {"checkpoint": cp.to_dict(), "agent": loop.to_dict()}

    @app.get("/api/evaluation")
    async def evaluation() -> dict[str, Any]:
        """The measured error bars, served next to the numbers they qualify."""
        data = json.loads((DATA_DIR / "evaluation.json").read_text(encoding="utf-8"))
        return data

    @app.get("/api/jobs/{job_id}/reasoning")
    async def get_reasoning(job_id: str) -> dict[str, Any]:
        """What Nemotron said about this job. Cached for the bundled cases; else not yet run."""
        r = REASONING.get(job_id)
        return {"reasoning": r, "source": ("cache" if r and not r.get("live") else "live" if r else None),
                "procedure": PROCEDURE.get(job_id, ""), "live_available": bool(_have_key())}

    @app.post("/api/jobs/{job_id}/reason")
    async def run_reasoning(job_id: str, payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
        """The one explicit 'Run it': a live Nemotron call over this job's measurements."""
        import anyio
        if not _have_key():
            raise ServiceError("NO_MODEL_KEY", "this deployment has no Nebius Token Factory key, "
                               "so a live run is not available; the cached reasoning is still shown")
        rec = _job_record(app, job_id)
        if not _allow_live():
            raise ServiceError("RATE_LIMITED", f"live runs are capped at {LIVE_PER_HOUR} an hour on this deployment")
        procedure = str(payload.get("procedure") or PROCEDURE.get(job_id) or "").strip()[:600]
        PROCEDURE[job_id] = procedure
        try:
            result = await anyio.to_thread.run_sync(lambda: reasoning.reason(rec, procedure))
        except Exception as exc:
            raise ServiceError("MODEL_FAILED", f"Nemotron did not answer: {exc}") from exc
        result["live"] = True
        REASONING[job_id] = result
        try:  # a failed cache write must never break the live run (read-only fs in prod)
            LIVE_DIR.mkdir(parents=True, exist_ok=True)
            (LIVE_DIR / f"{job_id}-{int(time.time())}.json").write_text(json.dumps(result, indent=1))
        except OSError as exc:
            import logging
            logging.getLogger("scopewatch").warning("could not save live reasoning to %s: %s", LIVE_DIR, exc)
        _apply_live(job_id, rec, result)
        return {"reasoning": result, "source": "live", "procedure": procedure,
                "agent": REGISTRY.get(job_id).to_dict()}


def _apply_live(job_id: str, rec: dict[str, Any], result: dict[str, Any]) -> None:
    JOBS_RECORD[job_id] = rec
    bundled = job_id in {m["slug"] for m in case_meta.CASES}
    if bundled:
        _restore_loop(job_id)         # drop any earlier model checkpoint; humans' answers on it reset too
    _apply_nemotron_checkpoint(job_id, result)


def _have_key() -> bool:
    try:
        return bool(nemotron_keys())
    except Exception:
        return False


def nemotron_keys() -> list[str]:
    import nemotron
    try:
        return nemotron._pool()
    except SystemExit:
        return []


def _allow_live() -> bool:
    with _live_guard:
        now = time.time()
        _live_times[:] = [t for t in _live_times if now - t < 3600]
        if len(_live_times) >= LIVE_PER_HOUR:
            return False
        _live_times.append(now)
        return True


app = build_app()
