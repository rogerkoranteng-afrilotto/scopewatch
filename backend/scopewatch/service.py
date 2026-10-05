"""The Scopewatch service: servicekit's shell, plus the two routes a checkpoint needs.

Everything except the checkpoint comes from `servicekit.create_app` unchanged - the
upload, the job queue, the Server-Sent Events progress stream, the JSON result and
the evidence endpoint are all the shared shell, and the shell is not edited.

What this file adds is the part a shared shell cannot know about: a held checkpoint
is a piece of state that outlives the analysis and is resolved by a person. So the
loop for each job is kept in a small registry, and two routes let a named human
confirm it or dismiss it with a reason. Both write into the same `AgentLoop`, so
the decision lands in the same transition log as the perception that raised it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, Request
from servicekit import JobContext, ProductInfo, ServiceConfig, ServiceError, create_app
from visioncore import Evidence, RunRecord, YoloxDetector

from . import __version__
from .agent import AgentLoop
from .config import DISMISSAL_REASONS, PipelineParams
from .pipeline import CaseResult, analyse_case, to_record

HERE = Path(__file__).resolve().parent
PRODUCT_ROOT = HERE.parent.parent
WEB_DIR = PRODUCT_ROOT / "web"
MEDIA_DIR = PRODUCT_ROOT / "media"
SAMPLE_VIDEO = MEDIA_DIR / "sample-case.mp4"

# Set OPENCV26_YOLOX_URI to a local path to use a model baked into the image.
MODEL_DIR = Path(os.environ.get("SCOPEWATCH_MODEL_DIR", PRODUCT_ROOT / "models"))


# ---------------------------------------------------------------------------
# The checkpoint registry
# ---------------------------------------------------------------------------


class CheckpointRegistry:
    """Job id to agent loop. In-process, which is the right scope for one instance.

    A held checkpoint is deliberately *not* persisted across a restart. If the
    service goes down mid-case, the honest state to come back in is "nothing was
    decided", not a checkpoint of unknown age that appears to have been reviewed.
    """

    def __init__(self, limit: int = 128) -> None:
        self._loops: dict[str, AgentLoop] = {}
        self._order: list[str] = []
        self.limit = limit

    def put(self, job_id: str, loop: AgentLoop) -> None:
        self._loops[job_id] = loop
        self._order.append(job_id)
        while len(self._order) > self.limit:
            self._loops.pop(self._order.pop(0), None)

    def get(self, job_id: str) -> AgentLoop:
        loop = self._loops.get(job_id)
        if loop is None:
            raise ServiceError("NOT_FOUND", f"no agent loop for job {job_id}", job_id=job_id)
        return loop


REGISTRY = CheckpointRegistry()


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------

_detector: YoloxDetector | None = None
_detector_error: str | None = None


def detector() -> YoloxDetector | None:
    """YOLOX-tiny, loaded once. A missing model degrades the run; it never fails it.

    The classical shaft channel is what the measurement depends on. The DNN adds a
    second opinion and a `cv2.dnn` latency figure for the report. If the ONNX file
    is not in the image, the run says so in a warning and carries on.
    """
    global _detector, _detector_error
    if _detector is not None or _detector_error is not None:
        return _detector
    try:
        _detector = YoloxDetector(cache_dir=MODEL_DIR)
    except Exception as exc:
        _detector_error = str(exc)
        _detector = None
    return _detector


# ---------------------------------------------------------------------------
# The analyzer
# ---------------------------------------------------------------------------


def _params_from(raw: dict[str, Any]) -> PipelineParams:
    params = PipelineParams()
    for key in (
        "stride",
        "max_side",
        "max_frames",
        "shaft_mm",
        "mm_per_px",
        "film_depth_mm",
        "onset_rate_ml_per_min",
        "onset_rate_pct_per_min",
    ):
        if raw.get(key) not in (None, ""):
            setattr(params, key, type(getattr(params, key) or 0.0)(raw[key]))
    for key in ("use_dnn", "rescan", "safety_view_established", "auto_checkpoint"):
        if key in raw:
            value = raw[key]
            setattr(params, key, value if isinstance(value, bool) else str(value).lower() in
                    ("1", "true", "yes", "on"))
    params.film_depth_low_mm = min(params.film_depth_low_mm, params.film_depth_mm)
    params.film_depth_high_mm = max(params.film_depth_high_mm, params.film_depth_mm)
    return params


def analyze(ctx: JobContext) -> RunRecord:
    """The one function servicekit needs. Runs in a worker thread, never on the loop."""
    params = _params_from(ctx.params)
    ctx.progress(4.0, "reading the clip")

    def save(name: str, data: bytes, meta: dict[str, Any]) -> str | None:
        uri = ctx.save_evidence(f"{name}.jpg", data)
        ctx.record.add_evidence(
            Evidence(
                label=name,
                kind="overlay",
                uri=uri,
                frame_index=meta.get("frame_index"),
                timestamp_ms=meta.get("timestamp_ms"),
                caption=meta.get("caption", ""),
                metrics=meta.get("metrics", {}),
            )
        )
        return uri

    result: CaseResult = analyse_case(
        ctx.input_path,
        params=params,
        detector=detector() if params.use_dnn else None,
        progress=lambda pct, msg: ctx.progress(pct, msg),
        save_evidence=save,
    )

    if params.use_dnn and detector() is None and _detector_error:
        ctx.record.warn(
            "the YOLOX-tiny ONNX model was not available, so the DNN channel did not "
            f"run; the classical instrument channel did ({_detector_error})"
        )

    ctx.progress(92.0, "fitting the series and the phase track")
    to_record(result, ctx.record, params)
    REGISTRY.put(ctx.job_id, result.loop)

    checkpoint = result.loop.open_checkpoint
    if checkpoint:
        ctx.note(
            "a safety checkpoint is held and needs a person: "
            f"{checkpoint.checkpoint_id} at {checkpoint.at_ms / 1000:.1f} s"
        )
    ctx.progress(100.0, "done")
    return ctx.record


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------

PRODUCT = ProductInfo(
    slug="scopewatch",
    title="Scopewatch",
    tagline="Measures the operating field instead of leaving it to the eye.",
    description=(
        "Scopewatch watches the laparoscopic camera that is already in the room. It "
        "measures how much of the visible field is covered in blood and how fast that "
        "share is changing, timestamps when the change crosses a threshold, and refuses "
        "frames it cannot read, including video that is not laparoscopic. A volume in "
        "millilitres is shown only when the instrument-shaft scale holds steady across "
        "the case, and it has never been validated on real footage. It is a "
        "retrospective measurement instrument, not a medical device."
    ),
    accent="#FF8A3D",
    version=__version__,
    repo_url="https://github.com/rogerkorantenng/scopewatch",
)

PARAMS_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "stride",
        "label": "Frame stride",
        "type": "number",
        "default": 5,
        "min": 1,
        "max": 30,
        "help": "Keep one frame in this many on the first pass. The agent re-reads "
                "densely around anything it finds.",
    },
    {
        "name": "shaft_mm",
        "label": "Instrument shaft diameter",
        "type": "number",
        "default": 5.0,
        "step": 0.5,
        "unit": "mm",
        "help": "The scale reference. Standard laparoscopic shafts are 5 mm.",
    },
    {
        "name": "film_depth_mm",
        "label": "Assumed film depth",
        "type": "number",
        "default": 2.0,
        "step": 0.5,
        "unit": "mm",
        "help": "A single camera cannot see depth. Volumes are reported as an "
                "interval over 1 to 3 mm around this value.",
    },
    {
        "name": "safety_view_established",
        "label": "Critical view of safety already recorded",
        "type": "boolean",
        "default": False,
        "help": "Tick this and no checkpoint is raised.",
    },
    {
        "name": "auto_checkpoint",
        "label": "Automatic safety checkpoint (experimental)",
        "type": "boolean",
        "default": False,
        "help": "Off by default. Its cue is instrument width, and on real video a "
                "grasper near the lens looks as wide as a clip applier.",
    },
    {
        "name": "use_dnn",
        "label": "Run the YOLOX channel",
        "type": "boolean",
        "default": True,
        "help": "YOLOX-tiny in cv2.dnn, Apache-2.0. COCO classes, so it names no "
                "surgical instrument; it is a second opinion and a latency figure.",
    },
]


def build_app() -> FastAPI:
    config = ServiceConfig(
        product=PRODUCT,
        static_dir=WEB_DIR if WEB_DIR.is_dir() else None,
        params_schema=PARAMS_SCHEMA,
        max_concurrent_jobs=int(os.environ.get("OPENCV26_MAX_CONCURRENT_JOBS", 2)),
    )
    app = create_app(config, analyze)
    _install_scopewatch_routes(app)
    return app


def _install_scopewatch_routes(app: FastAPI) -> None:
    from fastapi.responses import FileResponse

    @app.get("/api/sample")
    async def sample() -> dict[str, Any]:
        """What the judge clicks. A bundled clip, so a cold start needs no upload."""
        if not SAMPLE_VIDEO.is_file():
            raise ServiceError("NOT_FOUND", "no sample clip is bundled in this image")
        return {
            "url": "/api/sample/download",
            "filename": SAMPLE_VIDEO.name,
            "bytes": SAMPLE_VIDEO.stat().st_size,
            "description": (
                "Laparoscopic repair of a bleeding ulcer, 93 seconds. "
                "Di Saverio S et al., World J Emerg Surg 2014, CC BY 4.0."
            ),
        }

    @app.get("/api/sample/download", include_in_schema=False)
    async def sample_download() -> FileResponse:
        if not SAMPLE_VIDEO.is_file():
            raise ServiceError("NOT_FOUND", "no sample clip is bundled in this image")
        return FileResponse(SAMPLE_VIDEO, media_type="video/mp4", filename=SAMPLE_VIDEO.name)

    @app.get("/api/jobs/{job_id}/checkpoints")
    async def checkpoints(job_id: str) -> dict[str, Any]:
        loop = REGISTRY.get(job_id)
        return loop.to_dict()

    @app.post("/api/jobs/{job_id}/checkpoints/{checkpoint_id}/confirm")
    async def confirm(
        request: Request,
        job_id: str,
        checkpoint_id: str,
        payload: dict[str, Any] = Body(default_factory=dict),
    ) -> dict[str, Any]:
        """A named person records that the critical view of safety is established."""
        actor = str(payload.get("actor") or "").strip()
        if not actor:
            raise ServiceError(
                "BAD_REQUEST",
                "a checkpoint is confirmed by a named person, so `actor` is required",
            )
        loop = REGISTRY.get(job_id)
        try:
            cp = loop.confirm(checkpoint_id, actor, str(payload.get("note") or ""))
        except KeyError as exc:
            raise ServiceError("NOT_FOUND", str(exc)) from exc
        del request
        return {"checkpoint": cp.to_dict(), "agent": loop.to_dict()}

    @app.post("/api/jobs/{job_id}/checkpoints/{checkpoint_id}/dismiss")
    async def dismiss(
        job_id: str,
        checkpoint_id: str,
        payload: dict[str, Any] = Body(default_factory=dict),
    ) -> dict[str, Any]:
        """Dismissal needs a reason. There is no bare dismiss, by design."""
        actor = str(payload.get("actor") or "").strip()
        reason = str(payload.get("reason") or "").strip()
        if not actor:
            raise ServiceError("BAD_REQUEST", "`actor` is required")
        if not reason:
            raise ServiceError(
                "BAD_REQUEST",
                "a checkpoint may only be dismissed with a reason",
                reasons_offered=list(DISMISSAL_REASONS),
            )
        loop = REGISTRY.get(job_id)
        try:
            cp = loop.dismiss(checkpoint_id, actor, reason, str(payload.get("note") or ""))
        except KeyError as exc:
            raise ServiceError("NOT_FOUND", str(exc)) from exc
        return {"checkpoint": cp.to_dict(), "agent": loop.to_dict()}

    @app.get("/api/evaluation")
    async def evaluation() -> dict[str, Any]:
        """The measured error bars, served next to the numbers they qualify."""
        path = PRODUCT_ROOT / "docs" / "evaluation.json"
        if not path.is_file():
            raise ServiceError("NOT_FOUND", "evaluation results are not in this image")
        return json.loads(path.read_text(encoding="utf-8"))


app = build_app()
