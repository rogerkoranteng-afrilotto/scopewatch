"""The shared FastAPI app factory.

A product's whole service is:

    from servicekit import ProductInfo, ServiceConfig, create_app

    def analyze(ctx):
        ctx.progress(10, "decoding")
        ...
        return ctx.record

    app = create_app(ServiceConfig(product=ProductInfo(slug="scopewatch", ...)), analyze)

Routes it gets for free:
    GET  /                       the UI shell
    GET  /healthz                liveness, for App Runner
    GET  /version                git sha, OpenCV version, model versions
    POST /api/jobs               multipart upload, returns a job id
    GET  /api/jobs/{id}          the RunRecord as JSON
    GET  /api/jobs/{id}/events   Server-Sent Events progress stream
    GET  /api/jobs/{id}/evidence/{name}   the frames behind each result
    GET  /api/config             what the UI needs to render itself
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from visioncore import __version__ as visioncore_version
from visioncore import environment

from .config import ProductInfo, ServiceConfig
from .errors import ServiceError, error_response
from .jobs import Analyzer, JobStore
from .logs import configure_logging, extra, get_logger, request_id_var

SHELL_DIR = Path(__file__).parent / "static"
log = get_logger("servicekit.app")

HEARTBEAT_SECONDS = 15.0


def create_app(
    config: ServiceConfig,
    analyzer: Analyzer,
    *,
    log_level: str = "INFO",
) -> FastAPI:
    configure_logging(log_level)
    product = config.product

    app = FastAPI(
        title=product.title,
        version=product.version,
        description=product.description or product.tagline,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.config = config
    app.state.store = JobStore(
        analyzer,
        config.upload_dir,
        product=product.slug,
        max_jobs=config.max_jobs,
        max_concurrent=config.max_concurrent_jobs,
        ttl_seconds=config.job_ttl_seconds,
    )

    if config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(config.cors_origins),
            allow_methods=["*"],
            allow_headers=["*"],
        )

    _install_middleware(app)
    _install_error_handlers(app)
    _install_routes(app, config)
    _install_static(app, config)
    return app


# ---------------------------------------------------------------------------


def _install_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        rid = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-Id"] = rid
        return response


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def _service_error(request: Request, exc: ServiceError):
        return error_response(request, exc)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        return error_response(
            request, ServiceError("BAD_REQUEST", "invalid request", fields=exc.errors())
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        rid = getattr(request.state, "request_id", "")
        log.error("unhandled error", extra=extra(path=request.url.path), exc_info=exc)
        return JSONResponse(
            status_code=500,
            content=ServiceError("INTERNAL", "internal error").payload(rid),
        )


def _install_routes(app: FastAPI, config: ServiceConfig) -> None:
    product = config.product

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, Any]:
        return {"status": "ok", "product": product.slug}

    @app.get("/version")
    async def version() -> dict[str, Any]:
        env = environment()
        return {
            "product": product.to_dict(),
            "git_sha": env.git_sha,
            "opencv_version": env.opencv_version,
            "opencv_build": {
                "custom_hal": env.cpu.get("custom_hal"),
                "baseline": env.cpu.get("baseline"),
                "kleidicv": env.cpu.get("kleidicv"),
                "ipp": env.cpu.get("ipp"),
                "threads": env.cpu.get("threads"),
            },
            "numpy_version": env.numpy_version,
            "python_version": env.python_version,
            "platform": env.platform,
            "machine": env.machine,
            "visioncore_version": visioncore_version,
            "models": {
                name: (spec.to_dict() if hasattr(spec, "to_dict") else spec)
                for name, spec in config.models.items()
            },
        }

    @app.get("/api/config")
    async def ui_config() -> dict[str, Any]:
        return {
            "product": product.to_dict(),
            "accent": product.accent,
            "params": config.params_schema,
            "accepts": list(config.allowed_suffixes),
            "max_upload_bytes": config.max_upload_bytes,
            "opencv_version": cv2.__version__,
        }

    @app.post("/api/jobs", status_code=202)
    async def create_job(
        request: Request,
        file: UploadFile = File(...),
        params: str = Form("{}"),
    ) -> dict[str, Any]:
        data = await _read_upload(file, config)
        try:
            parsed = json.loads(params) if params else {}
        except json.JSONDecodeError as exc:
            raise ServiceError("BAD_REQUEST", f"params is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ServiceError("BAD_REQUEST", "params must be a JSON object")

        store: JobStore = request.app.state.store
        job = store.create(file.filename or "upload", data, parsed)
        store.start(job)
        log.info(
            "job accepted",
            extra=extra(job_id=job.job_id, upload_name=job.filename, bytes=len(data)),
        )
        return {"job_id": job.job_id, "status": job.status,
                "events_url": f"/api/jobs/{job.job_id}/events"}

    @app.get("/api/jobs")
    async def list_jobs(request: Request) -> dict[str, Any]:
        store: JobStore = request.app.state.store
        return {"jobs": [j.to_dict() for j in store.list()]}

    @app.get("/api/jobs/{job_id}")
    async def get_job(request: Request, job_id: str) -> dict[str, Any]:
        store: JobStore = request.app.state.store
        return store.get(job_id).to_dict()

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(request: Request, job_id: str) -> StreamingResponse:
        store: JobStore = request.app.state.store
        job = store.get(job_id)

        async def stream() -> AsyncIterator[bytes]:
            queue = store.subscribe(job)
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
                    except TimeoutError:
                        yield b": heartbeat\n\n"
                        continue
                    yield f"event: {event.get('type', 'message')}\n".encode()
                    yield f"data: {json.dumps(event, default=str)}\n\n".encode()
                    if event.get("type") == "end":
                        break
            finally:
                store.unsubscribe(job, queue)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.get("/api/jobs/{job_id}/evidence/{name}")
    async def job_evidence(request: Request, job_id: str, name: str) -> FileResponse:
        store: JobStore = request.app.state.store
        job = store.get(job_id)
        safe = Path(name).name
        path = store.upload_dir / job.job_id / safe
        if not path.is_file():
            raise ServiceError("NOT_FOUND", f"no evidence {safe} for job {job_id}")
        return FileResponse(path)


async def _read_upload(file: UploadFile, config: ServiceConfig) -> bytes:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in config.allowed_suffixes:
        raise ServiceError(
            "UNSUPPORTED_MEDIA",
            f"{suffix or 'that file type'} is not accepted",
            accepted=list(config.allowed_suffixes),
        )
    chunks, total = [], 0
    while chunk := await file.read(1 << 20):
        total += len(chunk)
        if total > config.max_upload_bytes:
            raise ServiceError(
                "TOO_LARGE",
                f"upload exceeds {config.max_upload_bytes // (1024 * 1024)} MB",
                max_bytes=config.max_upload_bytes,
            )
        chunks.append(chunk)
    if total == 0:
        raise ServiceError("BAD_REQUEST", "the uploaded file is empty")
    return b"".join(chunks)


def _install_static(app: FastAPI, config: ServiceConfig) -> None:
    """Shell assets at /shell, a product's own overrides at /assets, index at /."""
    app.mount("/shell", StaticFiles(directory=SHELL_DIR), name="shell")
    if config.static_dir and Path(config.static_dir).is_dir():
        app.mount("/assets", StaticFiles(directory=Path(config.static_dir)), name="assets")

    @app.get("/", include_in_schema=False, response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        custom = Path(config.static_dir) / "index.html" if config.static_dir else None
        path = custom if custom and custom.is_file() else SHELL_DIR / "index.html"
        html = path.read_text(encoding="utf-8")
        return HTMLResponse(
            html.replace("{{PRODUCT_TITLE}}", config.product.title)
            .replace("{{PRODUCT_TAGLINE}}", config.product.tagline)
            .replace("{{PRODUCT_ACCENT}}", config.product.accent)
        )


def make_product(slug: str, title: str, **kwargs: Any) -> ProductInfo:
    """Small convenience so a product's main.py is three lines."""
    return ProductInfo(slug=slug, title=title, **kwargs)
