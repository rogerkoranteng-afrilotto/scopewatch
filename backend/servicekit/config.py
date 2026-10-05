"""Per-product service configuration. One dataclass, no environment sprawl."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_UPLOAD_BYTES = int(os.environ.get("OPENCV26_MAX_UPLOAD_BYTES", 200 * 1024 * 1024))

# Deliberately narrow. Anything not on this list is rejected before it reaches cv2.
ALLOWED_SUFFIXES: tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
    ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm",
)


@dataclass
class ProductInfo:
    """Who this service is, for /version, the page title and the RunRecord."""

    slug: str
    title: str
    tagline: str = ""
    description: str = ""
    accent: str = "#4f8cff"  # each product overrides this; see static/css/theme.css
    version: str = "0.1.0"
    repo_url: str = ""
    docs_url: str = "/docs"

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "title": self.title,
            "tagline": self.tagline,
            "description": self.description,
            "version": self.version,
            "repo_url": self.repo_url,
        }


@dataclass
class ServiceConfig:
    """Everything the shell needs. Products override the fields they care about."""

    product: ProductInfo
    upload_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("OPENCV26_UPLOAD_DIR", "/tmp/opencv26-uploads"))
    )
    max_upload_bytes: int = MAX_UPLOAD_BYTES
    allowed_suffixes: tuple[str, ...] = ALLOWED_SUFFIXES
    job_ttl_seconds: int = int(os.environ.get("OPENCV26_JOB_TTL", 3600))
    max_jobs: int = int(os.environ.get("OPENCV26_MAX_JOBS", 64))
    max_concurrent_jobs: int = int(os.environ.get("OPENCV26_MAX_CONCURRENT_JOBS", 2))
    cors_origins: tuple[str, ...] = ()
    models: dict[str, Any] = field(default_factory=dict)
    static_dir: Path | None = None  # a product's own overrides, layered over the shell
    params_schema: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.upload_dir = Path(self.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
