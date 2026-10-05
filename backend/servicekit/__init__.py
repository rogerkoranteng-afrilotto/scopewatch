"""servicekit - the shared FastAPI service and UI shell for the opencv26 entries."""

from __future__ import annotations

from .app import SHELL_DIR, create_app, make_product
from .config import ALLOWED_SUFFIXES, ProductInfo, ServiceConfig
from .errors import ServiceError
from .jobs import Analyzer, Job, JobContext, JobStore
from .logs import configure_logging, get_logger

__version__ = "0.1.0"

__all__ = [
    "ALLOWED_SUFFIXES",
    "SHELL_DIR",
    "Analyzer",
    "Job",
    "JobContext",
    "JobStore",
    "ProductInfo",
    "ServiceConfig",
    "ServiceError",
    "__version__",
    "configure_logging",
    "create_app",
    "get_logger",
    "make_product",
]
