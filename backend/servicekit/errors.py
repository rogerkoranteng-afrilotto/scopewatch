"""One error shape, so the UI only ever has to render one thing.

    {"error": {"code": "...", "message": "...", "request_id": "...", "details": {...}}}
"""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

CODE_STATUS: dict[str, int] = {
    "BAD_REQUEST": 400,
    "UNSUPPORTED_MEDIA": 415,
    "TOO_LARGE": 413,
    "NOT_FOUND": 404,
    "BUSY": 429,
    "ANALYSIS_FAILED": 500,
    "INTERNAL": 500,
}


class ServiceError(Exception):
    """Anything the UI should render as a message rather than a stack trace."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int | None = None,
        **details: Any,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code or CODE_STATUS.get(code, 400)
        self.details = details

    def payload(self, request_id: str = "") -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "request_id": request_id,
                "details": self.details,
            }
        }


def error_response(request: Request, exc: ServiceError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "")
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.payload(request_id),
        headers={"X-Request-Id": request_id} if request_id else None,
    )
