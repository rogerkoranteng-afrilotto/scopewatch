"""Version and environment facts, surfaced in /version and in every RunRecord.

The competition requires OpenCV 5. `assert_opencv5()` is called at import time of
`visioncore` so a mis-resolved 4.x wheel fails loudly at startup rather than
silently producing a disqualifying entry.
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

REQUIRED_OPENCV = "5.0.0.93"
REQUIRED_OPENCV_MAJOR = 5


class OpenCVVersionError(RuntimeError):
    """Raised when the installed OpenCV is not the pinned major version."""


def assert_opencv5() -> None:
    major = int(cv2.__version__.split(".", 1)[0])
    if major != REQUIRED_OPENCV_MAJOR:
        raise OpenCVVersionError(
            f"opencv26 requires OpenCV {REQUIRED_OPENCV_MAJOR}.x "
            f"(pin opencv-python-headless=={REQUIRED_OPENCV}); "
            f"found {cv2.__version__}. An unpinned install resolves to 4.14.x."
        )


@lru_cache(maxsize=1)
def git_sha(short: bool = True) -> str:
    """Best-effort git sha of the working tree. 'unknown' outside a checkout."""
    env_sha = os.environ.get("OPENCV26_GIT_SHA") or os.environ.get("GIT_SHA")
    if env_sha:
        return env_sha[:7] if short else env_sha
    cmd = ["git", "rev-parse", "HEAD"]
    if short:
        cmd.insert(2, "--short")
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
            cwd=os.path.dirname(__file__),
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = out.stdout.strip()
    return sha or "unknown"


@lru_cache(maxsize=1)
def cpu_features() -> dict[str, object]:
    """The bits of getBuildInformation() that decide whether a benchmark means anything."""
    info = cv2.getBuildInformation()
    picked: dict[str, object] = {}
    for key in ("Custom HAL", "Baseline", "Dispatched", "Parallel"):
        for line in info.splitlines():
            stripped = line.strip()
            if stripped.startswith(key + ":"):
                picked[key.lower().replace(" ", "_")] = stripped.split(":", 1)[1].strip()
                break
    picked["kleidicv"] = "kleidicv" in info.lower()
    picked["ipp"] = "ipp" in info.lower()
    picked["threads"] = cv2.getNumThreads()
    picked["optimized"] = bool(cv2.useOptimized())
    return picked


@dataclass(frozen=True)
class Environment:
    """Everything a judge needs to reproduce a number."""

    opencv_version: str
    numpy_version: str
    python_version: str
    platform: str
    machine: str
    git_sha: str
    cpu: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "opencv_version": self.opencv_version,
            "numpy_version": self.numpy_version,
            "python_version": self.python_version,
            "platform": self.platform,
            "machine": self.machine,
            "git_sha": self.git_sha,
            "cpu": dict(self.cpu),
        }


@lru_cache(maxsize=1)
def environment() -> Environment:
    return Environment(
        opencv_version=cv2.__version__,
        numpy_version=np.__version__,
        python_version=platform.python_version(),
        platform=platform.platform(),
        machine=platform.machine(),
        git_sha=git_sha(),
        cpu=cpu_features(),
    )
