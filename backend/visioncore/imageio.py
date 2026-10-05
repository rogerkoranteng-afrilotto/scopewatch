"""Image and video IO with frame iteration, timestamps and decimation.

OpenCV 5 gotcha baked in here: `VideoCapture.get()` returns **-1** for an
unsupported property where 4.x returned 0. Every read goes through `_prop()`,
which treats anything < 0 as missing.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PathLike = str | os.PathLike[str]


class DecodeError(ValueError):
    """The bytes or file handed in are not decodable as an image or video."""


@dataclass(frozen=True)
class Frame:
    """One frame plus the provenance a RunRecord needs to cite it."""

    index: int
    timestamp_ms: float
    image: np.ndarray
    source: str = ""

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.image.shape[0]), int(self.image.shape[1])

    def with_image(self, image: np.ndarray) -> Frame:
        return Frame(self.index, self.timestamp_ms, image, self.source)


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_ms: float
    fourcc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "frame_count": self.frame_count,
            "duration_ms": round(self.duration_ms, 1),
            "fourcc": self.fourcc,
        }


def _prop(cap: cv2.VideoCapture, prop: int, default: float = 0.0) -> float:
    """OpenCV 5 returns -1 (not 0) for unsupported properties."""
    try:
        value = float(cap.get(prop))
    except cv2.error:
        return default
    if value is None or value < 0 or math.isnan(value):
        return default
    return value


def _fourcc_str(value: float) -> str:
    code = int(value)
    if code <= 0:
        return ""
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")


def video_info(path: PathLike) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise DecodeError(f"cannot open video: {path}")
    try:
        width = int(_prop(cap, cv2.CAP_PROP_FRAME_WIDTH))
        height = int(_prop(cap, cv2.CAP_PROP_FRAME_HEIGHT))
        fps = _prop(cap, cv2.CAP_PROP_FPS)
        count = int(_prop(cap, cv2.CAP_PROP_FRAME_COUNT))
        fourcc = _fourcc_str(_prop(cap, cv2.CAP_PROP_FOURCC))
    finally:
        cap.release()
    duration = (count / fps * 1000.0) if (fps > 0 and count > 0) else 0.0
    return VideoInfo(str(path), width, height, fps, count, duration, fourcc)


def iter_video(
    path: PathLike,
    *,
    stride: int = 1,
    max_frames: int | None = None,
    start_ms: float = 0.0,
    end_ms: float | None = None,
    resize_to: tuple[int, int] | None = None,
    max_side: int | None = None,
) -> Iterator[Frame]:
    """Yield decimated frames with real timestamps.

    stride     keep 1 frame in `stride` (decimation; 1 = every frame)
    max_frames stop after this many *kept* frames
    start_ms   seek before decoding (falls back to skipping if seek unsupported)
    end_ms     stop once a frame's timestamp passes this
    resize_to  exact (width, height)
    max_side   scale down so the longest side is at most this, preserving aspect
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise DecodeError(f"cannot open video: {path}")
    fps = _prop(cap, cv2.CAP_PROP_FPS)
    seeked = False
    if start_ms > 0:
        seeked = bool(cap.set(cv2.CAP_PROP_POS_MSEC, float(start_ms)))
    kept = 0
    index = -1
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            index += 1
            pos = _prop(cap, cv2.CAP_PROP_POS_MSEC, default=-1.0)
            if pos <= 0:
                pos = (index / fps * 1000.0) if fps > 0 else float(index)
                if seeked:
                    pos += start_ms
            if not seeked and pos < start_ms:
                continue
            if end_ms is not None and pos > end_ms:
                break
            if index % stride != 0:
                continue
            image = _rescale(image, resize_to, max_side)
            yield Frame(index=index, timestamp_ms=float(pos), image=image, source=str(path))
            kept += 1
            if max_frames is not None and kept >= max_frames:
                break
    finally:
        cap.release()


def _rescale(
    image: np.ndarray, resize_to: tuple[int, int] | None, max_side: int | None
) -> np.ndarray:
    if resize_to is not None:
        return cv2.resize(image, resize_to, interpolation=cv2.INTER_AREA)
    if max_side is not None:
        h, w = image.shape[:2]
        longest = max(h, w)
        if longest > max_side:
            scale = max_side / longest
            return cv2.resize(
                image,
                (max(1, round(w * scale)), max(1, round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
    return image


def read_image(path: PathLike, *, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image, raising rather than returning None (imread's silent failure)."""
    image = cv2.imread(str(path), flags)
    if image is None:
        raise DecodeError(f"cannot decode image: {path}")
    return image


def decode_image(data: bytes, *, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Decode image bytes from an upload."""
    if not data:
        raise DecodeError("empty image payload")
    buf = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buf, flags)
    if image is None:
        raise DecodeError("cannot decode image bytes")
    return image


def encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise DecodeError("jpeg encode failed")
    return buf.tobytes()


def encode_png(image: np.ndarray, compression: int = 3) -> bytes:
    ok, buf = cv2.imencode(".png", image, [int(cv2.IMWRITE_PNG_COMPRESSION), int(compression)])
    if not ok:
        raise DecodeError("png encode failed")
    return buf.tobytes()


def write_image(path: PathLike, image: np.ndarray) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), image):
        raise DecodeError(f"cannot write image: {out}")
    return out


def iter_images(paths: list[PathLike], *, max_side: int | None = None) -> Iterator[Frame]:
    """Treat a list of stills as a frame sequence, so the same pipeline accepts both."""
    for i, path in enumerate(paths):
        image = _rescale(read_image(path), None, max_side)
        yield Frame(index=i, timestamp_ms=float(i), image=image, source=str(path))


def sharpness(image: np.ndarray) -> float:
    """Laplacian variance. Higher is sharper; use it to pick keyframes."""
    gray = to_gray(image)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def pick_sharpest(frames: list[Frame], k: int = 1) -> list[Frame]:
    """Top-k frames by Laplacian variance, in original order."""
    scored = sorted(frames, key=sharpness_of_frame, reverse=True)[:k]
    return sorted(scored, key=lambda f: f.index)


def sharpness_of_frame(frame: Frame) -> float:
    return sharpness(frame.image)
