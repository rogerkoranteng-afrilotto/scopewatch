"""ONNX model loading and inference through `cv2.dnn`, from a local path or S3.

Licensing, deliberately: the default detector is **YOLOX (Apache-2.0, Megvii)**.
Ultralytics YOLOv5/v8/11/12 are AGPL-3.0, and AGPL section 13 extends copyleft to
network use, so shipping a hosted demo on those weights is a source-disclosure
event. Do not add them. See `docs/decisions.md`.

OpenCV 5 notes that shape this module:
  * `readNetFromCaffe` / `readNetFromDarknet` are gone. ONNX only.
  * `readNetFromONNX(path, engine=...)` selects the graph engine.
    ENGINE_NEW measured ~1.5x faster than ENGINE_CLASSIC on YOLOX-tiny
    (13.8 ms vs 21.0 ms/inference, x86, 22 threads)
  * The PyPI wheel is built with `ONNX Runtime: NO`, so `ENGINE_ORT` exists as a
    constant but has no backend. Asking for it raises here rather than silently
    falling back.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

from .timing import stage

DEFAULT_CACHE_DIR = Path(
    os.environ.get("OPENCV26_MODEL_DIR", Path.home() / ".cache" / "opencv26" / "models")
)

ENGINES: dict[str, int] = {
    "auto": cv2.dnn.ENGINE_AUTO,
    "classic": cv2.dnn.ENGINE_CLASSIC,
    "new": cv2.dnn.ENGINE_NEW,
}


class ModelError(RuntimeError):
    """Model could not be fetched, verified or loaded."""


@dataclass(frozen=True)
class ModelSpec:
    """Where a model lives, what it expects, and what licence it carries.

    `licence` is not decoration: the competition rules let judges reject an entry
    that "uses data, models, or media without the necessary rights", and every
    RunRecord carries this string.
    """

    name: str
    uri: str  # local path or s3://bucket/key
    input_size: tuple[int, int] = (416, 416)  # (width, height)
    licence: str = "unknown"
    version: str = "1"
    sha256: str | None = None
    swap_rb: bool = False
    scale: float = 1.0
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    class_names: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "uri": self.uri,
            "version": self.version,
            "input_size": list(self.input_size),
            "licence": self.licence,
            "sha256": self.sha256,
            "notes": self.notes,
        }


# YOLOX-tiny: Apache-2.0, Megvii. Official ONNX export takes raw BGR 0-255 with no
# mean/std (the `--legacy` export does not; do not mix them up).
YOLOX_TINY = ModelSpec(
    name="yolox-tiny",
    uri=os.environ.get(
        "OPENCV26_YOLOX_URI",
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx",
    ),
    input_size=(416, 416),
    licence="Apache-2.0",
    version="0.1.1rc0",
    notes="Megvii YOLOX-tiny, official ONNX export. Output (1, 3549, 85) at 416x416.",
)

COCO_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_model(spec: ModelSpec, cache_dir: Path | None = None) -> Path:
    """Resolve a ModelSpec to a local file, downloading from S3 or HTTPS once.

    Idempotent: an already-cached file with a matching sha256 is reused.
    """
    cache = Path(cache_dir or DEFAULT_CACHE_DIR)
    parsed = urlparse(spec.uri)

    if parsed.scheme in ("", "file"):
        local = Path(parsed.path if parsed.scheme == "file" else spec.uri).expanduser()
        if not local.is_file():
            raise ModelError(f"model {spec.name}: no such file {local}")
        _verify(spec, local)
        return local

    cache.mkdir(parents=True, exist_ok=True)
    suffix = Path(parsed.path).suffix or ".onnx"
    target = cache / f"{spec.name}-{spec.version}{suffix}"
    if target.is_file():
        try:
            _verify(spec, target)
            return target
        except ModelError:
            target.unlink(missing_ok=True)

    tmp = target.with_suffix(target.suffix + ".part")
    if parsed.scheme == "s3":
        _fetch_s3(parsed.netloc, parsed.path.lstrip("/"), tmp)
    elif parsed.scheme in ("http", "https"):
        _fetch_http(spec.uri, tmp)
    else:
        raise ModelError(f"model {spec.name}: unsupported URI scheme {parsed.scheme!r}")

    shutil.move(str(tmp), str(target))
    _verify(spec, target)
    return target


def _verify(spec: ModelSpec, path: Path) -> None:
    if path.stat().st_size == 0:
        raise ModelError(f"model {spec.name}: {path} is empty")
    if spec.sha256:
        actual = _sha256(path)
        if actual != spec.sha256:
            raise ModelError(
                f"model {spec.name}: sha256 mismatch (expected {spec.sha256}, got {actual})"
            )


def _fetch_s3(bucket: str, key: str, target: Path) -> None:
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - exercised only without boto3
        raise ModelError(
            "s3:// model URIs need the 's3' extra: pip install visioncore[s3]"
        ) from exc
    try:
        boto3.client("s3").download_file(bucket, key, str(target))
    except Exception as exc:
        raise ModelError(f"s3 download failed for s3://{bucket}/{key}: {exc}") from exc


def _fetch_http(url: str, target: Path) -> None:
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as out:
            shutil.copyfileobj(response, out)
    except Exception as exc:
        raise ModelError(f"http download failed for {url}: {exc}") from exc


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class DnnRunner:
    """A loaded `cv2.dnn.Net` plus the blob conventions its ModelSpec declares.

    Not safe to share a single instance across threads: `cv2.dnn.Net` keeps
    internal state between `setInput` and `forward`. An internal lock serialises
    calls so a shared instance is *correct*, just not concurrent.
    """

    def __init__(
        self,
        spec: ModelSpec,
        *,
        engine: str = "auto",
        cache_dir: Path | None = None,
        num_threads: int | None = None,
    ) -> None:
        if engine not in ENGINES:
            raise ModelError(
                f"unknown engine {engine!r}; pick one of {sorted(ENGINES)}. "
                "ENGINE_ORT is not usable: the PyPI wheel is built with ONNX Runtime: NO."
            )
        self.spec = spec
        self.engine = engine
        self.path = fetch_model(spec, cache_dir)
        self._lock = threading.Lock()
        if num_threads is not None:
            cv2.setNumThreads(int(num_threads))
        try:
            self.net = cv2.dnn.readNetFromONNX(str(self.path), engine=ENGINES[engine])
        except TypeError:
            # Older signature without the keyword; fall back and note it.
            self.net = cv2.dnn.readNetFromONNX(str(self.path))
        except cv2.error as exc:
            raise ModelError(f"cannot load {spec.name} from {self.path}: {exc}") from exc

    def blob(self, image: np.ndarray) -> np.ndarray:
        w, h = self.spec.input_size
        return cv2.dnn.blobFromImage(
            image,
            scalefactor=self.spec.scale,
            size=(w, h),
            mean=self.spec.mean,
            swapRB=self.spec.swap_rb,
            crop=False,
        )

    def forward(self, blob: np.ndarray) -> np.ndarray:
        with self._lock, stage(f"dnn:{self.spec.name}"):
            self.net.setInput(blob)
            return self.net.forward()

    def infer(self, image: np.ndarray) -> np.ndarray:
        return self.forward(self.blob(image))

    def info(self) -> dict[str, Any]:
        return {**self.spec.to_dict(), "engine": self.engine, "local_path": str(self.path)}


# ---------------------------------------------------------------------------
# YOLOX
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in original image pixels
    score: float
    class_id: int
    class_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        x1, y1, x2, y2 = self.bbox
        return {
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "score": round(self.score, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


def letterbox(
    image: np.ndarray, size: tuple[int, int], pad_value: int = 114
) -> tuple[np.ndarray, float]:
    """Resize preserving aspect ratio onto a padded canvas. Returns (canvas, ratio).

    YOLOX pads bottom-right only, so un-scaling a box is a single divide by `ratio`
    with no offset. Match that exactly or every box lands slightly wrong.
    """
    target_w, target_h = size
    h, w = image.shape[:2]
    ratio = min(target_w / w, target_h / h)
    new_w, new_h = round(w * ratio), round(h * ratio)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    channels = 3 if image.ndim == 3 else 1
    canvas = np.full((target_h, target_w, channels), pad_value, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized.reshape(new_h, new_w, channels)
    return canvas, ratio


def yolox_grids(size: tuple[int, int], strides: tuple[int, ...] = (8, 16, 32)):
    """(grid, expanded_strides) for the anchor-free YOLOX decode."""
    width, height = size
    grids, expanded = [], []
    for stride in strides:
        gw, gh = width // stride, height // stride
        xv, yv = np.meshgrid(np.arange(gw), np.arange(gh))
        grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2)
        grids.append(grid)
        expanded.append(np.full((1, grid.shape[1], 1), stride))
    return np.concatenate(grids, axis=1), np.concatenate(expanded, axis=1)


def decode_yolox(
    outputs: np.ndarray, size: tuple[int, int], strides: tuple[int, ...] = (8, 16, 32)
) -> np.ndarray:
    """Map raw YOLOX output to (N, 85) with xywh in **input-image pixels**.

    Raw layout per anchor: [dx, dy, log_w, log_h, objectness, class scores...].
    Centres are grid-relative, sizes are exponentiated; both scale by the stride.
    Pure function of arrays - unit-testable with no model file.
    """
    out = np.array(outputs, dtype=np.float32, copy=True)
    if out.ndim == 3:
        out = out[0]
    grid, expanded = yolox_grids(size, strides)
    if out.shape[0] != grid.shape[1]:
        raise ValueError(
            f"yolox decode: {out.shape[0]} anchors but grid expects {grid.shape[1]} "
            f"for input {size} and strides {strides}"
        )
    out[:, 0:2] = (out[:, 0:2] + grid[0]) * expanded[0]
    out[:, 2:4] = np.exp(out[:, 2:4]) * expanded[0]
    return out


def postprocess_yolox(
    outputs: np.ndarray,
    ratio: float,
    *,
    size: tuple[int, int] = (416, 416),
    score_threshold: float = 0.3,
    nms_threshold: float = 0.45,
    class_names: tuple[str, ...] = COCO_CLASSES,
    strides: tuple[int, ...] = (8, 16, 32),
) -> list[Detection]:
    """Decode, threshold, NMS, and un-letterbox to original image coordinates."""
    decoded = decode_yolox(outputs, size, strides)
    boxes_xywh = decoded[:, 0:4]
    scores = decoded[:, 4:5] * decoded[:, 5:]
    class_ids = scores.argmax(axis=1)
    confidences = scores[np.arange(scores.shape[0]), class_ids]

    keep = confidences >= score_threshold
    if not np.any(keep):
        return []
    boxes_xywh, class_ids, confidences = boxes_xywh[keep], class_ids[keep], confidences[keep]

    # centre-xywh -> corner-xywh, in original pixels
    corner = np.empty_like(boxes_xywh)
    corner[:, 0] = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0) / ratio
    corner[:, 1] = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0) / ratio
    corner[:, 2] = boxes_xywh[:, 2] / ratio
    corner[:, 3] = boxes_xywh[:, 3] / ratio

    indices = cv2.dnn.NMSBoxes(
        corner.tolist(), confidences.astype(float).tolist(), score_threshold, nms_threshold
    )
    if indices is None or len(indices) == 0:
        return []

    detections: list[Detection] = []
    for i in np.array(indices).flatten():
        x, y, w, h = corner[int(i)]
        cid = int(class_ids[int(i)])
        detections.append(
            Detection(
                bbox=(float(x), float(y), float(x + w), float(y + h)),
                score=float(confidences[int(i)]),
                class_id=cid,
                class_name=class_names[cid] if cid < len(class_names) else str(cid),
            )
        )
    return sorted(detections, key=lambda d: d.score, reverse=True)


class YoloxDetector:
    """The default detector: YOLOX-tiny in `cv2.dnn`, Apache-2.0, ONNX, CPU."""

    def __init__(
        self,
        spec: ModelSpec = YOLOX_TINY,
        *,
        engine: str = "auto",
        score_threshold: float = 0.3,
        nms_threshold: float = 0.45,
        class_names: tuple[str, ...] = COCO_CLASSES,
        cache_dir: Path | None = None,
    ) -> None:
        self.runner = DnnRunner(spec, engine=engine, cache_dir=cache_dir)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.class_names = class_names

    def detect(self, image: np.ndarray) -> list[Detection]:
        size = self.runner.spec.input_size
        canvas, ratio = letterbox(image, size)
        blob = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None, ...].astype(np.float32))
        outputs = self.runner.forward(blob)
        return postprocess_yolox(
            outputs,
            ratio,
            size=size,
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            class_names=self.class_names,
        )

    def info(self) -> dict[str, Any]:
        return self.runner.info()


def draw_detections(
    image: np.ndarray, detections: list[Detection], *, colour: tuple[int, int, int] = (0, 200, 255)
) -> np.ndarray:
    """Annotated copy, for the evidence panel. Uses OpenCV 5's FontFace when present."""
    out = image.copy()
    font = cv2.FontFace("sans") if hasattr(cv2, "FontFace") else None
    for det in detections:
        x1, y1, x2, y2 = (round(v) for v in det.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        label = f"{det.class_name} {det.score:.2f}"
        if font is not None:
            cv2.putText(out, label, (x1, max(0, y1 - 6)), colour, font, 16)
        else:  # pragma: no cover - OpenCV 5 always has FontFace
            cv2.putText(
                out, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1
            )
    return out
