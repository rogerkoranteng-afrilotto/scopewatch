"""visioncore - shared OpenCV 5 primitives for the opencv26 competition entries.

Import this and you have already checked that OpenCV 5 is what got installed.
"""

from __future__ import annotations

from .calibration import (
    Calibration,
    CalibrationRefused,
    calibrate,
    calibrate_from_aruco,
    calibrate_from_checkerboard,
    detect_aruco,
    draw_marker,
    local_scale,
    quad_obliquity_deg,
)
from .imageio import (
    DecodeError,
    Frame,
    VideoInfo,
    decode_image,
    encode_jpeg,
    encode_png,
    iter_images,
    iter_video,
    pick_sharpest,
    read_image,
    sharpness,
    to_gray,
    video_info,
    write_image,
)
from .measure import (
    WidthProfile,
    elongated_components,
    medial_axis,
    stroke_width_profile,
)
from .models import (
    COCO_CLASSES,
    YOLOX_TINY,
    Detection,
    DnnRunner,
    ModelError,
    ModelSpec,
    YoloxDetector,
    decode_yolox,
    draw_detections,
    fetch_model,
    letterbox,
    postprocess_yolox,
)
from .records import Evidence, Refusal, RunRecord, Stage
from .timing import StageTimer, active_record, recording, stage, timed
from .version import (
    REQUIRED_OPENCV,
    Environment,
    OpenCVVersionError,
    assert_opencv5,
    cpu_features,
    environment,
    git_sha,
)

assert_opencv5()

__version__ = "0.1.0"

__all__ = [
    "COCO_CLASSES",
    "REQUIRED_OPENCV",
    "YOLOX_TINY",
    "Calibration",
    "CalibrationRefused",
    "DecodeError",
    "Detection",
    "DnnRunner",
    "Environment",
    "Evidence",
    "Frame",
    "ModelError",
    "ModelSpec",
    "OpenCVVersionError",
    "Refusal",
    "RunRecord",
    "Stage",
    "StageTimer",
    "VideoInfo",
    "WidthProfile",
    "YoloxDetector",
    "__version__",
    "active_record",
    "assert_opencv5",
    "calibrate",
    "calibrate_from_aruco",
    "calibrate_from_checkerboard",
    "cpu_features",
    "decode_image",
    "decode_yolox",
    "detect_aruco",
    "draw_detections",
    "draw_marker",
    "elongated_components",
    "encode_jpeg",
    "encode_png",
    "environment",
    "fetch_model",
    "git_sha",
    "iter_images",
    "iter_video",
    "letterbox",
    "local_scale",
    "medial_axis",
    "pick_sharpest",
    "postprocess_yolox",
    "quad_obliquity_deg",
    "read_image",
    "recording",
    "sharpness",
    "stage",
    "stroke_width_profile",
    "timed",
    "to_gray",
    "video_info",
    "write_image",
]
