"""Pixels-to-millimetres from a fiducial, with an explicit refusal path.

Why this module is paranoid
---------------------------
Research recorded a 380x measurement error: a
crack-width routine filled an open contour and took `max(distanceTransform)`,
reporting 283 mm for a 0.75 mm crack. The number looked plausible. Nothing
crashed. That is the failure mode this module exists to prevent.

So every entry point returns a `Calibration` that is either `ok` with a scale
**and an uncertainty**, or not ok with a machine-readable `Refusal`. There is no
third state and no default scale. Callers that want a number must check `ok`.

Refusal reasons
---------------
NO_FIDUCIAL        nothing detected
MARKER_TOO_SMALL   the fiducial is too few pixels for the requested precision
TOO_OBLIQUE        the fiducial plane is too far from fronto-parallel
DEGENERATE         the quad is near-collinear / homography is ill-conditioned
HIGH_RESIDUAL      the fitted homography does not explain the detected points
INCONSISTENT_SCALE multiple fiducials disagree, so no single scalar scale is valid
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .records import Refusal

# Defaults chosen so a refusal is the safe outcome, not a rare one.
DEFAULT_MAX_OBLIQUITY_DEG = 35.0  # cos(35 deg) ~ 0.82 foreshortening
DEFAULT_MIN_MARKER_PX = 40.0  # below this, 1 px of error is > 2.5% of the scale
DEFAULT_MAX_RESIDUAL_PX = 2.0
DEFAULT_MAX_SCALE_SPREAD = 0.08  # 8% disagreement between fiducials

ARUCO_DICTS: dict[str, int] = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


class CalibrationRefused(RuntimeError):
    """Raised by `Calibration.require()` when a caller demands a scale we do not have."""

    def __init__(self, refusal: Refusal) -> None:
        super().__init__(f"{refusal.code}: {refusal.message}")
        self.refusal = refusal


@dataclass(frozen=True)
class Calibration:
    """A pixel-to-millimetre scale, or a refusal. Never both, never neither."""

    ok: bool
    method: str
    px_per_mm: float | None = None
    obliquity_deg: float | None = None
    residual_px: float | None = None
    min_feature_px: float | None = None
    homography_mm_to_px: np.ndarray | None = None
    marker_ids: tuple[int, ...] = ()
    refusal: Refusal | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    # ---- derived ----------------------------------------------------------
    @property
    def mm_per_px(self) -> float | None:
        if self.px_per_mm is None or self.px_per_mm <= 0:
            return None
        return 1.0 / self.px_per_mm

    @property
    def precision_mm(self) -> float | None:
        """Millimetres represented by one pixel. Never report a number below this."""
        return self.mm_per_px

    def require(self) -> float:
        """px_per_mm, or raise. Use when a caller genuinely cannot proceed without it."""
        if not self.ok or self.px_per_mm is None:
            raise CalibrationRefused(
                self.refusal or Refusal("NO_FIDUCIAL", "no calibration available")
            )
        return self.px_per_mm

    def px_to_mm(self, pixels: float) -> float | None:
        """Scalar conversion. Only valid near the fiducial; use `segment_mm` for accuracy."""
        if not self.ok or self.px_per_mm is None:
            return None
        return float(pixels) / self.px_per_mm

    def segment_mm(self, p0: tuple[float, float], p1: tuple[float, float]) -> float | None:
        """Perspective-correct length of an image-space segment, via the plane homography.

        This is the honest conversion: it undoes foreshortening across the frame
        instead of applying one scalar everywhere. Only meaningful for points that
        lie in the fiducial's plane.
        """
        if not self.ok:
            return None
        if self.homography_mm_to_px is None:
            dist = math.dist(p0, p1)
            return self.px_to_mm(dist)
        h_inv = np.linalg.inv(self.homography_mm_to_px)
        pts = np.array([[p0, p1]], dtype=np.float64)
        mm = cv2.perspectiveTransform(pts, h_inv)[0]
        return float(math.dist(tuple(mm[0]), tuple(mm[1])))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "method": self.method,
            "px_per_mm": None if self.px_per_mm is None else round(self.px_per_mm, 5),
            "mm_per_px": None if self.mm_per_px is None else round(self.mm_per_px, 6),
            "precision_mm": None if self.precision_mm is None else round(self.precision_mm, 6),
            "obliquity_deg": None
            if self.obliquity_deg is None
            else round(self.obliquity_deg, 2),
            "residual_px": None if self.residual_px is None else round(self.residual_px, 3),
            "min_feature_px": None
            if self.min_feature_px is None
            else round(self.min_feature_px, 2),
            "marker_ids": list(self.marker_ids),
            "refusal": self.refusal.to_dict() if self.refusal else None,
            "diagnostics": dict(self.diagnostics),
        }


def _refuse(method: str, code: str, message: str, **details: Any) -> Calibration:
    return Calibration(
        ok=False,
        method=method,
        refusal=Refusal(code=code, message=message, details=details),
        diagnostics=dict(details),
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def local_scale(h_mm_to_px: np.ndarray, x_mm: float, y_mm: float) -> float:
    """px per mm at a point on the plane, from the homography's local Jacobian.

    For H mapping plane-mm to image-px, the isotropic local scale is
    sqrt(|det J|) where J is the 2x2 Jacobian of the projective map. This is the
    right answer at the fiducial and degrades gracefully away from it; using a
    naive edge-length ratio instead bakes the foreshortening into the scale.
    """
    h = np.asarray(h_mm_to_px, dtype=np.float64)
    p = h @ np.array([x_mm, y_mm, 1.0])
    w = p[2]
    if abs(w) < 1e-12:
        return 0.0
    u, v = p[0], p[1]
    j = np.empty((2, 2), dtype=np.float64)
    for k in range(2):
        j[0, k] = (h[0, k] * w - u * h[2, k]) / (w * w)
        j[1, k] = (h[1, k] * w - v * h[2, k]) / (w * w)
    det = abs(float(np.linalg.det(j)))
    return math.sqrt(det)


def quad_obliquity_deg(corners: np.ndarray, aspect: float = 1.0) -> float:
    """Tilt of a fiducial of known aspect ratio, estimated without camera intrinsics.

    A rectangle of known aspect viewed fronto-parallel images at that aspect;
    tilting by theta about an in-plane axis foreshortens the perpendicular edge
    pair by ~cos(theta). Normalise each edge by its expected physical length and
    theta ~= acos(min / max). Approximate (it ignores the perspective divide and
    any lens distortion) but intrinsics-free, monotonic in the real tilt, and
    good enough to decide *whether to refuse*. When intrinsics are supplied,
    `_obliquity_from_pnp` replaces this.

    `aspect` is expected_width / expected_height of the fiducial. Corners are in
    order: top-left, top-right, bottom-right, bottom-left (ArUco's convention),
    so edges 0 and 2 are the width pair.
    """
    if aspect <= 0:
        raise ValueError("aspect must be positive")
    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    edges = [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)]
    # px per unit of physical length along each edge
    normalised = [edges[0] / aspect, edges[1], edges[2] / aspect, edges[3]]
    longest, shortest = max(normalised), min(normalised)
    if longest <= 0:
        return 90.0
    ratio = max(0.0, min(1.0, shortest / longest))
    return math.degrees(math.acos(ratio))


def _obliquity_from_homography(
    h_mm_to_px: np.ndarray, camera_matrix: np.ndarray
) -> float | None:
    """Plane tilt in degrees, from the homography and known intrinsics.

    A plane homography factorises as H ~ K [r1 r2 t]. Recover the scale from
    ||K^-1 h1|| = 1, take r3 = r1 x r2 as the plane normal in camera coordinates,
    and measure its angle to the viewing ray t/||t||.

    Why not solvePnP: `SOLVEPNP_IPPE_SQUARE` has two solutions for a planar
    square and returns one of them. Measured against a synthetic camera it
    reported **2.0 deg for a true 20 deg tilt and 1.1 deg for a true 30 deg
    tilt** - it picked the mirror solution, and a refusal gate built on that
    would wave through exactly the geometry it exists to catch. Selecting by
    reprojection error with `solvePnPGeneric` did not fix it. This decomposition
    is deterministic and was within 1.7 deg of truth across 0-60 deg.
    """
    k_inv = np.linalg.inv(np.asarray(camera_matrix, dtype=np.float64))
    m = k_inv @ np.asarray(h_mm_to_px, dtype=np.float64)
    norm = np.linalg.norm(m[:, 0])
    if norm < 1e-12 or not np.all(np.isfinite(m)):
        return None
    scale = 1.0 / norm
    r1, r2, t = m[:, 0] * scale, m[:, 1] * scale, m[:, 2] * scale
    if t[2] < 0:  # the plane must be in front of the camera
        r1, r2, t = -r1, -r2, -t
    normal = np.cross(r1, r2)
    n_norm, t_norm = np.linalg.norm(normal), np.linalg.norm(t)
    if n_norm < 1e-12 or t_norm < 1e-12:
        return None
    cos = abs(float(np.dot(normal / n_norm, t / t_norm)))
    return math.degrees(math.acos(max(0.0, min(1.0, cos))))


def _quad_is_degenerate(corners: np.ndarray, min_px: float) -> tuple[bool, float]:
    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    area = abs(float(cv2.contourArea(pts.astype(np.float32))))
    edges = [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)]
    shortest = min(edges)
    # A square of edge e has area e^2; anything under half of that is folded or collinear.
    expected = (sum(edges) / 4.0) ** 2
    return (area < 0.5 * expected or shortest < min_px), shortest


def _residual_px(h_mm_to_px: np.ndarray, obj_mm: np.ndarray, img_px: np.ndarray) -> float:
    pts = np.asarray(obj_mm, dtype=np.float64).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts, np.asarray(h_mm_to_px, dtype=np.float64))
    diff = projected.reshape(-1, 2) - np.asarray(img_px, dtype=np.float64).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


# ---------------------------------------------------------------------------
# ArUco
# ---------------------------------------------------------------------------


def _detector(dictionary: str) -> cv2.aruco.ArucoDetector:
    if dictionary not in ARUCO_DICTS:
        raise ValueError(f"unknown aruco dictionary {dictionary!r}; known: {sorted(ARUCO_DICTS)}")
    adict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dictionary])
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(adict, params)


def detect_aruco(
    image: np.ndarray, dictionary: str = "DICT_4X4_50"
) -> tuple[list[np.ndarray], list[int]]:
    """Detected marker corners (4x2, clockwise from top-left) and their ids."""
    corners, ids, _ = _detector(dictionary).detectMarkers(image)
    if ids is None or len(ids) == 0:
        return [], []
    return [c.reshape(4, 2).astype(np.float64) for c in corners], [int(i) for i in ids.flatten()]


def calibrate_from_aruco(
    image: np.ndarray,
    marker_length_mm: float,
    *,
    dictionary: str = "DICT_4X4_50",
    expected_ids: tuple[int, ...] | None = None,
    max_obliquity_deg: float = DEFAULT_MAX_OBLIQUITY_DEG,
    min_marker_px: float = DEFAULT_MIN_MARKER_PX,
    max_residual_px: float = DEFAULT_MAX_RESIDUAL_PX,
    max_scale_spread: float = DEFAULT_MAX_SCALE_SPREAD,
    camera_matrix: np.ndarray | None = None,
    dist_coeffs: np.ndarray | None = None,
) -> Calibration:
    """px-per-mm from one or more square ArUco markers of known physical edge length.

    Without `camera_matrix` the tilt estimate is intrinsics-free and slightly
    conservative (it over-reads tilt by about 2 deg), which is the right
    direction for a gate that exists to refuse. With intrinsics it is metric.

    A note on what `px_per_mm` means: it is the **isotropic** scale, the geometric
    mean of the two principal scales at the marker centre. On a tilted plane the
    real scale differs along and across the tilt axis, so at the 35 deg refusal
    limit this scalar can understate a length along the tilt direction by about
    10%. For an actual measurement use `segment_mm()`, which goes through the
    homography and is perspective-correct.
    """
    if marker_length_mm <= 0:
        raise ValueError("marker_length_mm must be positive")
    method = f"aruco:{dictionary}"

    corners, ids = detect_aruco(image, dictionary)
    if dist_coeffs is not None and camera_matrix is not None and corners:
        # Undistort first, or barrel distortion on a phone lens shows up as tilt.
        corners = [
            cv2.undistortPoints(
                quad.reshape(-1, 1, 2), camera_matrix, dist_coeffs, P=camera_matrix
            ).reshape(4, 2)
            for quad in corners
        ]
    if expected_ids is not None:
        keep = [i for i, mid in enumerate(ids) if mid in expected_ids]
        corners = [corners[i] for i in keep]
        ids = [ids[i] for i in keep]

    if not corners:
        return _refuse(
            method,
            "NO_FIDUCIAL",
            "no ArUco marker detected; the scale reference must be visible and in focus",
            dictionary=dictionary,
            expected_ids=list(expected_ids) if expected_ids else None,
        )

    half = marker_length_mm / 2.0
    obj_mm = np.array(
        [[-half, half], [half, half], [half, -half], [-half, -half]], dtype=np.float64
    )

    scales: list[float] = []
    obliquities: list[float] = []
    residuals: list[float] = []
    homographies: list[np.ndarray] = []
    shortest_edges: list[float] = []

    for quad in corners:
        degenerate, shortest = _quad_is_degenerate(quad, min_px=4.0)
        shortest_edges.append(shortest)
        if degenerate:
            return _refuse(
                method,
                "DEGENERATE",
                "the detected marker quad is folded or near-collinear; "
                "re-shoot square to the surface",
                shortest_edge_px=round(shortest, 2),
            )
        h = cv2.getPerspectiveTransform(obj_mm.astype(np.float32), quad.astype(np.float32))
        if not np.all(np.isfinite(h)) or abs(float(np.linalg.det(h))) < 1e-12:
            return _refuse(method, "DEGENERATE", "marker homography is singular")
        homographies.append(h)
        scales.append(local_scale(h, 0.0, 0.0))
        residuals.append(_residual_px(h, obj_mm, quad))
        if camera_matrix is not None:
            metric = _obliquity_from_homography(h, camera_matrix)
            obliquities.append(metric if metric is not None else quad_obliquity_deg(quad))
        else:
            obliquities.append(quad_obliquity_deg(quad))

    min_edge = float(min(shortest_edges))
    obliquity = float(max(obliquities))
    residual = float(max(residuals))
    scale = float(np.median(scales))

    if min_edge < min_marker_px:
        return _refuse(
            method,
            "MARKER_TOO_SMALL",
            f"marker edge is {min_edge:.1f} px, below the {min_marker_px:.0f} px floor; "
            "move closer or use a larger marker",
            min_edge_px=round(min_edge, 2),
            min_marker_px=min_marker_px,
            marker_ids=ids,
        )

    if obliquity > max_obliquity_deg:
        return _refuse(
            method,
            "TOO_OBLIQUE",
            f"marker plane is ~{obliquity:.0f} deg off fronto-parallel "
            f"(limit {max_obliquity_deg:.0f} deg); re-shoot square to the surface",
            obliquity_deg=round(obliquity, 2),
            max_obliquity_deg=max_obliquity_deg,
            marker_ids=ids,
        )

    if residual > max_residual_px:
        return _refuse(
            method,
            "HIGH_RESIDUAL",
            f"marker corners fit the plane model to only {residual:.2f} px",
            residual_px=round(residual, 3),
            max_residual_px=max_residual_px,
        )

    if len(scales) > 1:
        spread = (max(scales) - min(scales)) / max(1e-9, float(np.median(scales)))
        if spread > max_scale_spread:
            return _refuse(
                method,
                "INCONSISTENT_SCALE",
                f"markers disagree on scale by {spread * 100:.1f}%; they are probably not "
                "coplanar, so no single px/mm is valid for this frame",
                scale_spread=round(spread, 4),
                px_per_mm_per_marker=[round(s, 4) for s in scales],
                marker_ids=ids,
            )

    if scale <= 0 or not math.isfinite(scale):
        return _refuse(method, "DEGENERATE", "computed scale is not finite")

    best = int(np.argmin(np.abs(np.array(scales) - scale)))
    return Calibration(
        ok=True,
        method=method,
        px_per_mm=scale,
        obliquity_deg=obliquity,
        residual_px=residual,
        min_feature_px=min_edge,
        homography_mm_to_px=homographies[best],
        marker_ids=tuple(ids),
        diagnostics={
            "markers_used": len(scales),
            "px_per_mm_per_marker": [round(s, 4) for s in scales],
            "obliquity_source": "homography" if camera_matrix is not None else "edge-ratio",
        },
    )


# ---------------------------------------------------------------------------
# Checkerboard
# ---------------------------------------------------------------------------


def calibrate_from_checkerboard(
    image: np.ndarray,
    pattern_size: tuple[int, int],
    square_size_mm: float,
    *,
    max_obliquity_deg: float = DEFAULT_MAX_OBLIQUITY_DEG,
    min_marker_px: float = DEFAULT_MIN_MARKER_PX,
    max_residual_px: float = DEFAULT_MAX_RESIDUAL_PX,
) -> Calibration:
    """px-per-mm from a checkerboard of `pattern_size` **inner** corners (cols, rows)."""
    if square_size_mm <= 0:
        raise ValueError("square_size_mm must be positive")
    method = f"checkerboard:{pattern_size[0]}x{pattern_size[1]}"
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCornersSB(gray, pattern_size, cv2.CALIB_CB_EXHAUSTIVE)
    if not found or corners is None:
        found, corners = cv2.findChessboardCorners(
            gray, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if found and corners is not None:
            cv2.cornerSubPix(
                gray,
                corners,
                (5, 5),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
            )
    if not found or corners is None:
        return _refuse(
            method,
            "NO_FIDUCIAL",
            "no checkerboard found; the whole board must be visible and in focus",
            pattern_size=list(pattern_size),
        )

    cols, rows = pattern_size
    img_px = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    grid = np.array(
        [[c * square_size_mm, r * square_size_mm] for r in range(rows) for c in range(cols)],
        dtype=np.float64,
    )

    h, mask = cv2.findHomography(grid, img_px, cv2.LMEDS)
    if h is None or not np.all(np.isfinite(h)):
        return _refuse(method, "DEGENERATE", "checkerboard homography could not be fitted")

    residual = _residual_px(h, grid, img_px)
    centre = grid.mean(axis=0)
    scale = local_scale(h, float(centre[0]), float(centre[1]))

    outer = np.array(
        [img_px[0], img_px[cols - 1], img_px[-1], img_px[-cols]], dtype=np.float64
    )
    # The corner grid is (cols-1) x (rows-1) squares across, so the outer quad is
    # not square; feed its known aspect in or a plain board reads as 48 deg tilted.
    aspect = (cols - 1) / max(1, rows - 1)
    obliquity = quad_obliquity_deg(outer, aspect=aspect)
    board_edges = [float(np.linalg.norm(outer[(i + 1) % 4] - outer[i])) for i in range(4)]
    min_square_px = min(
        board_edges[0] / max(1, cols - 1),
        board_edges[1] / max(1, rows - 1),
        board_edges[2] / max(1, cols - 1),
        board_edges[3] / max(1, rows - 1),
    )

    if min_square_px < min_marker_px / 4.0:
        return _refuse(
            method,
            "MARKER_TOO_SMALL",
            f"checkerboard squares are only {min_square_px:.1f} px; move closer",
            square_px=round(min_square_px, 2),
        )
    if obliquity > max_obliquity_deg:
        return _refuse(
            method,
            "TOO_OBLIQUE",
            f"board plane is ~{obliquity:.0f} deg off fronto-parallel "
            f"(limit {max_obliquity_deg:.0f} deg)",
            obliquity_deg=round(obliquity, 2),
        )
    if residual > max_residual_px:
        return _refuse(
            method,
            "HIGH_RESIDUAL",
            f"corner grid fits the plane model to only {residual:.2f} px; "
            "the board may be bent or the lens heavily distorted",
            residual_px=round(residual, 3),
        )

    return Calibration(
        ok=True,
        method=method,
        px_per_mm=scale,
        obliquity_deg=obliquity,
        residual_px=residual,
        min_feature_px=min_square_px,
        homography_mm_to_px=h,
        diagnostics={
            "corners": int(img_px.shape[0]),
            "inliers": int(mask.sum()) if mask is not None else None,
            "square_px": round(min_square_px, 2),
            "obliquity_source": "edge-ratio",
        },
    )


def calibrate(
    image: np.ndarray,
    *,
    marker_length_mm: float | None = None,
    dictionary: str = "DICT_4X4_50",
    checkerboard: tuple[int, int] | None = None,
    square_size_mm: float | None = None,
    **kwargs: Any,
) -> Calibration:
    """Try ArUco, then checkerboard. Returns the first success, else the first refusal."""
    attempts: list[Calibration] = []
    if marker_length_mm:
        result = calibrate_from_aruco(
            image, marker_length_mm, dictionary=dictionary, **kwargs
        )
        if result.ok:
            return result
        attempts.append(result)
    if checkerboard and square_size_mm:
        result = calibrate_from_checkerboard(image, checkerboard, square_size_mm, **kwargs)
        if result.ok:
            return result
        attempts.append(result)
    if attempts:
        return attempts[0]
    return _refuse(
        "none",
        "NO_FIDUCIAL",
        "no fiducial configured: pass marker_length_mm or checkerboard + square_size_mm",
    )


def draw_marker(marker_id: int, side_px: int, dictionary: str = "DICT_4X4_50") -> np.ndarray:
    """Render a printable marker. Print it, measure the printed edge, pass that in mm."""
    adict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dictionary])
    return cv2.aruco.generateImageMarker(adict, marker_id, side_px)
