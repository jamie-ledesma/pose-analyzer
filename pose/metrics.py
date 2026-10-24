"""Body-specific measurements built on pose landmarks and the pure math in angles.py."""

from dataclasses import dataclass

import numpy as np
from mediapipe.tasks.python.vision import PoseLandmark

from .angles import angle_from_vertical, joint_angle
from .estimator import PoseResult

# (proximal, joint, distal) landmark triples, keyed by "<side>_<joint>".
JOINTS = {
    "left_elbow": (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    "right_elbow": (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    "left_knee": (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),
    "right_knee": (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE),
}

SHOULDERS = (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER)
HIPS = (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP)
KNEES = (PoseLandmark.LEFT_KNEE, PoseLandmark.RIGHT_KNEE)
ANKLES = (PoseLandmark.LEFT_ANKLE, PoseLandmark.RIGHT_ANKLE)


@dataclass
class JointMeasurement:
    name: str
    vertex_px: tuple[int, int]  # pixel location of the joint, for drawing
    angle_2d: float             # angle in the image plane (pixel space), degrees
    angle_3d: float             # angle from MediaPipe world landmarks, degrees


def measure_joint(
    pose: PoseResult, name: str, width: int, height: int, min_visibility: float = 0.5
) -> JointMeasurement | None:
    """Measures one joint, or returns None if any of its three landmarks is poorly visible."""
    ids = JOINTS[name]
    if any(pose.visibility[i] < min_visibility for i in ids):
        return None

    # Normalized x and y are scaled by different amounts (width vs height), so convert
    # to pixels first; otherwise 2D angles are distorted on non-square video.
    px = pose.image[:, :2] * np.array([width, height])
    a, b, c = (px[i] for i in ids)
    wa, wb, wc = (pose.world[i] for i in ids)

    return JointMeasurement(
        name=name,
        vertex_px=(int(round(b[0])), int(round(b[1]))),
        angle_2d=joint_angle(a, b, c),
        angle_3d=joint_angle(wa, wb, wc),
    )


def _visible(pose: PoseResult, ids, min_visibility: float) -> bool:
    return all(pose.visibility[i] >= min_visibility for i in ids)


def torso_lean(pose: PoseResult, min_visibility: float = 0.5) -> float | None:
    """Degrees the torso (hip center to shoulder center) leans away from vertical, in 3D.

    Uses world landmarks, so forward lean toward or away from the camera is measured too.
    Assumes a level camera, since "vertical" is the camera's up direction.
    """
    if not _visible(pose, SHOULDERS + HIPS, min_visibility):
        return None
    shoulders = pose.world[list(SHOULDERS)].mean(axis=0)
    hips = pose.world[list(HIPS)].mean(axis=0)
    return angle_from_vertical(shoulders, hips)


def knee_width_ratio(pose: PoseResult, width: int, min_visibility: float = 0.5) -> float | None:
    """Horizontal distance between the knees divided by the distance between the ankles, in pixels.

    Below ~1 while squatting means the knees are inside the feet (caving in). Only meaningful
    when the person faces the camera; see `facing_score`.
    """
    if not _visible(pose, KNEES + ANKLES, min_visibility):
        return None
    x = pose.image[:, 0] * width
    ankle_gap = abs(x[ANKLES[0]] - x[ANKLES[1]])
    if ankle_gap < 1.0:
        return None
    return abs(x[KNEES[0]] - x[KNEES[1]]) / ankle_gap


def facing_score(pose: PoseResult) -> float:
    """How squarely the hips face the camera: 1 = straight on, 0 = side-on. From world landmarks."""
    hip_axis = pose.world[HIPS[0]] - pose.world[HIPS[1]]
    horizontal = np.hypot(hip_axis[0], hip_axis[2])
    return float(abs(hip_axis[0]) / horizontal) if horizontal > 1e-6 else 0.0
