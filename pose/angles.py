"""Pure geometry utilities for joint angles. No MediaPipe or OpenCV dependencies."""

import numpy as np

_EPS = 1e-9


def joint_angle(a, b, c) -> float:
    """Returns the angle ABC in degrees (the angle at vertex B), in [0, 180].

    Accepts 2D or 3D points (any array-like of length 2 or 3; all three must match).
    Uses atan2(|BA x BC|, BA . BC), which stays accurate near 0 and 180 degrees
    where the arccos-of-dot-product formulation loses precision.

    Returns NaN if either segment has zero length (the angle is undefined).
    """
    a, b, c = (np.asarray(p, dtype=np.float64) for p in (a, b, c))
    if not (a.shape == b.shape == c.shape) or a.shape not in ((2,), (3,)):
        raise ValueError(f"Points must all be 2D or all 3D, got shapes {a.shape}, {b.shape}, {c.shape}")

    ba = a - b
    bc = c - b
    if np.linalg.norm(ba) < _EPS or np.linalg.norm(bc) < _EPS:
        return float("nan")

    if ba.shape == (2,):
        cross = abs(ba[0] * bc[1] - ba[1] * bc[0])
    else:
        cross = np.linalg.norm(np.cross(ba, bc))
    dot = float(np.dot(ba, bc))
    return float(np.degrees(np.arctan2(cross, dot)))


def angle_from_vertical(top, bottom, up=None) -> float:
    """Returns the angle in degrees between the segment bottom->top and straight up, in [0, 180].

    0 means the segment points straight up, 90 means horizontal. `up` defaults to -y, i.e.
    (0, -1) in 2D or (0, -1, 0) in 3D, matching image coordinates and MediaPipe world
    landmarks, where y points down.
    """
    bottom = np.asarray(bottom, dtype=np.float64)
    if bottom.shape not in ((2,), (3,)):
        raise ValueError(f"Points must be 2D or 3D, got shape {bottom.shape}")
    if up is None:
        up = np.zeros_like(bottom)
        up[1] = -1.0
    return joint_angle(bottom + np.asarray(up, dtype=np.float64), bottom, top)
