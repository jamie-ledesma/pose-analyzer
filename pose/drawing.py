"""Rendering helpers for skeletons and overlays."""

import cv2
import numpy as np
from mediapipe.tasks.python.vision import PoseLandmarksConnections

from .estimator import PoseResult

POSE_CONNECTIONS = [(c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS]

# Extra touchpoints derived from the 33 BlazePose landmarks, numbered from 33 on.
# Each is (a, b, t): the point a + t * (b - a), where a and b may be earlier derived points.
NECK, PELVIS, SPINE_LOW, SPINE_MID, SPINE_HIGH, HEAD = range(33, 39)
DERIVED_POINTS = [
    (11, 12, 0.5),            # neck: between the shoulders
    (23, 24, 0.5),            # pelvis: between the hips
    (PELVIS, NECK, 0.25),     # lumbar spine
    (PELVIS, NECK, 0.5),      # mid spine
    (PELVIS, NECK, 0.75),     # upper spine
    (7, 8, 0.5),              # head center: between the ears
    # Midpoints of each long bone (upper arms, forearms, thighs, shins) and the sides of the torso.
    (11, 13, 0.5), (13, 15, 0.5), (12, 14, 0.5), (14, 16, 0.5),
    (23, 25, 0.5), (25, 27, 0.5), (24, 26, 0.5), (26, 28, 0.5),
    (11, 23, 0.5), (12, 24, 0.5),
]
LIMB_MIDPOINTS = set(range(HEAD + 1, 33 + len(DERIVED_POINTS)))
SPINE_CONNECTIONS = [(PELVIS, SPINE_LOW), (SPINE_LOW, SPINE_MID), (SPINE_MID, SPINE_HIGH),
                     (SPINE_HIGH, NECK), (NECK, HEAD)]

# BGR colors
WHITE = (255, 255, 255)
GREEN = (80, 200, 60)
RED = (40, 40, 230)
YELLOW = (0, 200, 255)
LANDMARK_COLOR = (0, 0, 255)
CONNECTION_COLOR = (255, 255, 255)


def to_pixels(pose: PoseResult, width: int, height: int) -> np.ndarray:
    """Converts normalized image landmarks to integer pixel coordinates, shape (33, 2)."""
    return np.round(pose.image[:, :2] * [width, height]).astype(int)


def with_derived_points(pts: np.ndarray, visibility: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Appends the DERIVED_POINTS to (33, 2) points and (33,) visibility.

    A derived point is only as visible as the less visible of its two sources.
    """
    pts, visibility = list(pts.astype(float)), list(visibility)
    for a, b, t in DERIVED_POINTS:
        pts.append(pts[a] + t * (pts[b] - pts[a]))
        visibility.append(min(visibility[a], visibility[b]))
    return np.array(pts), np.array(visibility)


def draw_skeleton(
    frame: np.ndarray,
    pose: PoseResult,
    line_color: tuple[int, int, int] = CONNECTION_COLOR,
    point_color: tuple[int, int, int] = LANDMARK_COLOR,
    min_visibility: float = 0.5,
    segment_colors: dict[frozenset[int], tuple[int, int, int]] | None = None,
) -> None:
    """Draws pose connections and landmarks onto the frame in place.

    `segment_colors` overrides `line_color` for specific connections, keyed by
    frozenset({landmark_a, landmark_b}).
    """
    segment_colors = segment_colors or {}
    h, w = frame.shape[:2]
    pts, visibility = with_derived_points(pose.image[:, :2] * [w, h], pose.visibility)
    pts = np.round(pts).astype(int)
    visible = visibility >= min_visibility
    thickness = max(2, w // 400)

    for a, b in POSE_CONNECTIONS + SPINE_CONNECTIONS:
        if visible[a] and visible[b]:
            color = segment_colors.get(frozenset((a, b)), line_color)
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, thickness, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        if visible[i]:
            # Limb midpoints sit on a line, so draw them smaller than the joints.
            radius = thickness + 1 if i in LIMB_MIDPOINTS else thickness + 2
            cv2.circle(frame, (x, y), radius, point_color, -1, cv2.LINE_AA)


def draw_angle_label(
    frame: np.ndarray,
    vertex: tuple[int, int],
    angle_deg: float,
    offset: tuple[int, int] = (15, -15),
    color: tuple[int, int, int] = WHITE,
) -> None:
    """Draws an angle value (e.g. "142°") on a dark box next to a joint."""
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, max(0.5, frame.shape[1] / 1100)
    thickness = max(1, int(scale * 2))
    text = f"{angle_deg:.0f}"
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    deg_r = max(2, th // 5)  # Hershey fonts have no degree sign, so draw a small ring.

    x, y = vertex[0] + offset[0], vertex[1] + offset[1]
    if offset[0] < 0:
        x -= tw + 3 * deg_r
    pad = 4
    cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + 3 * deg_r + pad, y + baseline), (20, 20, 20), -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    cv2.circle(frame, (x + tw + deg_r + 1, y - th + deg_r), deg_r, color, max(1, thickness // 2), cv2.LINE_AA)


def draw_panel(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int], float]]) -> None:
    """Draws a translucent panel in the top-left corner with lines of (text, color, relative size)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    base = max(0.6, frame.shape[1] / 900)
    pad, gap = int(12 * base), int(10 * base)
    sizes = []
    for text, _, rel in lines:
        scale = base * rel
        thickness = max(1, int(round(scale * 2)))
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        sizes.append((scale, thickness, tw, th))
    width = max(tw for _, _, tw, _ in sizes) + 2 * pad
    height = sum(th for *_, th in sizes) + gap * (len(lines) - 1) + 2 * pad

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, dst=frame)

    y = pad
    for (text, color, _), (scale, thickness, _, th) in zip(lines, sizes):
        y += th
        cv2.putText(frame, text, (pad, y), font, scale, color, thickness, cv2.LINE_AA)
        y += gap


def draw_status(frame: np.ndarray, text: str) -> None:
    """Draws a small status line in the bottom-left corner."""
    h = frame.shape[0]
    cv2.putText(frame, text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
