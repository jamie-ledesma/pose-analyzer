"""Pose estimation: wraps MediaPipe's PoseLandmarker (BlazePose) Tasks API."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS = {
    "full": MODELS_DIR / "pose_landmarker_full.task",    # faster
    "heavy": MODELS_DIR / "pose_landmarker_heavy.task",  # slower, more accurate on hard poses
}
DEFAULT_MODEL = MODELS["full"]


@dataclass
class PoseResult:
    """Landmarks for one detected person in one frame.

    image: (33, 3) array of normalized image coords (x, y in [0, 1], z relative depth).
    world: (33, 3) array of 3D coords in meters, origin at the hip center.
    visibility: (33,) array of per-landmark visibility scores in [0, 1].
    """

    image: np.ndarray
    world: np.ndarray
    visibility: np.ndarray


class PoseEstimator:
    """Runs BlazePose on a stream of video frames."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Pose model not found: {model_path}")
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> PoseResult | None:
        """Returns landmarks for the first detected person, or None if nobody is found."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks:
            return None

        lms = result.pose_landmarks[0]
        world = result.pose_world_landmarks[0]
        return PoseResult(
            image=np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32),
            world=np.array([[lm.x, lm.y, lm.z] for lm in world], dtype=np.float32),
            visibility=np.array([lm.visibility or 0.0 for lm in lms], dtype=np.float32),
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
