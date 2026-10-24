"""Video input/output wrappers around OpenCV."""

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class VideoReader:
    """Iterates over a video file, yielding (frame_index, timestamp_ms, BGR frame)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Input video not found: {self.path}")
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise IOError(f"OpenCV could not open video: {self.path}")

        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> Iterator[tuple[int, int, np.ndarray]]:
        index = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            # Derive timestamps from the frame index so they are strictly increasing,
            # which MediaPipe's VIDEO running mode requires.
            timestamp_ms = int(round(index * 1000.0 / self.fps))
            yield index, timestamp_ms, frame
            index += 1

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class VideoWriter:
    """Writes BGR frames to an .mp4 file."""

    def __init__(self, path: str | Path, fps: float, width: int, height: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # H.264 ("avc1") plays in browsers; fall back to MPEG-4 Part 2 if this OpenCV build lacks it.
        for codec in ("avc1", "mp4v"):
            self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
            if self._writer.isOpened():
                self.codec = codec
                break
        else:
            raise IOError(f"OpenCV could not open writer for: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
