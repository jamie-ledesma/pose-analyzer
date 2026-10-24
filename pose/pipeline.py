"""The shared analysis pipeline: pose estimation -> landmark smoothing -> exercise annotator.

Used by the command line (main.py), the web app's uploads, and its live camera mode.
"""

import csv
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .annotators import ElbowAnnotator, SquatAnnotator
from .drawing import draw_status
from .estimator import MODELS, PoseEstimator
from .landmark_filter import LandmarkSmoother
from .squat import SquatThresholds
from .video_io import VideoReader, VideoWriter

EXERCISES = ("squat", "elbows")


@dataclass(frozen=True)
class Settings:
    exercise: str = "squat"
    model: str = "full"
    mode: str = "3d"  # elbows only
    thresholds: SquatThresholds = field(default_factory=SquatThresholds)
    raw: bool = False


class Analyzer:
    """Annotates a stream of frames in place. Keeps per-stream state (smoothing, rep counts).

    With hud=False only the skeleton and joint angles are drawn, without the rep panel and status
    line: the web app shows those beside the video instead.
    """

    def __init__(self, fps: float, width: int, height: int, settings: Settings = Settings(), hud: bool = True):
        if settings.exercise not in EXERCISES:
            raise ValueError(f"Unknown exercise: {settings.exercise}")
        self.fps, self.width, self.height, self.settings, self.hud = fps, width, height, settings, hud
        self._estimator = PoseEstimator(MODELS[settings.model])
        self.frames = 0
        self.detected = 0
        self.landmarks_held = 0
        self.reset()

    def reset(self) -> None:
        """Starts a new set: clears smoothing state and rep counts."""
        s = self.settings
        self._smoother = None if s.raw else LandmarkSmoother(self.fps, self.width / self.height)
        if s.exercise == "squat":
            self.annotator = SquatAnnotator(self.fps, self.width, self.height, s.thresholds, self.hud)
        else:
            self.annotator = ElbowAnnotator(self.fps, self.width, self.height, s.mode)

    def process(self, frame: np.ndarray, timestamp_ms: int, status: str = "") -> dict:
        """Runs one frame (timestamps must strictly increase) and draws the overlay onto it."""
        self.frames += 1
        detection = self._estimator.process(frame, timestamp_ms)
        self.detected += detection is not None
        pose = detection if self._smoother is None else self._smoother.update(detection)
        if self._smoother is not None:
            self.landmarks_held += int(self._smoother.rejected.sum())
        row = self.annotator.annotate(frame, pose)
        if self.hud:
            draw_status(frame, status or f"pose: {'yes' if detection is not None else 'no'}")
        return row

    def report(self) -> dict:
        """JSON-friendly summary of the current set."""
        return {"exercise": self.settings.exercise, "frames": self.frames, "detected": self.detected,
                **self.annotator.report()}

    def summary(self) -> list[str]:
        return self.annotator.summary()

    def close(self) -> None:
        self._estimator.close()

    def __enter__(self) -> "Analyzer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def analyze_file(
    input_path: str | Path,
    output_path: str | Path,
    settings: Settings = Settings(),
    csv_path: str | Path | None = None,
    progress: Callable[[int, int], None] | None = None,
    hud: bool = True,
) -> dict:
    """Analyzes a video file and writes the annotated video (and optionally a CSV). Returns the report."""
    rows = []
    start = time.perf_counter()
    with (
        VideoReader(input_path) as reader,
        VideoWriter(output_path, reader.fps, reader.width, reader.height) as writer,
        Analyzer(reader.fps, reader.width, reader.height, settings, hud) as analyzer,
    ):
        for index, timestamp_ms, frame in reader:
            row = analyzer.process(frame, timestamp_ms, status=f"frame {index}")
            rows.append({"frame": index, **row})
            writer.write(frame)
            if progress is not None:
                progress(index + 1, reader.frame_count)
        report = analyzer.report()
        report.update(
            seconds=round(time.perf_counter() - start, 2),
            video={"width": reader.width, "height": reader.height, "fps": reader.fps},
            landmarks_held=analyzer.landmarks_held,
            summary=analyzer.summary(),
        )
        fields = ["frame"] + analyzer.annotator.csv_fields

    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    return report
