"""Live camera mode: reads a camera in a background thread and keeps the latest annotated frame."""

import threading
import time
from collections.abc import Iterator

import cv2

from pose.pipeline import Analyzer, Settings


class LiveSession:
    """One shared live session. `source` is a camera index (0 = built-in camera) or a video
    file path, which is looped and paced like a camera (useful for testing without a camera)."""

    def __init__(self, source: int | str = 0):
        self.source = source
        self._cond = threading.Condition()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._settings = Settings()
        self._settings_changed = False
        self._reset_requested = False
        self._jpeg: bytes | None = None
        self._frame_id = 0
        self._report: dict = {}
        self._fps = 0.0
        self.error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, settings: Settings) -> None:
        self.apply_settings(settings)
        if self.running:
            return
        self.error = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="live-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        with self._cond:
            self._jpeg = None
            self._report = {}  # don't show the old set's reps when the camera restarts
            self._fps = 0.0
            self._cond.notify_all()

    def reset(self) -> None:
        self._reset_requested = True

    def apply_settings(self, settings: Settings) -> None:
        if settings != self._settings:
            self._settings = settings
            self._settings_changed = True

    def status(self) -> dict:
        with self._cond:
            return {"running": self.running, "error": self.error, "fps": round(self._fps, 1),
                    "report": self._report}

    def mjpeg(self) -> Iterator[bytes]:
        """Yields the annotated frames as a multipart MJPEG stream until the session stops."""
        last = -1
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._frame_id != last or not self.running, timeout=2)
                if not self.running:
                    return
                if self._frame_id == last:
                    continue
                last, jpeg = self._frame_id, self._jpeg
            if jpeg is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    def _run(self) -> None:
        is_file = isinstance(self.source, str)
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error = ("Could not open the camera. On macOS, allow camera access for the app that "
                          "started the server (System Settings > Privacy & Security > Camera)."
                          if not is_file else f"Could not open video source: {self.source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_period = 1.0 / fps
        analyzer: Analyzer | None = None
        start = time.monotonic()
        last_ts = -1
        tick, ticks = time.monotonic(), 0
        try:
            while not self._stop.is_set():
                loop_start = time.monotonic()
                ok, frame = cap.read()
                if not ok:
                    if is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self.error = "The camera stopped sending frames."
                    break
                if not is_file:
                    frame = cv2.flip(frame, 1)  # mirror, like a selfie view

                if analyzer is None or self._settings_changed:
                    if analyzer is not None:
                        analyzer.close()
                    self._settings_changed = False
                    h, w = frame.shape[:2]
                    analyzer = Analyzer(fps, w, h, self._settings, hud=False)
                if self._reset_requested:
                    self._reset_requested = False
                    analyzer.reset()

                # MediaPipe needs strictly increasing timestamps.
                ts = max(int((time.monotonic() - start) * 1000), last_ts + 1)
                last_ts = ts
                analyzer.process(frame, ts)
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

                ticks += 1
                now = time.monotonic()
                if now - tick >= 1.0:
                    self._fps, tick, ticks = ticks / (now - tick), now, 0
                with self._cond:
                    if ok:
                        self._jpeg = buf.tobytes()
                        self._frame_id += 1
                    self._report = analyzer.report()
                    self._cond.notify_all()

                if is_file:  # pace a file like a real camera
                    time.sleep(max(0.0, frame_period - (time.monotonic() - loop_start)))
        finally:
            cap.release()
            if analyzer is not None:
                analyzer.close()
            with self._cond:
                self._cond.notify_all()
