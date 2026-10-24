"""Upload mode: analyzes uploaded videos in background threads."""

import csv
import json
import shutil
import threading
import time
import uuid
from pathlib import Path

import cv2
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from pose.pipeline import Settings, analyze_file

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}
EXAMPLE_ID = "example"


class JobStore:
    """Keeps uploaded videos and their results under `root/<job id>/`."""

    def __init__(self, root: Path, max_age: float | None = None):
        """max_age: seconds after which a finished upload and its files are deleted (None keeps them)."""
        self.root = root
        self.max_age = max_age
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, dict] = {}
        self._born: dict[str, float] = {}
        self._lock = threading.Lock()

    def create(self, upload: FileStorage, settings: Settings) -> str:
        name = secure_filename(upload.filename or "") or "video.mp4"
        ext = Path(name).suffix.lower()
        if ext not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{ext or name}'. Use one of: {', '.join(sorted(VIDEO_EXTENSIONS))}")

        self._prune()
        job_id, folder = self._new_folder()
        input_path = folder / f"input{ext}"
        upload.save(input_path)
        self._start(job_id, name, input_path, settings)
        return job_id

    def create_from_path(self, input_path: Path, settings: Settings) -> str:
        """Analyzes a video already on disk (the example clip) without copying it."""
        job_id, _ = self._new_folder()
        self._start(job_id, input_path.name, input_path, settings)
        return job_id

    def add_example(self, input_path: Path, sources: list[Path]) -> None:
        """Registers the example clip, analyzed with default settings, as the job "example".

        The result is cached on disk and reused until the clip or any of `sources` (the
        analysis code) is newer than it.
        """
        folder = self.root / EXAMPLE_ID
        folder.mkdir(exist_ok=True)
        report_path = folder / "report.json"
        newest = max(p.stat().st_mtime for p in [input_path, *sources])
        if report_path.exists() and report_path.stat().st_mtime > newest and (folder / "result.mp4").exists():
            with self._lock:
                self._jobs[EXAMPLE_ID] = {"id": EXAMPLE_ID, "filename": input_path.name, "status": "done",
                                          "progress": 1.0, "report": json.loads(report_path.read_text()),
                                          "error": None}
            return
        report_path.unlink(missing_ok=True)
        self._start(EXAMPLE_ID, input_path.name, input_path, Settings(), report_path=report_path)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def folder(self, job_id: str) -> Path | None:
        """The job's folder, or None if the id is unknown (guards against path tricks)."""
        with self._lock:
            return self.root / job_id if job_id in self._jobs else None

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["status"] == "running" or job_id == EXAMPLE_ID:
                return False
            del self._jobs[job_id]
        shutil.rmtree(self.root / job_id, ignore_errors=True)
        return True

    def _new_folder(self) -> tuple[str, Path]:
        job_id = uuid.uuid4().hex[:12]
        folder = self.root / job_id
        folder.mkdir()
        self._born[job_id] = time.monotonic()
        return job_id, folder

    def _prune(self) -> None:
        """Deletes finished jobs older than max_age. The example is never pruned."""
        if self.max_age is None:
            return
        with self._lock:
            old = [j for j, born in self._born.items()
                   if time.monotonic() - born > self.max_age and self._jobs.get(j, {}).get("status") != "running"]
            for job_id in old:
                self._jobs.pop(job_id, None)
                del self._born[job_id]
        for job_id in old:
            shutil.rmtree(self.root / job_id, ignore_errors=True)

    def _start(self, job_id: str, name: str, input_path: Path, settings: Settings,
               report_path: Path | None = None) -> None:
        with self._lock:
            self._jobs[job_id] = {"id": job_id, "filename": name, "status": "running", "progress": 0.0,
                                  "report": None, "error": None}
        threading.Thread(target=self._run, args=(job_id, input_path, settings, report_path), daemon=True).start()

    def _update(self, job_id: str, **fields) -> None:
        with self._lock:
            self._jobs[job_id].update(fields)

    def _run(self, job_id: str, input_path: Path, settings: Settings, report_path: Path | None) -> None:
        folder = self.root / job_id

        def progress(done: int, total: int) -> None:
            if total > 0:
                self._update(job_id, progress=min(done / total, 1.0))

        try:
            report = analyze_file(input_path, folder / "result.mp4", settings, folder / "data.csv", progress,
                                  hud=False)
            write_poster(folder / "result.mp4", folder / "data.csv", folder / "poster.jpg")
            if report_path is not None:
                report_path.write_text(json.dumps(report))
            self._update(job_id, status="done", progress=1.0, report=report)
        except Exception as exc:  # report any failure (bad codec, unreadable file) to the page
            self._update(job_id, status="error", error=str(exc))


def write_poster(video_path: Path, csv_path: Path, poster_path: Path) -> None:
    """Saves the most telling frame as the video's poster: the deepest squat, or the middle frame."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    index = len(rows) // 2
    knees = [(sum(vals) / len(vals), i) for i, row in enumerate(rows)
             if (vals := [float(row[k]) for k in ("knee_left", "knee_right") if row.get(k)])]
    if knees:
        index = min(knees)[1]
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(str(poster_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
