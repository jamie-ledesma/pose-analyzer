"""Local web app for the exercise analyzer: an example clip, live camera mode and video upload mode.

Usage:
    python app.py [--port 5055] [--camera 0]

Then open http://127.0.0.1:5055 in a browser. --camera takes a camera index (0 is the
built-in MacBook camera) or a video file path to use as a stand-in camera for testing.
The server only listens on this computer (127.0.0.1).
"""

import argparse
import os
import re
from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory

from pose.estimator import MODELS
from pose.pipeline import EXERCISES, Settings
from pose.squat import SquatThresholds
from webapp.jobs import EXAMPLE_ID, JobStore
from webapp.live import LiveSession

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "samples" / "squat_front.mp4"
UPLOAD_LIFETIME_S = 3600  # hosted mode: uploads and results are deleted after an hour
JOB_ID = re.compile(rf"^([0-9a-f]{{12}}|{EXAMPLE_ID})$")


def parse_settings(data) -> Settings:
    """Builds Settings from form or JSON fields, falling back to defaults for anything missing."""
    defaults = SquatThresholds()
    exercise = data.get("exercise", "squat")
    model = data.get("model", "full")
    if exercise not in EXERCISES or model not in MODELS:
        abort(400, "Unknown exercise or model")
    try:
        max_lean = float(data.get("max_lean", defaults.max_torso_lean))
        depth = float(data.get("depth", defaults.depth))
    except (TypeError, ValueError):
        abort(400, "Thresholds must be numbers")
    return Settings(exercise=exercise, model=model,
                    thresholds=SquatThresholds(max_torso_lean=max_lean, depth=depth))


def create_app(camera: int | str = 0, runs_dir: Path = ROOT / "runs", hosted: bool | None = None) -> Flask:
    """hosted=True (or HOSTED=1 in the environment) is for running on a server: no live camera,
    smaller uploads, and finished uploads deleted after an hour."""
    if hosted is None:
        hosted = os.environ.get("HOSTED") == "1"
    app = Flask(__name__, template_folder="webapp/templates", static_folder="webapp/static")
    app.config["MAX_CONTENT_LENGTH"] = (200 * 1024**2) if hosted else (2 * 1024**3)
    live = LiveSession(camera)
    jobs = JobStore(runs_dir, max_age=UPLOAD_LIFETIME_S if hosted else None)
    # Analysis code the cached example depends on: re-analyze it when any of this changes.
    jobs.add_example(SAMPLE, sources=sorted((ROOT / "pose").glob("*.py")))

    @app.get("/")
    def index():
        defaults = SquatThresholds()
        return render_template("index.html", max_lean=defaults.max_torso_lean, depth=defaults.depth,
                               camera_is_file=isinstance(camera, str), hosted=hosted)

    @app.before_request
    def no_live_camera_on_a_server():
        # A server's camera is not the visitor's, and video would leave the visitor's device.
        if hosted and (request.path.startswith("/api/live") or request.path == "/live.mjpg"):
            abort(404, "The live camera is only available when the app runs on your own computer")

    # Live camera
    @app.post("/api/live/start")
    def live_start():
        live.start(parse_settings(request.get_json(silent=True) or {}))
        return jsonify(live.status())

    @app.post("/api/live/stop")
    def live_stop():
        live.stop()
        return jsonify(live.status())

    @app.post("/api/live/reset")
    def live_reset():
        live.reset()
        return jsonify(live.status())

    @app.post("/api/live/settings")
    def live_settings():
        live.apply_settings(parse_settings(request.get_json(silent=True) or {}))
        return jsonify(live.status())

    @app.get("/api/live/status")
    def live_status():
        return jsonify(live.status())

    @app.get("/live.mjpg")
    def live_stream():
        if not live.running:
            abort(409, "Live camera is not running")
        return Response(live.mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-store"})

    # Uploads
    @app.post("/api/upload")
    def upload():
        file = request.files.get("video")
        if file is None or not file.filename:
            abort(400, "No video file in the request")
        try:
            job_id = jobs.create(file, parse_settings(request.form))
        except ValueError as exc:
            abort(400, str(exc))
        return jsonify(jobs.get(job_id)), 202

    @app.post("/api/example")
    def example_rerun():
        settings = parse_settings(request.get_json(silent=True) or {})
        job_id = EXAMPLE_ID if settings == Settings() else jobs.create_from_path(SAMPLE, settings)
        return jsonify(jobs.get(job_id)), 202

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id):
        job = jobs.get(job_id) if JOB_ID.match(job_id) else None
        if job is None:
            abort(404)
        return jsonify(job)

    @app.delete("/api/jobs/<job_id>")
    def job_delete(job_id):
        if not JOB_ID.match(job_id) or not jobs.delete(job_id):
            abort(404)
        return "", 204

    @app.get("/runs/<job_id>/<name>")
    def job_file(job_id, name):
        folder = jobs.folder(job_id) if JOB_ID.match(job_id) else None
        if folder is None or name not in ("result.mp4", "data.csv", "poster.jpg"):
            abort(404)
        download = request.args.get("download") is not None
        return send_from_directory(folder, name, as_attachment=download,
                                   download_name=f"{job_id}-{name}" if download else None)

    @app.errorhandler(400)
    @app.errorhandler(404)
    @app.errorhandler(409)
    @app.errorhandler(413)
    def error(exc):
        if exc.code == 413:
            return jsonify({"error": f"That video is too large. The limit is {app.config['MAX_CONTENT_LENGTH'] // 1024**2} MB."}), 413
        return jsonify({"error": exc.description}), exc.code

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5055)))
    parser.add_argument("--host", default="127.0.0.1",
                        help="address to listen on; the default only accepts connections from this computer")
    parser.add_argument("--camera", default="0",
                        help="camera index (0 = built-in camera) or a video file to loop as a stand-in camera")
    args = parser.parse_args()
    camera = int(args.camera) if args.camera.isdigit() else args.camera
    app = create_app(camera)
    print(f"Open http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
