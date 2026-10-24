"""Per-exercise annotators: turn a smoothed pose into measurements, overlays and a CSV row."""

import numpy as np
from mediapipe.tasks.python.vision import PoseLandmark

from .drawing import (GREEN, RED, SPINE_CONNECTIONS, WHITE, YELLOW, draw_angle_label, draw_panel,
                      draw_skeleton)
from .estimator import PoseResult
from .filters import AngleFilter
from .metrics import facing_score, knee_width_ratio, measure_joint, torso_lean
from .squat import FAULT_MESSAGES, KNEES_CAVING, TORSO_LEAN, SquatMetrics, SquatThresholds, SquatTracker

GRAY = (190, 190, 190)


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def _outward_offset(pose: PoseResult, vertex_x: int, width: int, mid_ids) -> tuple[int, int]:
    """Label offset that pushes a label away from the body's midline."""
    mid_x = pose.image[list(mid_ids), 0].mean() * width
    return (15 if vertex_x >= mid_x else -15, -15)


class ElbowAnnotator:
    """Milestone 2: draws the elbow angles next to each elbow."""

    JOINTS = ("left_elbow", "right_elbow")

    def __init__(self, fps: float, width: int, height: int, mode: str = "3d"):
        self.width, self.height, self.mode = width, height, mode
        self.filters = {name: AngleFilter(fps, smooth=False) for name in self.JOINTS}
        self.csv_fields = [f"{n}_{c}" for n in self.JOINTS for c in ("2d", "3d", "shown", "rejected")]
        self.rejected = 0

    def annotate(self, frame: np.ndarray, pose: PoseResult | None) -> dict:
        row = {}
        if pose is not None:
            draw_skeleton(frame, pose)
        for name in self.JOINTS:
            m = measure_joint(pose, name, self.width, self.height) if pose is not None else None
            angle = None if m is None else (m.angle_3d if self.mode == "3d" else m.angle_2d)
            shown = self.filters[name].update(angle)
            self.rejected += self.filters[name].rejected
            if m is not None:
                row[f"{name}_2d"] = round(m.angle_2d, 2)
                row[f"{name}_3d"] = round(m.angle_3d, 2)
                row[f"{name}_rejected"] = int(self.filters[name].rejected)
            if shown is not None:
                row[f"{name}_shown"] = round(shown, 2)
            if m is not None and shown is not None:
                offset = _outward_offset(pose, m.vertex_px[0], self.width,
                                         (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER))
                draw_angle_label(frame, m.vertex_px, shown, offset=offset)
        return row

    def summary(self) -> list[str]:
        return [f"Angle samples held as glitches: {self.rejected}"]

    def report(self) -> dict:
        return {"angles": {name: _round(f.value) for name, f in self.filters.items()}}


_TORSO = [(11, 12), (11, 23), (12, 24), (23, 24)] + SPINE_CONNECTIONS[:-1]  # all but neck-to-head
_LEGS = [(23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32)]
FAULT_SEGMENTS = {
    TORSO_LEAN: [frozenset(s) for s in _TORSO],
    KNEES_CAVING: [frozenset(s) for s in _LEGS],
}


class SquatAnnotator:
    """Milestone 3: squat rep counter with green/red form feedback."""

    def __init__(self, fps: float, width: int, height: int, thresholds: SquatThresholds = SquatThresholds(),
                 hud: bool = True):
        self.width, self.height, self.hud = width, height, hud
        self.tracker = SquatTracker(thresholds)
        self.filters = {k: AngleFilter(fps, smooth=False) for k in ("knee_left", "knee_right", "torso_lean")}
        self.csv_fields = ["knee_left", "knee_right", "torso_lean", "knee_width_ratio", "facing",
                           "phase", "reps", "faults"]
        self.state = None
        self.metrics = SquatMetrics()

    def annotate(self, frame: np.ndarray, pose: PoseResult | None) -> dict:
        knees = {
            side: measure_joint(pose, f"{side}_knee", self.width, self.height) if pose is not None else None
            for side in ("left", "right")
        }
        metrics = SquatMetrics(
            knee_left=self.filters["knee_left"].update(knees["left"].angle_3d if knees["left"] else None),
            knee_right=self.filters["knee_right"].update(knees["right"].angle_3d if knees["right"] else None),
            torso_lean=self.filters["torso_lean"].update(torso_lean(pose) if pose is not None else None),
            knee_width_ratio=knee_width_ratio(pose, self.width) if pose is not None else None,
            facing=facing_score(pose) if pose is not None else None,
        )
        self.metrics = metrics
        state = self.state = self.tracker.update(metrics)

        if pose is not None:
            colors = {seg: RED for fault in state.faults for seg in FAULT_SEGMENTS[fault]}
            draw_skeleton(frame, pose, line_color=GREEN, point_color=WHITE, segment_colors=colors)
            for side, value in (("left", metrics.knee_left), ("right", metrics.knee_right)):
                m = knees[side]
                if m is not None and value is not None:
                    offset = _outward_offset(pose, m.vertex_px[0], self.width,
                                             (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP))
                    draw_angle_label(frame, m.vertex_px, value, offset=offset)
        if self.hud:
            draw_panel(frame, self._panel(state, metrics))

        return {
            "knee_left": _round(metrics.knee_left), "knee_right": _round(metrics.knee_right),
            "torso_lean": _round(metrics.torso_lean), "knee_width_ratio": _round(metrics.knee_width_ratio),
            "facing": _round(metrics.facing), "phase": state.phase, "reps": state.reps,
            "faults": "|".join(sorted(state.faults)),
        }

    @staticmethod
    def _panel(state, metrics: SquatMetrics) -> list[tuple[str, tuple[int, int, int], float]]:
        knee = "--" if metrics.knee is None else f"{metrics.knee:.0f}"
        lean = "--" if metrics.torso_lean is None else f"{metrics.torso_lean:.0f}"
        lines = [
            (f"REPS {state.reps}", WHITE, 1.6),
            (f"{state.phase}   knee {knee}   torso {lean}", GRAY, 0.6),
        ]
        if state.faults:
            lines.append((" / ".join(FAULT_MESSAGES[f] for f in sorted(state.faults)), RED, 0.9))
        elif state.phase != "standing":
            lines.append(("Form OK", GREEN, 0.9))
        if state.last_rep is not None:
            good = state.last_rep.counted and not state.last_rep.faults
            lines.append((state.last_rep.message, GREEN if good else YELLOW, 0.6))
        return lines

    def report(self) -> dict:
        state, m = self.state, self.metrics
        if state is None:
            return {"reps": 0, "phase": "standing", "faults": [], "history": [], "knee": None, "torso_lean": None}
        return {
            "reps": state.reps,
            "phase": state.phase,
            "faults": [FAULT_MESSAGES[f] for f in sorted(state.faults)],
            "history": [
                {"message": r.message, "counted": r.counted, "good": r.counted and not r.faults,
                 "bottom_knee": round(r.bottom_angle, 1)}
                for r in state.history
            ],
            "knee": _round(m.knee), "torso_lean": _round(m.torso_lean),
        }

    def summary(self) -> list[str]:
        if self.state is None:
            return ["No frames processed"]
        lines = [f"Reps counted: {self.state.reps}"]
        lines += [f"  {r.message}" + (f" (bottom knee {r.bottom_angle:.0f} deg)" if r.counted else "")
                  for r in self.state.history]
        return lines
