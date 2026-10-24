"""Unit tests for pose.squat and angle_from_vertical. Run with: python -m unittest discover tests"""

import math
import unittest

import numpy as np

from pose.angles import angle_from_vertical
from pose.squat import KNEES_CAVING, TORSO_LEAN, SquatMetrics, SquatThresholds, SquatTracker


class AngleFromVerticalTest(unittest.TestCase):
    def test_upright_is_zero(self):
        self.assertAlmostEqual(angle_from_vertical((0, -1), (0, 0)), 0.0)

    def test_horizontal_is_90(self):
        self.assertAlmostEqual(angle_from_vertical((1, 0), (0, 0)), 90.0)
        self.assertAlmostEqual(angle_from_vertical((-1, 0), (0, 0)), 90.0)

    def test_45_degree_lean_in_image_coords(self):
        # y points down, so "up and to the right" is (+1, -1).
        self.assertAlmostEqual(angle_from_vertical((101, 99), (100, 100)), 45.0)

    def test_3d_lean_toward_camera(self):
        # 30 degree lean in the z direction, invisible in a 2D front view.
        top = (0, -math.cos(math.radians(30)), math.sin(math.radians(30)))
        self.assertAlmostEqual(angle_from_vertical(top, (0, 0, 0)), 30.0)

    def test_upside_down_is_180(self):
        self.assertAlmostEqual(angle_from_vertical((0, 1, 0), (0, 0, 0)), 180.0)

    def test_bad_shape_raises(self):
        with self.assertRaises(ValueError):
            angle_from_vertical((1,), (0,))


def squat_curve(bottom: float, frames: int = 40, top: float = 168.0):
    """Knee angles for one rep: top -> bottom -> top, cosine-shaped."""
    t = np.linspace(0, 2 * np.pi, frames)
    return top - (top - bottom) * (1 - np.cos(t)) / 2


def feed(tracker, knees, **metrics):
    state = None
    for k in knees:
        state = tracker.update(SquatMetrics(knee_left=k, knee_right=k, facing=0.9, **metrics))
    return state


class SquatTrackerRepTest(unittest.TestCase):
    def test_full_depth_reps_are_counted(self):
        tracker = SquatTracker()
        state = feed(tracker, np.concatenate([squat_curve(85), squat_curve(90), [168] * 5]))
        self.assertEqual(state.reps, 2)
        self.assertEqual(state.phase, "standing")
        self.assertEqual(state.last_rep.message, "Rep 2: good")

    def test_rep_not_counted_until_standing(self):
        tracker = SquatTracker()
        state = feed(tracker, squat_curve(85)[:30])  # bottom reached, still rising at ~130 degrees
        self.assertEqual(state.reps, 0)
        self.assertEqual(state.phase, "ascending")

    def test_shallow_rep_is_flagged_not_counted(self):
        tracker = SquatTracker()
        state = feed(tracker, squat_curve(120))
        self.assertEqual(state.reps, 0)
        self.assertFalse(state.last_rep.counted)
        self.assertIn("go deeper", state.last_rep.message)

    def test_tiny_dip_is_ignored(self):
        tracker = SquatTracker()
        state = feed(tracker, squat_curve(145))
        self.assertEqual(state.reps, 0)
        self.assertIsNone(state.last_rep)

    def test_phases(self):
        tracker = SquatTracker()
        knees = squat_curve(85, frames=60)
        phases = [tracker.update(SquatMetrics(knee_left=k, knee_right=k)).phase for k in knees]
        self.assertEqual(phases[0], "standing")
        self.assertIn("descending", phases[:30])
        self.assertIn("bottom", phases)
        self.assertIn("ascending", phases[30:])
        self.assertEqual(phases[-1], "standing")

    def test_missing_frames_hold_state(self):
        tracker = SquatTracker()
        feed(tracker, squat_curve(85)[:20])
        state = tracker.update(SquatMetrics())
        self.assertNotEqual(state.phase, "standing")

    def test_uses_one_knee_when_other_missing(self):
        tracker = SquatTracker()
        for k in np.concatenate([squat_curve(85), [168] * 3]):
            state = tracker.update(SquatMetrics(knee_left=k))
        self.assertEqual(state.reps, 1)


class SquatTrackerFormTest(unittest.TestCase):
    def test_excessive_lean_is_flagged_after_debounce(self):
        tracker = SquatTracker(SquatThresholds(fault_frames=3))
        knees = squat_curve(85)
        states = [tracker.update(SquatMetrics(knee_left=k, knee_right=k, torso_lean=60.0)) for k in knees]
        in_rep = [i for i, k in enumerate(knees) if k < 150]
        first = in_rep[0]
        # The rep starts once the knee drops below 150; the lean must then persist 3 frames.
        self.assertNotIn(TORSO_LEAN, states[first + 1].faults)
        self.assertIn(TORSO_LEAN, states[first + 2].faults)
        state = feed(tracker, [168] * 3, torso_lean=60.0)
        self.assertEqual(state.reps, 1)
        self.assertIn("chest up", state.last_rep.message)

    def test_lean_not_checked_while_standing(self):
        tracker = SquatTracker()
        state = feed(tracker, [168] * 20, torso_lean=60.0)
        self.assertEqual(state.faults, frozenset())

    def test_single_frame_fault_is_ignored(self):
        tracker = SquatTracker(SquatThresholds(fault_frames=3))
        knees = squat_curve(85)
        for i, k in enumerate(knees):
            lean = 60.0 if i == 20 else 20.0
            state = tracker.update(SquatMetrics(knee_left=k, knee_right=k, torso_lean=lean))
            self.assertNotIn(TORSO_LEAN, state.faults)
        self.assertEqual(state.last_rep.message, "Rep 1: good")

    def test_knees_caving_flagged_when_bent_and_facing_camera(self):
        tracker = SquatTracker()
        state = feed(tracker, squat_curve(85)[:20], knee_width_ratio=0.7)
        self.assertIn(KNEES_CAVING, state.faults)

    def test_narrow_knees_ok_while_standing(self):
        # Wide stance: knees inside ankles is normal when standing.
        tracker = SquatTracker()
        state = feed(tracker, [168] * 10 + [140] * 10, knee_width_ratio=0.7)
        self.assertNotIn(KNEES_CAVING, state.faults)

    def test_knee_check_skipped_in_side_view(self):
        tracker = SquatTracker()
        for k in squat_curve(85)[:20]:
            state = tracker.update(SquatMetrics(knee_left=k, knee_right=k, knee_width_ratio=0.5, facing=0.2))
        self.assertNotIn(KNEES_CAVING, state.faults)

    def test_faults_clear_after_debounce_and_reset_between_reps(self):
        tracker = SquatTracker()
        feed(tracker, squat_curve(85)[:20], knee_width_ratio=0.7)
        state = feed(tracker, squat_curve(85)[20:], knee_width_ratio=1.2)
        self.assertEqual(state.faults, frozenset())
        self.assertIn("knees out", state.last_rep.message)
        state = feed(tracker, np.concatenate([squat_curve(85), [168] * 3]), knee_width_ratio=1.2)
        self.assertEqual(state.last_rep.message, "Rep 2: good")


if __name__ == "__main__":
    unittest.main()
