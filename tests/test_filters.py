"""Unit tests for pose.filters. Run with: python -m unittest discover tests"""

import math
import unittest

import numpy as np

from pose.filters import AngleFilter, OneEuroFilter

FPS = 30


class OneEuroFilterTest(unittest.TestCase):
    def test_constant_input_is_unchanged(self):
        f = OneEuroFilter(FPS)
        for _ in range(50):
            out = f.update(120.0)
        self.assertAlmostEqual(out, 120.0)

    def test_reduces_jitter_on_a_still_signal(self):
        rng = np.random.default_rng(0)
        noisy = 90 + rng.normal(0, 3, 300)
        f = OneEuroFilter(FPS)
        smoothed = np.array([f.update(x) for x in noisy])
        # The default beta favors low lag, so heavy white noise is cut by ~40%, not eliminated.
        self.assertLess(np.std(smoothed[30:]), np.std(noisy[30:]) * 0.7)

    def test_follows_a_ramp_with_small_lag(self):
        # 90 degrees per second, a typical exercise speed.
        f = OneEuroFilter(FPS)
        ramp = [90 + 3 * i for i in range(60)]
        out = [f.update(x) for x in ramp]
        self.assertLess(abs(out[-1] - ramp[-1]), 6.0)


class AngleFilterTest(unittest.TestCase):
    def test_single_frame_spike_is_rejected(self):
        f = AngleFilter(FPS)
        for _ in range(10):
            f.update(40.0)
        out = f.update(165.0)  # one-frame glitch
        self.assertTrue(f.rejected)
        self.assertAlmostEqual(out, 40.0, places=3)
        out = f.update(41.0)
        self.assertFalse(f.rejected)
        self.assertLess(abs(out - 40.5), 1.0)

    def test_persistent_jump_is_accepted_after_max_rejects(self):
        f = AngleFilter(FPS, max_rejects=3)
        for _ in range(10):
            f.update(40.0)
        outs = [f.update(160.0) for _ in range(4)]
        self.assertTrue(all(abs(o - 40.0) < 1e-3 for o in outs[:3]))
        self.assertAlmostEqual(outs[3], 160.0)

    def test_normal_motion_is_not_rejected(self):
        f = AngleFilter(FPS)  # 600 deg/s -> 20 deg/frame allowed
        for i in range(30):
            f.update(60 + 10 * i)
            self.assertFalse(f.rejected)

    def test_missing_samples_hold_then_drop(self):
        f = AngleFilter(FPS, max_hold=3)
        f.update(100.0)
        self.assertAlmostEqual(f.update(None), 100.0)
        self.assertAlmostEqual(f.update(float("nan")), 100.0)
        self.assertAlmostEqual(f.update(None), 100.0)
        self.assertIsNone(f.update(None))
        # After the reset, a new value is taken as-is rather than rejected as a jump.
        self.assertAlmostEqual(f.update(30.0), 30.0)

    def test_smooth_false_passes_values_through_but_still_rejects(self):
        f = AngleFilter(FPS, smooth=False)
        self.assertEqual(f.update(40.0), 40.0)
        self.assertEqual(f.update(47.0), 47.0)
        self.assertEqual(f.update(170.0), 47.0)
        self.assertTrue(f.rejected)

    def test_no_value_before_first_sample(self):
        self.assertIsNone(AngleFilter(FPS).update(None))


if __name__ == "__main__":
    unittest.main()
