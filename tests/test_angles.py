"""Unit tests for pose.angles. Run with: python -m unittest discover tests"""

import math
import unittest

import numpy as np

from pose.angles import joint_angle


class JointAngle2DTest(unittest.TestCase):
    def test_right_angle(self):
        self.assertAlmostEqual(joint_angle((1, 0), (0, 0), (0, 1)), 90.0)

    def test_straight_line_is_180(self):
        self.assertAlmostEqual(joint_angle((-1, 0), (0, 0), (1, 0)), 180.0)

    def test_fully_folded_is_0(self):
        self.assertAlmostEqual(joint_angle((2, 0), (0, 0), (1, 0)), 0.0)

    def test_45_and_135(self):
        self.assertAlmostEqual(joint_angle((1, 0), (0, 0), (1, 1)), 45.0)
        self.assertAlmostEqual(joint_angle((1, 0), (0, 0), (-1, 1)), 135.0)

    def test_order_of_outer_points_does_not_matter(self):
        a, b, c = (3, 1), (0.5, -2), (-4, 2)
        self.assertAlmostEqual(joint_angle(a, b, c), joint_angle(c, b, a))

    def test_invariant_to_translation_and_scale(self):
        a, b, c = np.array([3, 1]), np.array([0.5, -2]), np.array([-4, 2])
        base = joint_angle(a, b, c)
        offset = np.array([100, -50])
        self.assertAlmostEqual(joint_angle(a * 7 + offset, b * 7 + offset, c * 7 + offset), base)

    def test_image_coordinates_with_y_down(self):
        # Shoulder above elbow, wrist out to the side: a 90 degree elbow in pixel space.
        self.assertAlmostEqual(joint_angle((100, 100), (100, 200), (200, 200)), 90.0)

    def test_precision_near_180(self):
        # A 0.01 degree bend: arccos(dot) loses this, atan2 should not.
        bend = math.radians(0.01)
        c = (math.cos(bend), math.sin(bend))
        self.assertAlmostEqual(joint_angle((-1, 0), (0, 0), c), 179.99, places=6)


class JointAngle3DTest(unittest.TestCase):
    def test_axes_are_perpendicular(self):
        self.assertAlmostEqual(joint_angle((1, 0, 0), (0, 0, 0), (0, 0, 1)), 90.0)

    def test_known_3d_angle(self):
        # Angle between (1,1,0) and (1,0,0) is 45 degrees.
        self.assertAlmostEqual(joint_angle((1, 1, 0), (0, 0, 0), (1, 0, 0)), 45.0)

    def test_planar_3d_matches_2d(self):
        self.assertAlmostEqual(
            joint_angle((3, 1, 0), (0.5, -2, 0), (-4, 2, 0)),
            joint_angle((3, 1), (0.5, -2), (-4, 2)),
        )

    def test_matches_arccos_reference_on_random_points(self):
        rng = np.random.default_rng(0)
        for _ in range(500):
            a, b, c = rng.normal(size=(3, 3))
            ba, bc = a - b, c - b
            ref = math.degrees(math.acos(np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)), -1, 1)))
            self.assertAlmostEqual(joint_angle(a, b, c), ref, places=6)


class JointAngleEdgeCaseTest(unittest.TestCase):
    def test_zero_length_segment_is_nan(self):
        self.assertTrue(math.isnan(joint_angle((0, 0), (0, 0), (1, 0))))
        self.assertTrue(math.isnan(joint_angle((1, 2, 3), (4, 5, 6), (4, 5, 6))))

    def test_mismatched_dimensions_raise(self):
        with self.assertRaises(ValueError):
            joint_angle((0, 0), (1, 1, 1), (2, 2))

    def test_unsupported_dimensions_raise(self):
        with self.assertRaises(ValueError):
            joint_angle((0,), (1,), (2,))


if __name__ == "__main__":
    unittest.main()
