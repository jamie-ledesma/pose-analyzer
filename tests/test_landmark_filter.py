"""Unit tests for pose.landmark_filter. Run with: python -m unittest discover tests"""

import unittest

import numpy as np

from pose.estimator import PoseResult
from pose.landmark_filter import LandmarkSmoother

FPS = 30


def make_pose(image_xy, world=None) -> PoseResult:
    image_xy = np.asarray(image_xy, dtype=np.float64)
    image = np.column_stack([image_xy, np.zeros(len(image_xy))])
    world = image.copy() if world is None else np.asarray(world, dtype=np.float64)
    return PoseResult(image=image, world=world, visibility=np.ones(len(image_xy)))


def still_pose(n=33, x=0.5, y=0.5):
    return np.tile([x, y], (n, 1))


class LandmarkSmootherTest(unittest.TestCase):
    def test_still_pose_is_unchanged(self):
        s = LandmarkSmoother(FPS, aspect=1.0)
        for _ in range(20):
            out = s.update(make_pose(still_pose()))
        np.testing.assert_allclose(out.image[:, :2], still_pose())

    def test_reduces_jitter(self):
        rng = np.random.default_rng(0)
        s = LandmarkSmoother(FPS, aspect=1.0)
        raw, smoothed = [], []
        for _ in range(200):
            p = still_pose() + rng.normal(0, 0.003, (33, 2))  # ~2 px jitter on a 720p frame
            raw.append(p)
            smoothed.append(s.update(make_pose(p)).image[:, :2])
        raw, smoothed = np.array(raw[20:]), np.array(smoothed[20:])
        self.assertLess(smoothed.std(axis=0).mean(), raw.std(axis=0).mean() * 0.6)

    def test_follows_steady_motion(self):
        # 0.3 frame heights per second, a brisk body movement.
        s = LandmarkSmoother(FPS, aspect=1.0)
        for i in range(60):
            target = still_pose(x=0.2 + 0.01 * i)
            out = s.update(make_pose(target))
        self.assertLess(np.abs(out.image[:, 0] - target[:, 0]).max(), 0.02)

    def test_single_landmark_glitch_is_held(self):
        s = LandmarkSmoother(FPS, aspect=1.0, max_rejects=3)
        for _ in range(10):
            s.update(make_pose(still_pose()))
        glitch = still_pose()
        glitch[14] = [0.9, 0.1]  # one landmark teleports across the frame
        out = s.update(make_pose(glitch))
        self.assertTrue(s.rejected[14])
        self.assertEqual(s.rejected.sum(), 1)
        np.testing.assert_allclose(out.image[14, :2], [0.5, 0.5])

    def test_persistent_jump_snaps_after_max_rejects(self):
        s = LandmarkSmoother(FPS, aspect=1.0, max_rejects=3)
        for _ in range(10):
            s.update(make_pose(still_pose()))
        moved = still_pose()
        moved[14] = [0.9, 0.1]
        outs = [s.update(make_pose(moved)).image[14, :2].copy() for _ in range(4)]
        for o in outs[:3]:
            np.testing.assert_allclose(o, [0.5, 0.5])
        np.testing.assert_allclose(outs[3], [0.9, 0.1])

    def test_world_coordinates_are_smoothed_with_image(self):
        s = LandmarkSmoother(FPS, aspect=1.0)
        world = np.zeros((33, 3))
        s.update(make_pose(still_pose(), world))
        moved = still_pose(x=0.51)
        out = s.update(make_pose(moved, world + 1.0))
        frac_img = (out.image[0, 0] - 0.5) / 0.01
        frac_world = out.world[0, 0] / 1.0
        self.assertAlmostEqual(frac_img, frac_world)
        self.assertTrue(0 < frac_world < 1)

    def test_missing_pose_holds_then_drops(self):
        s = LandmarkSmoother(FPS, aspect=1.0, max_hold=2)
        s.update(make_pose(still_pose()))
        self.assertIsNotNone(s.update(None))
        self.assertIsNotNone(s.update(None))
        self.assertIsNone(s.update(None))

    def test_input_pose_is_not_mutated(self):
        s = LandmarkSmoother(FPS, aspect=1.0)
        first = make_pose(still_pose())
        s.update(first)
        s.update(make_pose(still_pose(x=0.52)))
        np.testing.assert_allclose(first.image[:, 0], 0.5)


if __name__ == "__main__":
    unittest.main()
