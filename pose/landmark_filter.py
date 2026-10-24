"""Temporal smoothing for whole-body pose landmarks."""

import numpy as np

from .estimator import PoseResult
from .filters import lowpass_alpha


class LandmarkSmoother:
    """Smooths a stream of PoseResults: per-landmark jump rejection, then One Euro smoothing.

    Each landmark gets its own adaptive cutoff from how fast it moves in the image (measured in
    frame heights per second, so the settings mean the same thing at any resolution). The same
    smoothing factor is applied to that landmark's image and 3D world coordinates, so the drawn
    skeleton and the 3D angles stay consistent with each other.

    - Jump rejection: a landmark that moves faster than `max_speed` frame heights per second is
      treated as a glitch and held in place. If the jump persists for more than `max_rejects`
      consecutive frames, it is accepted as real and that landmark snaps to its new position.
    - Missing poses (None) hold the last output for up to `max_hold` frames, then return None.
    """

    def __init__(
        self,
        fps: float,
        aspect: float,
        min_cutoff: float = 2.0,
        beta: float = 20.0,
        d_cutoff: float = 1.0,
        max_speed: float = 3.0,
        max_rejects: int = 3,
        max_hold: int | None = None,
    ):
        """`aspect` is frame width / height, used to put x and y in the same units.

        Defaults tuned on the synthetic clip (vs ground truth) and samples/squat_front.mp4:
        they halve skeleton jitter and match the 3D-angle steadiness of the old angle-only
        smoothing, at ~1 degree of extra mean angle error from lag.
        """
        self.dt = 1.0 / fps
        self.scale = np.array([aspect, 1.0])
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.max_step = max_speed * self.dt
        self.max_rejects = max_rejects
        self.max_hold = max_hold if max_hold is not None else round(fps / 2)
        self.reset()

    def reset(self) -> None:
        self._out = None      # last smoothed PoseResult
        self._anchor = None   # last accepted raw image position per landmark, frame-height units
        self._speed = None    # filtered speed per landmark, frame heights per second
        self._rejects = None  # consecutive rejected frames per landmark
        self._missing = 0
        self.rejected = np.zeros(0, dtype=bool)  # landmarks held as glitches on the last update

    def update(self, pose: PoseResult | None) -> PoseResult | None:
        if pose is None:
            self._missing += 1
            if self._missing > self.max_hold:
                self.reset()
            return self._out
        self._missing = 0

        pts = pose.image[:, :2] * self.scale
        if self._out is None:
            self._out = PoseResult(pose.image.copy(), pose.world.copy(), pose.visibility.copy())
            self._anchor = pts
            self._speed = np.zeros(len(pts))
            self._rejects = np.zeros(len(pts), dtype=int)
            self.rejected = np.zeros(len(pts), dtype=bool)
            return self._out

        step = np.linalg.norm(pts - self._anchor, axis=1)
        jump = step > self.max_step
        self._rejects = np.where(jump, self._rejects + 1, 0)
        hold = jump & (self._rejects <= self.max_rejects)
        snap = jump & ~hold  # the jump persisted, so it is real motion

        speed = np.where(hold, self._speed, step / self.dt)
        self._speed = self._speed + lowpass_alpha(self.d_cutoff, self.dt) * (speed - self._speed)
        alpha = lowpass_alpha(self.min_cutoff + self.beta * self._speed, self.dt)
        alpha = np.where(hold, 0.0, np.where(snap, 1.0, alpha))[:, None]

        prev = self._out
        self._out = PoseResult(
            image=prev.image + alpha * (pose.image - prev.image),
            world=prev.world + alpha * (pose.world - prev.world),
            visibility=pose.visibility.copy(),
        )
        self._anchor = np.where(hold[:, None], self._anchor, pts)
        self._speed[snap] = 0.0
        self._rejects[snap] = 0
        self.rejected = hold
        return self._out
