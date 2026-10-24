"""Temporal filters for noisy per-frame signals such as joint angles. Pure Python, no dependencies."""

import math


def lowpass_alpha(cutoff_hz, dt):
    """Smoothing factor for a first-order low-pass filter. Works on floats or NumPy arrays."""
    tau = 1.0 / (2 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """One Euro filter (Casiez et al., CHI 2012): an adaptive low-pass filter.

    When the signal is still, it smooths heavily (cutoff near `min_cutoff`) to remove jitter.
    When the signal moves fast, the cutoff rises with speed (scaled by `beta`) to keep lag low.
    """

    def __init__(self, fps: float, min_cutoff: float = 1.0, beta: float = 0.1, d_cutoff: float = 1.0):
        self.dt = 1.0 / fps
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x = None   # last filtered value
        self._dx = 0.0   # last filtered derivative
        self._raw = None  # last raw value

    def update(self, x: float) -> float:
        if self._x is None:
            self._x, self._raw = x, x
            return x
        dx = (x - self._raw) / self.dt
        self._raw = x
        self._dx += lowpass_alpha(self.d_cutoff, self.dt) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        self._x += lowpass_alpha(cutoff, self.dt) * (x - self._x)
        return self._x


class AngleFilter:
    """Cleans up a stream of joint angles: rejects glitches, then smooths with a One Euro filter.

    - Jump rejection: a sample that differs from the last accepted sample by more than
      `max_speed_dps / fps` degrees is treated as a landmark glitch and dropped; the previous
      output is held. If the jump persists for more than `max_rejects` consecutive frames,
      it is accepted as real motion and the smoother restarts at the new value.
    - Missing samples (None or NaN, e.g. a joint not visible) hold the last output for up
      to `max_hold` frames, after which the output becomes None and the filter resets.
    """

    def __init__(
        self,
        fps: float,
        max_speed_dps: float = 600.0,
        max_rejects: int = 3,
        max_hold: int | None = None,
        min_cutoff: float = 1.0,
        beta: float = 0.1,
        smooth: bool = True,
    ):
        """Set `smooth=False` to keep only jump rejection and hold, e.g. when the landmarks
        feeding the angle are already smoothed (smoothing twice doubles the lag)."""
        # Defaults tuned on the synthetic clip (vs ground truth) and samples/squat_front.mp4:
        # beta=0.1 roughly halves frame-to-frame jitter while adding <1 degree of mean error.
        self.max_step = max_speed_dps / fps
        self.max_rejects = max_rejects
        self.max_hold = max_hold if max_hold is not None else round(fps / 2)
        self._smoother = OneEuroFilter(fps, min_cutoff=min_cutoff, beta=beta) if smooth else None
        self.reset()

    def reset(self) -> None:
        if self._smoother is not None:
            self._smoother.reset()
        self._last_accepted = None
        self._value = None
        self._rejects = 0
        self._missing = 0
        self.rejected = False  # True if the most recent sample was dropped as a glitch

    @property
    def value(self) -> float | None:
        return self._value

    def update(self, angle: float | None) -> float | None:
        """Feeds one raw sample and returns the filtered angle (or None if there is none yet)."""
        self.rejected = False
        if angle is None or math.isnan(angle):
            self._missing += 1
            if self._missing > self.max_hold:
                self.reset()
            return self._value
        self._missing = 0

        if self._last_accepted is not None and abs(angle - self._last_accepted) > self.max_step:
            self._rejects += 1
            if self._rejects <= self.max_rejects:
                self.rejected = True
                return self._value
            # The jump persisted, so it is real motion rather than a one-off glitch.
            if self._smoother is not None:
                self._smoother.reset()

        self._rejects = 0
        self._last_accepted = angle
        self._value = self._smoother.update(angle) if self._smoother is not None else angle
        return self._value
