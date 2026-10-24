"""Squat rep counting and form checks. Pure logic on per-frame measurements; no drawing or I/O."""

from dataclasses import dataclass, field

TORSO_LEAN = "torso_lean"
KNEES_CAVING = "knees_caving"

FAULT_MESSAGES = {
    TORSO_LEAN: "Chest up",
    KNEES_CAVING: "Knees out",
}


@dataclass(frozen=True)
class SquatThresholds:
    start_descent: float = 150.0      # knee angle below this starts a rep
    stand: float = 155.0              # knee angle above this ends a rep
    depth: float = 100.0              # knee angle at or below this is full depth (thighs ~parallel)
    partial: float = 140.0            # attempts that bottom out between `depth` and this are "too shallow"
    max_torso_lean: float = 45.0      # degrees from vertical before "leaning too far forward"
    min_knee_width_ratio: float = 0.9  # knee gap / ankle gap below this while bent = knees caving in
    knee_check_below: float = 130.0   # only check knee tracking once the knees bend past this angle
    min_facing: float = 0.6           # knee tracking needs a roughly front-on view to be measurable
    fault_frames: int = 3             # consecutive frames a fault must persist to show, or be absent to clear


@dataclass(frozen=True)
class SquatMetrics:
    """One frame's measurements. Any field may be None when it can't be measured."""

    knee_left: float | None = None
    knee_right: float | None = None
    torso_lean: float | None = None
    knee_width_ratio: float | None = None
    facing: float | None = None

    @property
    def knee(self) -> float | None:
        values = [k for k in (self.knee_left, self.knee_right) if k is not None]
        return sum(values) / len(values) if values else None


@dataclass(frozen=True)
class RepResult:
    counted: bool
    number: int | None      # rep number if counted
    bottom_angle: float     # smallest knee angle reached
    faults: frozenset[str]  # faults seen at any point during the rep

    @property
    def message(self) -> str:
        faults = ", ".join(FAULT_MESSAGES[f].lower() for f in sorted(self.faults))
        if not self.counted:
            # Plain ASCII: OpenCV's Hershey fonts cannot render a degree sign.
            return f"Not counted: go deeper (knee {self.bottom_angle:.0f} deg)" + (f"; {faults}" if faults else "")
        return f"Rep {self.number}: " + (faults or "good")


@dataclass(frozen=True)
class SquatState:
    reps: int
    phase: str                      # "standing", "descending", "bottom" or "ascending"
    faults: frozenset[str]          # faults active right now
    last_rep: RepResult | None
    history: tuple[RepResult, ...] = field(default=())


class SquatTracker:
    """Counts squat reps and flags form faults from a stream of SquatMetrics.

    A rep starts when the knee angle drops below `start_descent` and ends when it rises above
    `stand`. It counts if the knee reached `depth` in between. Form checks only run during a rep.
    """

    def __init__(self, thresholds: SquatThresholds = SquatThresholds()):
        self.t = thresholds
        self._in_rep = False
        self._min_knee = 180.0
        self._rep_faults: set[str] = set()
        self._streaks = {f: [0, 0] for f in FAULT_MESSAGES}  # fault -> [frames present, frames absent]
        self._active: set[str] = set()
        self._history: list[RepResult] = []
        self._phase = "standing"

    def _raw_faults(self, m: SquatMetrics, knee: float) -> set[str]:
        faults = set()
        if m.torso_lean is not None and m.torso_lean > self.t.max_torso_lean:
            faults.add(TORSO_LEAN)
        if (
            knee < self.t.knee_check_below
            and m.facing is not None and m.facing >= self.t.min_facing
            and m.knee_width_ratio is not None and m.knee_width_ratio < self.t.min_knee_width_ratio
        ):
            faults.add(KNEES_CAVING)
        return faults

    def _debounce(self, raw: set[str]) -> None:
        for fault, streak in self._streaks.items():
            if fault in raw:
                streak[0], streak[1] = streak[0] + 1, 0
            else:
                streak[0], streak[1] = 0, streak[1] + 1
            if streak[0] >= self.t.fault_frames:
                self._active.add(fault)
            elif streak[1] >= self.t.fault_frames:
                self._active.discard(fault)

    def update(self, m: SquatMetrics) -> SquatState:
        knee = m.knee
        if knee is not None:
            if not self._in_rep and knee < self.t.start_descent:
                self._in_rep = True
                self._min_knee = knee
                self._rep_faults = set()

            self._debounce(self._raw_faults(m, knee) if self._in_rep else set())

            if self._in_rep:
                self._min_knee = min(self._min_knee, knee)
                self._rep_faults |= self._active
                if knee > self.t.stand:
                    self._finish_rep()
                elif knee <= self.t.depth:
                    self._phase = "bottom"
                elif knee > self._min_knee + 5:
                    self._phase = "ascending"
                else:
                    self._phase = "descending"
            if not self._in_rep:
                self._phase = "standing"

        return SquatState(
            reps=sum(r.counted for r in self._history),
            phase=self._phase,
            faults=frozenset(self._active),
            last_rep=self._history[-1] if self._history else None,
            history=tuple(self._history),
        )

    def _finish_rep(self) -> None:
        self._in_rep = False
        self._active.clear()
        for streak in self._streaks.values():
            streak[0] = streak[1] = 0
        if self._min_knee <= self.t.depth:
            number = sum(r.counted for r in self._history) + 1
            self._history.append(RepResult(True, number, self._min_knee, frozenset(self._rep_faults)))
        elif self._min_knee <= self.t.partial:
            self._history.append(RepResult(False, None, self._min_knee, frozenset(self._rep_faults)))
        # Shallower dips than `partial` are ignored as noise or shifting weight.
