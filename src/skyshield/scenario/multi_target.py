"""Multi-target ground truth: targets that appear and disappear over a fixed-step timeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from skyshield.scenario.trajectories import loiter_trajectory, waypoint_trajectory


@dataclass
class TargetTruth:
    """One target's ground truth.

    ``states[i]`` is the state at global step ``start_step + i``.
    """

    target_id: int
    start_step: int
    states: np.ndarray  # (T_i, 4)

    @property
    def end_step(self) -> int:
        """First step at which the target no longer exists (exclusive)."""
        return self.start_step + len(self.states)

    def state_at(self, k: int) -> np.ndarray | None:
        if self.start_step <= k < self.end_step:
            return self.states[k - self.start_step]
        return None


@dataclass
class MultiTargetScenario:
    dt: float
    targets: list[TargetTruth] = field(default_factory=list)
    n_steps: int | None = None  # default: until the last target disappears

    def __post_init__(self) -> None:
        if self.n_steps is None:
            self.n_steps = max((t.end_step for t in self.targets), default=0)

    def truth_at(self, k: int) -> tuple[list[int], np.ndarray]:
        """IDs and states (N, 4) of every target alive at step k."""
        ids, states = [], []
        for t in self.targets:
            s = t.state_at(k)
            if s is not None:
                ids.append(t.target_id)
                states.append(s)
        return ids, (np.array(states) if states else np.empty((0, 4)))

    def cardinality(self) -> np.ndarray:
        """Number of live targets at each step, shape (n_steps,)."""
        return np.array([len(self.truth_at(k)[0]) for k in range(self.n_steps)])


def crossing_scenario(
    dt: float = 1.0, rng: np.random.Generator | None = None, accel_noise: float = 0.5
) -> MultiTargetScenario:
    """Four drones with staggered births and deaths.

    - T0 and T1 cross near the protected site (stress test for association).
    - T2 arrives late and leaves early (tests initiation and deletion mid-run).
    - T3 loiters to the north for the whole run.
    """
    rng = rng or np.random.default_rng()
    t0 = waypoint_trajectory(
        np.array([[-2200.0, -400.0], [0.0, 100.0], [2200.0, 500.0]]), 15.0, dt, accel_noise, rng
    )
    t1 = waypoint_trajectory(
        np.array([[-1800.0, 1400.0], [0.0, 100.0], [1600.0, -1500.0]]), 13.0, dt, accel_noise, rng
    )
    t2 = waypoint_trajectory(
        np.array([[1800.0, 1800.0], [600.0, 600.0], [1500.0, -400.0]]), 18.0, dt, accel_noise, rng
    )
    t3 = loiter_trajectory(np.array([-300.0, 1600.0]), radius=350.0, speed=12.0, dt=dt, duration=300.0)
    targets = [
        TargetTruth(0, 0, t0),
        TargetTruth(1, 10, t1),
        TargetTruth(2, 60, t2[: int(140 / dt)]),
        TargetTruth(3, 0, t3),
    ]
    return MultiTargetScenario(dt=dt, targets=targets, n_steps=int(300 / dt))
