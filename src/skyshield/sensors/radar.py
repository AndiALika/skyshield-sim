"""2D range-bearing radar measurement model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Radar:
    position: np.ndarray
    sigma_r: float = 5.0  # range noise std (m)
    sigma_b: float = np.deg2rad(1.0)  # bearing noise std (rad)
    pd: float = 0.9  # probability of detection
    max_range: float = 3000.0  # m
    clutter_rate: float = 0.0  # expected false alarms per scan (Poisson)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)

    @property
    def R(self) -> np.ndarray:
        """Measurement noise covariance for [range, bearing]."""
        return np.diag([self.sigma_r**2, self.sigma_b**2])

    def measure(self, target_xy: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
        """Return a list of [range, bearing] measurements for one scan.

        May contain zero or one target detection plus Poisson clutter.
        """
        out: list[np.ndarray] = []
        d = np.asarray(target_xy, dtype=float) - self.position
        r = float(np.hypot(*d))
        if r <= self.max_range and rng.random() < self.pd:
            b = np.arctan2(d[1], d[0])
            out.append(np.array([r + self.sigma_r * rng.standard_normal(),
                                 b + self.sigma_b * rng.standard_normal()]))
        for _ in range(rng.poisson(self.clutter_rate)):
            cr = self.max_range * np.sqrt(rng.random())
            cb = rng.uniform(-np.pi, np.pi)
            out.append(np.array([cr, cb]))
        return out
