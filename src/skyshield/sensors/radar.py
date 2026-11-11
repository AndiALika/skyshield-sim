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
        """Return a list of [range, bearing] measurements for one scan of a single target.

        May contain zero or one target detection plus Poisson clutter.
        """
        zs, _ = self.measure_multi(np.atleast_2d(target_xy), rng)
        return zs

    def measure_multi(
        self, targets_xy: np.ndarray, rng: np.random.Generator
    ) -> tuple[list[np.ndarray], list[int]]:
        """One scan over several targets.

        targets_xy: (N, 2) target positions (N may be 0).
        Returns (measurements, origins): origins[i] is the index of the target that produced
        measurement i, or -1 for clutter. Order is target detections first, then clutter.
        """
        out: list[np.ndarray] = []
        origins: list[int] = []
        targets_xy = np.asarray(targets_xy, dtype=float).reshape(-1, 2)
        for i, xy in enumerate(targets_xy):
            d = xy - self.position
            r = float(np.hypot(*d))
            if r <= self.max_range and rng.random() < self.pd:
                b = np.arctan2(d[1], d[0])
                out.append(np.array([r + self.sigma_r * rng.standard_normal(),
                                     b + self.sigma_b * rng.standard_normal()]))
                origins.append(i)
        for _ in range(rng.poisson(self.clutter_rate)):
            # Uniform over the disc of radius max_range.
            cr = self.max_range * np.sqrt(rng.random())
            cb = rng.uniform(-np.pi, np.pi)
            out.append(np.array([cr, cb]))
            origins.append(-1)
        return out, origins

    @property
    def clutter_density(self) -> float:
        """Spatial clutter density in Cartesian space (false alarms per m^2 per scan)."""
        return self.clutter_rate / (np.pi * self.max_range**2)

    def to_cartesian(self, z: np.ndarray) -> np.ndarray:
        """Convert a [range, bearing] measurement to an (x, y) position."""
        return self.position + z[0] * np.array([np.cos(z[1]), np.sin(z[1])])
