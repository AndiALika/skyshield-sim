"""Extended Kalman Filter with a constant-velocity model and range-bearing measurements."""

from __future__ import annotations

import numpy as np


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


class EKF:
    """State: [x, y, vx, vy]. Measurement: [range, bearing] from a known sensor position."""

    def __init__(self, x0: np.ndarray, P0: np.ndarray, q: float = 1.0) -> None:
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.asarray(P0, dtype=float).copy()
        self.q = q  # white-noise acceleration intensity (m^2/s^3)

    @staticmethod
    def F(dt: float) -> np.ndarray:
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        return F

    def Q(self, dt: float) -> np.ndarray:
        q = self.q
        g = np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]) * q
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = g
        Q[np.ix_([1, 3], [1, 3])] = g
        return Q

    def predict(self, dt: float) -> None:
        F = self.F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q(dt)

    @staticmethod
    def h(x: np.ndarray, sensor_pos: np.ndarray) -> np.ndarray:
        dx, dy = x[0] - sensor_pos[0], x[1] - sensor_pos[1]
        return np.array([np.hypot(dx, dy), np.arctan2(dy, dx)])

    @staticmethod
    def H(x: np.ndarray, sensor_pos: np.ndarray) -> np.ndarray:
        dx, dy = x[0] - sensor_pos[0], x[1] - sensor_pos[1]
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        return np.array([[dx / r, dy / r, 0.0, 0.0], [-dy / r2, dx / r2, 0.0, 0.0]])

    def innovation(self, z: np.ndarray, sensor_pos: np.ndarray, R: np.ndarray):
        """Return (residual, S) for a measurement, used for gating and the update."""
        y = np.asarray(z, dtype=float) - self.h(self.x, sensor_pos)
        y[1] = wrap_angle(y[1])
        H = self.H(self.x, sensor_pos)
        S = H @ self.P @ H.T + R
        return y, S, H

    def update(self, z: np.ndarray, sensor_pos: np.ndarray, R: np.ndarray) -> None:
        y, S, H = self.innovation(z, sensor_pos, R)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(4) - K @ H
        # Joseph form for numerical stability.
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    @classmethod
    def from_measurement(
        cls, z: np.ndarray, sensor_pos: np.ndarray, R: np.ndarray, v_sigma: float = 30.0, q: float = 1.0
    ) -> "EKF":
        """One-point initialisation from a range-bearing measurement."""
        r, b = z
        x0 = np.array([sensor_pos[0] + r * np.cos(b), sensor_pos[1] + r * np.sin(b), 0.0, 0.0])
        # Convert polar covariance to Cartesian position covariance.
        J = np.array([[np.cos(b), -r * np.sin(b)], [np.sin(b), r * np.cos(b)]])
        Ppos = J @ R @ J.T
        P0 = np.zeros((4, 4))
        P0[:2, :2] = Ppos
        P0[2, 2] = P0[3, 3] = v_sigma**2
        return cls(x0, P0, q=q)
