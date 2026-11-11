"""Tracking evaluation metrics."""

from __future__ import annotations

import numpy as np


def position_rmse(est: np.ndarray, truth: np.ndarray) -> float:
    """RMSE of position (first two state components) over time. Shapes (T, >=2)."""
    err = est[:, :2] - truth[:, :2]
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


def nees(x_est: np.ndarray, x_true: np.ndarray, P: np.ndarray) -> float:
    """Normalised estimation error squared for one time step."""
    e = x_true - x_est
    return float(e @ np.linalg.solve(P, e))
