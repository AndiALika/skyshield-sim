"""Tracking evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


def position_rmse(est: np.ndarray, truth: np.ndarray) -> float:
    """RMSE of position (first two state components) over time. Shapes (T, >=2)."""
    err = est[:, :2] - truth[:, :2]
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


def nees(x_est: np.ndarray, x_true: np.ndarray, P: np.ndarray) -> float:
    """Normalised estimation error squared for one time step."""
    e = x_true - x_est
    return float(e @ np.linalg.solve(P, e))


@dataclass(frozen=True)
class OSPAResult:
    """OSPA distance and its decomposition.

    ``localisation**p + cardinality**p == distance**p``.
    """

    distance: float
    localisation: float
    cardinality: float


def ospa(X: np.ndarray, Y: np.ndarray, c: float = 100.0, p: float = 2.0) -> OSPAResult:
    """Optimal Sub-Pattern Assignment metric (Schuhmacher, Vo & Vo, 2008).

    X, Y: point sets of shape (m, d) and (n, d), e.g. estimated and true positions.
    c: cut-off distance (m). It is also the penalty per missed or false object.
    p: order (p >= 1). p = 2 is the usual choice for tracking.
    """
    if c <= 0:
        raise ValueError("c must be > 0")
    if p < 1:
        raise ValueError("p must be >= 1")
    X = np.asarray(X, dtype=float).reshape(-1, 2) if np.size(X) else np.empty((0, 2))
    Y = np.asarray(Y, dtype=float).reshape(-1, 2) if np.size(Y) else np.empty((0, 2))
    m, n = len(X), len(Y)
    if m == 0 and n == 0:
        return OSPAResult(0.0, 0.0, 0.0)
    if m > n:
        X, Y, m, n = Y, X, n, m

    loc_sum = 0.0
    if m > 0:
        D = np.minimum(np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2), c) ** p
        rows, cols = linear_sum_assignment(D)
        loc_sum = float(D[rows, cols].sum())
    card_sum = c**p * (n - m)

    loc = (loc_sum / n) ** (1 / p)
    card = (card_sum / n) ** (1 / p)
    dist = ((loc_sum + card_sum) / n) ** (1 / p)
    return OSPAResult(dist, loc, card)
