"""Global Nearest Neighbour (GNN) data association.

Each scan, every track gets at most one measurement and every measurement goes to at most one
track. Among the assignments allowed by the chi-square gate, GNN picks the one with the lowest total
cost, solved exactly with the Hungarian algorithm (scipy's ``linear_sum_assignment``).

Cost of pairing track i with measurement j is the negative log-likelihood up to a constant:
    c_ij = d2_ij + ln|S_i|,    d2_ij = y_ij^T S_i^-1 y_ij  (squared Mahalanobis distance)
so a track with a large innovation covariance does not grab measurements just because it is uncertain.

A track can also be left unassigned (a missed detection). To allow that, the cost matrix is augmented
with one dummy column per track whose cost is ``gate + ln|S_i|``. Any gated pair is therefore
always cheaper than a miss, and an ungated pair is never chosen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from skyshield.tracking.ekf import EKF

_FORBIDDEN = 1e12


@dataclass
class Association:
    pairs: list[tuple[int, int]] = field(default_factory=list)  # (track index, meas index)
    unassigned_tracks: list[int] = field(default_factory=list)
    unassigned_meas: list[int] = field(default_factory=list)


def gate_threshold(gate_prob: float, dim: int = 2) -> float:
    """Chi-square gate on d^2 so that a true measurement falls inside with probability gate_prob."""
    return float(chi2.ppf(gate_prob, dim))


def mahalanobis_matrix(
    filters: list[EKF], measurements: list[np.ndarray], sensor_pos: np.ndarray, R: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (d2, logdetS): d2 has shape (n_tracks, n_meas), logdetS has shape (n_tracks,)."""
    n, m = len(filters), len(measurements)
    d2 = np.empty((n, m))
    logdet = np.empty(n)
    Z = np.array(measurements, dtype=float).reshape(m, 2)
    for i, f in enumerate(filters):
        z_hat = f.h(f.x, sensor_pos)
        H = f.H(f.x, sensor_pos)
        S = H @ f.P @ H.T + R
        Y = Z - z_hat
        Y[:, 1] = (Y[:, 1] + np.pi) % (2 * np.pi) - np.pi
        S_inv = np.linalg.inv(S)
        d2[i] = np.einsum("ij,jk,ik->i", Y, S_inv, Y)
        logdet[i] = np.linalg.slogdet(S)[1]
    return d2, logdet


def gnn_associate(
    filters: list[EKF],
    measurements: list[np.ndarray],
    sensor_pos: np.ndarray,
    R: np.ndarray,
    gate_prob: float = 0.99,
) -> Association:
    """Optimal one-to-one assignment of measurements to tracks within the gate."""
    n, m = len(filters), len(measurements)
    if n == 0 or m == 0:
        return Association([], list(range(n)), list(range(m)))

    gamma = gate_threshold(gate_prob)
    d2, logdet = mahalanobis_matrix(filters, measurements, sensor_pos, R)

    cost = np.full((n, m + n), _FORBIDDEN)
    real = d2 + logdet[:, None]
    cost[:, :m] = np.where(d2 <= gamma, real, _FORBIDDEN)
    cost[np.arange(n), m + np.arange(n)] = gamma + logdet  # dummy "missed" column per track

    rows, cols = linear_sum_assignment(cost)
    pairs = [(int(r), int(c)) for r, c in zip(rows, cols) if c < m and cost[r, c] < _FORBIDDEN]
    assigned_t = {r for r, _ in pairs}
    assigned_m = {c for _, c in pairs}
    return Association(
        pairs=pairs,
        unassigned_tracks=[i for i in range(n) if i not in assigned_t],
        unassigned_meas=[j for j in range(m) if j not in assigned_m],
    )


def gnn_associate_bearings(
    filters: list[EKF],
    bearings: list[float],
    sigmas: list[float],
    sensor_pos: np.ndarray,
    gate_prob: float = 0.99,
) -> Association:
    """GNN for bearing-only reports (1-D gate). Every bearing carries its own std, so S and its
    log-determinant depend on the pair; the miss cost uses each track's largest ln S over the
    bearings, which keeps every gated pair cheaper than a miss."""
    n, m = len(filters), len(bearings)
    if n == 0 or m == 0:
        return Association([], list(range(n)), list(range(m)))
    gamma = gate_threshold(gate_prob, dim=1)
    b = np.asarray(bearings, float)
    var = np.asarray(sigmas, float) ** 2
    d2 = np.empty((n, m))
    logs = np.empty((n, m))
    for i, f in enumerate(filters):
        H = f.H_bearing(f.x, sensor_pos)
        hph = float((H @ f.P @ H.T)[0, 0])
        y = (b - f.h_bearing(f.x, sensor_pos) + np.pi) % (2 * np.pi) - np.pi
        S = hph + var
        d2[i] = y**2 / S
        logs[i] = np.log(S)
    cost = np.full((n, m + n), _FORBIDDEN)
    cost[:, :m] = np.where(d2 <= gamma, d2 + logs, _FORBIDDEN)
    cost[np.arange(n), m + np.arange(n)] = gamma + logs.max(axis=1)
    rows, cols = linear_sum_assignment(cost)
    pairs = [(int(r), int(c)) for r, c in zip(rows, cols) if c < m and cost[r, c] < _FORBIDDEN]
    assigned_t = {r for r, _ in pairs}
    assigned_m = {c for _, c in pairs}
    return Association(
        pairs=pairs,
        unassigned_tracks=[i for i in range(n) if i not in assigned_t],
        unassigned_meas=[j for j in range(m) if j not in assigned_m],
    )
