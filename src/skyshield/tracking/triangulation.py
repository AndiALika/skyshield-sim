"""Position fixes from bearing-only reports (passive RF direction finding).

One DF sensor gives a line, not a point, so a bearing can update an existing track but cannot
start one. Two bearings from different sites that cross give a point. With several targets, two
sensors also produce "ghost" crossings (target A's bearing from one site crossing target B's
from the other), so by default a fix is only accepted when a bearing from a third site agrees
with it (``min_support = 1``). Accepted fixes are refined by Gauss-Newton least squares over all
the bearings that support them, and their covariance comes from the same normal equations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skyshield.tracking.gnn import gate_threshold


@dataclass(frozen=True)
class PendingBearing:
    sensor_id: int
    sensor_pos: np.ndarray
    bearing: float  # rad, from +x
    sigma: float  # rad


@dataclass
class Fix:
    xy: np.ndarray
    cov: np.ndarray  # 2 x 2
    used: list[int]  # indices into the pending list


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def intersect(p1, b1, p2, b2) -> tuple[np.ndarray, float, float] | None:
    """Crossing of the rays p1 + r1 u(b1) and p2 + r2 u(b2). None if (near) parallel."""
    u1 = np.array([np.cos(b1), np.sin(b1)])
    u2 = np.array([np.cos(b2), np.sin(b2)])
    A = np.column_stack([u1, -u2])
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    r1, r2 = np.linalg.solve(A, np.asarray(p2, float) - np.asarray(p1, float))
    return np.asarray(p1, float) + r1 * u1, float(r1), float(r2)


def _jacobian(xy, positions) -> np.ndarray:
    d = xy[None, :] - positions
    r2 = np.maximum((d**2).sum(1), 1.0)
    return np.column_stack([-d[:, 1] / r2, d[:, 0] / r2])


def refine(
    xy0, positions, bearings, sigmas, iters: int = 5, max_cond: float = 1e10
) -> tuple[np.ndarray, np.ndarray] | None:
    """Weighted least-squares position from >= 2 bearings. Returns (xy, covariance), or None
    when the geometry is degenerate (near-parallel bearings, or Gauss-Newton wandering to a
    point where the normal matrix is singular): such a fix is not usable."""
    xy = np.asarray(xy0, float).copy()
    positions = np.asarray(positions, float)
    w = 1.0 / np.asarray(sigmas, float) ** 2
    for _ in range(iters + 1):
        J = _jacobian(xy, positions)
        N = J.T @ (w[:, None] * J)
        if not np.all(np.isfinite(N)) or np.linalg.cond(N) > max_cond:
            return None
        if _ == iters:
            return xy, np.linalg.inv(N)
        pred = np.arctan2(xy[1] - positions[:, 1], xy[0] - positions[:, 0])
        res = _wrap(np.asarray(bearings) - pred)
        xy = xy + np.linalg.solve(N, J.T @ (w * res))
        if not np.all(np.isfinite(xy)):
            return None
    return None  # unreachable


def find_fixes(
    pending: list[PendingBearing],
    min_crossing_deg: float = 10.0,
    max_range: float = 4000.0,
    min_support: int = 1,
    gate_prob: float = 0.99,
) -> list[Fix]:
    """Greedy fixes: candidate crossings from every pair of sites, each checked against the
    other sites' bearings; best-supported first, every bearing used at most once."""
    gamma = gate_threshold(gate_prob, dim=1)
    sin_min = np.sin(np.deg2rad(min_crossing_deg))
    cands = []
    for i in range(len(pending)):
        for j in range(i + 1, len(pending)):
            a, b = pending[i], pending[j]
            if a.sensor_id == b.sensor_id or abs(np.sin(a.bearing - b.bearing)) < sin_min:
                continue
            hit = intersect(a.sensor_pos, a.bearing, b.sensor_pos, b.bearing)
            if hit is None:
                continue
            xy, r1, r2 = hit
            if r1 <= 0 or r2 <= 0 or max(r1, r2) > max_range:
                continue
            ref = refine(xy, [a.sensor_pos, b.sensor_pos], [a.bearing, b.bearing],
                         [a.sigma, b.sigma])
            if ref is None:
                continue
            xy, cov = ref
            # Support: the best-matching bearing from each other site, if it gates.
            support: dict[int, tuple[float, int]] = {}
            for k, c in enumerate(pending):
                if c.sensor_id in (a.sensor_id, b.sensor_id):
                    continue
                H = _jacobian(xy, c.sensor_pos[None])
                pred = np.arctan2(xy[1] - c.sensor_pos[1], xy[0] - c.sensor_pos[0])
                s2 = float((H @ cov @ H.T)[0, 0]) + c.sigma**2
                d2 = float(_wrap(c.bearing - pred)) ** 2 / s2
                if d2 <= gamma and d2 < support.get(c.sensor_id, (np.inf, -1))[0]:
                    support[c.sensor_id] = (d2, k)
            if len(support) >= min_support:
                cands.append((len(support), sum(d for d, _ in support.values()), xy,
                              [i, j, *(k for _, k in support.values())]))
    cands.sort(key=lambda c: (-c[0], c[1]))
    used: set[int] = set()
    fixes = []
    for _, _, xy, idx in cands:
        if used.intersection(idx):
            continue
        sel = [pending[k] for k in idx]
        ref = refine(xy, [p.sensor_pos for p in sel], [p.bearing for p in sel],
                     [p.sigma for p in sel])
        if ref is None:
            continue
        used.update(idx)
        fixes.append(Fix(ref[0], ref[1], idx))
    return fixes
