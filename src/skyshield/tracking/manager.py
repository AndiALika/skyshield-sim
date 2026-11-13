"""Multi-target tracker: EKF per track + GNN association + M/N initiation + miss-based deletion.

Track life cycle
----------------
1. Every measurement not used by an existing track starts a TENTATIVE track (one-point
   initialisation: position from the measurement, zero velocity with a wide prior).
2. A tentative track is CONFIRMED once it has M hits within its last N scans. It is deleted as soon
   as reaching M of N is no longer possible, so clutter-born tracks die within a few scans.
3. A confirmed track is deleted after ``max_misses`` consecutive missed scans, or if its position
   uncertainty grows past ``max_pos_std`` (it would otherwise start gating everything).

Association runs in two stages: confirmed tracks first get the pick of the measurements, then
tentative tracks compete for what is left. This stops a new tentative track stealing a measurement
from an established one.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from skyshield.tracking.ekf import EKF
from skyshield.tracking.gnn import gate_threshold, gnn_associate, gnn_associate_bearings
from skyshield.tracking.triangulation import PendingBearing, find_fixes


class TrackStatus(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"


@dataclass
class Track:
    track_id: int
    filter: EKF
    status: TrackStatus = TrackStatus.TENTATIVE
    history: deque = field(default_factory=deque)  # hit (True) / miss (False), last N scans
    consecutive_misses: int = 0
    age: int = 1  # scans since birth, including the birth scan
    source: str = "radar"  # what started it: "radar" or "df" (a bearing triangulation)

    @property
    def position(self) -> np.ndarray:
        return self.filter.x[:2]

    @property
    def confirmed(self) -> bool:
        return self.status is TrackStatus.CONFIRMED


@dataclass
class TrackerConfig:
    q: float = 1.0  # process noise intensity (m^2/s^3)
    v_sigma: float = 30.0  # initial velocity std for new tracks (m/s)
    gate_prob: float = 0.99
    # 4-of-5 rather than 3-of-4: with one-point initialisation the first gates are wide, and
    # 3/4 confirmed ~0.17 false tracks per scan at 20 clutter/scan versus 0 for 4/5, with no
    # OSPA penalty on the crossing scenario (sweep notes in CLAUDE.md, "Tracking design").
    confirm_m: int = 4
    confirm_n: int = 5
    max_misses: int = 5
    max_pos_std: float = 300.0  # m
    # Bearing-only (RF DF) initiation: a fix needs two crossing bearings plus df_min_support
    # agreeing bearings from other sites (1 = a third site must confirm; suppresses ghosts).
    df_initiate: bool = True
    df_min_support: int = 1
    df_min_crossing_deg: float = 10.0
    df_max_range: float = 4000.0  # m
    # Radar is the authority where it is heard: a bearing-only hit does not count toward
    # confirmation or keep a track alive if a radar covering the track reported this step and
    # did not detect it (negative information). Bearings still update the estimate. Without
    # this, ghost crossings and stale tracks lived on bearings that merely passed nearby.
    df_hits_in_radar_coverage: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.confirm_m <= self.confirm_n:
            raise ValueError("need 1 <= confirm_m <= confirm_n")
        if self.max_misses < 1:
            raise ValueError("max_misses must be >= 1")


class MultiTargetTracker:
    """Multi-target tracker driven either one scan at a time (``step``) or, for several sensors
    and lossy links, as ``begin_step(t)`` -> ``update(...)`` per delivered report -> ``end_step()``.

    Track management (M/N confirmation, miss counting, deletion) runs once per fusion step: a
    track is "hit" in a step if any report updated it. If no report arrives at all (every link
    jammed), every track misses that step and coasts on prediction.
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.tracks: list[Track] = []
        self._next_id = 0
        self.time: float | None = None
        self._hits: set[int] = set()  # track_ids updated in the current step
        self._born: set[int] = set()  # track_ids created in the current step
        self._pending: list[PendingBearing] = []  # unassociated DF bearings this step
        self._radar_hits: set[int] = set()  # track_ids updated by a radar this step
        self._df_hits: set[int] = set()  # track_ids updated by a bearing this step
        self._coverage: list[tuple[np.ndarray, float]] = []  # (position, range) of radars heard
        self._in_step = False

    # ----- public API -------------------------------------------------------------------------

    @property
    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.confirmed]

    def step(
        self,
        measurements: list[np.ndarray],
        sensor_pos: np.ndarray,
        R: np.ndarray,
        dt: float,
    ) -> list[Track]:
        """Single-sensor convenience: one scan, dt after the previous one."""
        self.begin_step((self.time or 0.0) + dt if self.time is not None else 0.0)
        self.update(measurements, sensor_pos, R)
        return self.end_step()

    def begin_step(self, t: float) -> None:
        """Predict every track to time t and open a fusion step."""
        if self._in_step:
            raise RuntimeError("begin_step called twice without end_step")
        if self.time is not None and t < self.time:
            raise ValueError(f"time went backwards: {t} < {self.time}")
        dt = 0.0 if self.time is None else t - self.time
        for tr in self.tracks:
            tr.filter.predict(dt)
            tr.age += 1
        self.time = t
        self._hits.clear()
        self._born.clear()
        self._pending.clear()
        self._radar_hits.clear()
        self._df_hits.clear()
        self._coverage.clear()
        self._in_step = True

    def update(
        self,
        measurements: list[np.ndarray],
        sensor_pos: np.ndarray,
        R: np.ndarray,
        coverage_m: float | None = None,
    ) -> int:
        """Fuse one sensor's report taken at the current step time. Returns tracks updated.

        Confirmed tracks get first pick, then tentative ones; leftover measurements start new
        tentative tracks immediately, so a second sensor seeing the same new target in the same
        step updates that track instead of starting a duplicate. ``coverage_m`` (the radar's
        range) enables the negative-information check on DF-born tracks.
        """
        if not self._in_step:
            raise RuntimeError("update called outside begin_step/end_step")
        cfg = self.config
        sensor_pos = np.asarray(sensor_pos, dtype=float)
        if coverage_m is not None:
            self._coverage.append((sensor_pos, coverage_m))
        free = list(range(len(measurements)))
        n_updated = 0
        for status in (TrackStatus.CONFIRMED, TrackStatus.TENTATIVE):
            idx = [i for i, t in enumerate(self.tracks) if t.status is status]
            zs = [measurements[j] for j in free]
            assoc = gnn_associate(
                [self.tracks[i].filter for i in idx], zs, sensor_pos, R, cfg.gate_prob
            )
            used = set()
            for ti, mj in assoc.pairs:
                track = self.tracks[idx[ti]]
                track.filter.update(zs[mj], sensor_pos, R)
                self._hits.add(track.track_id)
                self._radar_hits.add(track.track_id)
                used.add(free[mj])
                n_updated += 1
            free = [j for j in free if j not in used]
        for j in free:
            self._born.add(self._spawn(measurements[j], sensor_pos, R).track_id)
        return n_updated

    def update_bearings(
        self,
        bearings: list[float],
        sigmas: list[float],
        sensor_pos: np.ndarray,
        sensor_id: int,
    ) -> int:
        """Fuse one DF sensor's bearings taken at the current step time. Returns tracks updated.

        Bearings update confirmed tracks first, then tentative ones. They count as hits only
        where no covering radar reported this step (see ``df_hits_in_radar_coverage``). A bearing
        cannot start a track on its own; leftovers are kept until ``end_step``, which tries to
        triangulate them with other sites' leftovers.
        """
        if not self._in_step:
            raise RuntimeError("update_bearings called outside begin_step/end_step")
        cfg = self.config
        sensor_pos = np.asarray(sensor_pos, dtype=float)
        free = list(range(len(bearings)))
        n_updated = 0
        for status in (TrackStatus.CONFIRMED, TrackStatus.TENTATIVE):
            idx = [i for i, t in enumerate(self.tracks) if t.status is status]
            assoc = gnn_associate_bearings(
                [self.tracks[i].filter for i in idx], [bearings[j] for j in free],
                [sigmas[j] for j in free], sensor_pos, cfg.gate_prob)
            used = set()
            for ti, mj in assoc.pairs:
                track = self.tracks[idx[ti]]
                track.filter.update_bearing(bearings[free[mj]], sensor_pos, sigmas[free[mj]])
                self._df_hits.add(track.track_id)
                used.add(free[mj])
                n_updated += 1
            free = [j for j in free if j not in used]
        self._pending += [PendingBearing(sensor_id, sensor_pos, float(bearings[j]),
                                         float(sigmas[j])) for j in free]
        return n_updated

    def end_step(self) -> list[Track]:
        """Record hits/misses for this step, confirm and delete tracks. Returns confirmed tracks."""
        if not self._in_step:
            raise RuntimeError("end_step called without begin_step")
        if self.config.df_initiate and self._pending:
            self._initiate_from_bearings()
        for t in self.tracks:
            if t.track_id not in self._born:  # newborn tracks already logged their birth hit
                self._record(t, self._counts_as_hit(t))
        self.tracks = [t for t in self.tracks if not self._should_delete(t)]
        self._in_step = False
        return self.confirmed_tracks

    # ----- internals --------------------------------------------------------------------------

    def _record(self, t: Track, hit: bool) -> None:
        cfg = self.config
        t.history.append(hit)
        while len(t.history) > cfg.confirm_n:
            t.history.popleft()
        t.consecutive_misses = 0 if hit else t.consecutive_misses + 1
        if t.status is TrackStatus.TENTATIVE and sum(t.history) >= cfg.confirm_m:
            t.status = TrackStatus.CONFIRMED

    def _should_delete(self, t: Track) -> bool:
        cfg = self.config
        pos_std = np.sqrt(max(t.filter.P[0, 0], t.filter.P[1, 1]))
        if pos_std > cfg.max_pos_std:
            return True
        if t.confirmed:
            return t.consecutive_misses >= cfg.max_misses
        # Tentative: delete once M hits in the first N scans can no longer be reached.
        misses = len(t.history) - sum(t.history)
        return misses > cfg.confirm_n - cfg.confirm_m

    def _initiate_from_bearings(self) -> None:
        cfg = self.config
        gamma = gate_threshold(cfg.gate_prob, dim=2)
        for fix in find_fixes(self._pending, cfg.df_min_crossing_deg, cfg.df_max_range,
                              cfg.df_min_support, cfg.gate_prob):
            # No new track where one already is: the bearings were left over (e.g. the radar
            # track was not eligible for them), not evidence of a new target.
            if any(self._position_d2(t, fix.xy, fix.cov) <= gamma for t in self.tracks):
                continue
            # Nor where a covering radar reported this step: it would have started the track.
            if not cfg.df_hits_in_radar_coverage and self._covered(fix.xy):
                continue
            P0 = np.zeros((4, 4))
            P0[:2, :2] = fix.cov
            P0[2, 2] = P0[3, 3] = cfg.v_sigma**2
            ekf = EKF(np.array([fix.xy[0], fix.xy[1], 0.0, 0.0]), P0, q=cfg.q)
            track = self._add_track(ekf)
            track.source = "df"
            self._born.add(track.track_id)

    def _counts_as_hit(self, t: Track) -> bool:
        tid = t.track_id
        if tid in self._hits or tid in self._radar_hits:
            return True
        if tid not in self._df_hits:
            return False
        if self.config.df_hits_in_radar_coverage:
            return True
        return not self._covered(t.position)

    def _covered(self, xy: np.ndarray) -> bool:
        """True if a radar whose report was fused this step covers position xy."""
        return any(np.hypot(*(xy - p)) <= r for p, r in self._coverage)

    @staticmethod
    def _position_d2(t: Track, xy: np.ndarray, cov: np.ndarray) -> float:
        e = t.filter.x[:2] - xy
        return float(e @ np.linalg.solve(t.filter.P[:2, :2] + cov, e))

    def _spawn(self, z: np.ndarray, sensor_pos: np.ndarray, R: np.ndarray) -> Track:
        cfg = self.config
        ekf = EKF.from_measurement(z, sensor_pos, R, v_sigma=cfg.v_sigma, q=cfg.q)
        return self._add_track(ekf)

    def _add_track(self, ekf: EKF) -> Track:
        cfg = self.config
        t = Track(self._next_id, ekf)
        t.history.append(True)  # the birth measurement counts as the first hit
        self._next_id += 1
        if cfg.confirm_m == 1:
            t.status = TrackStatus.CONFIRMED
        self.tracks.append(t)
        return t
