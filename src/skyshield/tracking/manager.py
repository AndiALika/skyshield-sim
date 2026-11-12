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
from skyshield.tracking.gnn import gnn_associate


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
        self._in_step = True

    def update(self, measurements: list[np.ndarray], sensor_pos: np.ndarray, R: np.ndarray) -> int:
        """Fuse one sensor's report taken at the current step time. Returns tracks updated.

        Confirmed tracks get first pick, then tentative ones; leftover measurements start new
        tentative tracks immediately, so a second sensor seeing the same new target in the same
        step updates that track instead of starting a duplicate.
        """
        if not self._in_step:
            raise RuntimeError("update called outside begin_step/end_step")
        cfg = self.config
        sensor_pos = np.asarray(sensor_pos, dtype=float)
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
                used.add(free[mj])
                n_updated += 1
            free = [j for j in free if j not in used]
        for j in free:
            self._born.add(self._spawn(measurements[j], sensor_pos, R).track_id)
        return n_updated

    def end_step(self) -> list[Track]:
        """Record hits/misses for this step, confirm and delete tracks. Returns confirmed tracks."""
        if not self._in_step:
            raise RuntimeError("end_step called without begin_step")
        for t in self.tracks:
            if t.track_id not in self._born:  # newborn tracks already logged their birth hit
                self._record(t, t.track_id in self._hits)
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

    def _spawn(self, z: np.ndarray, sensor_pos: np.ndarray, R: np.ndarray) -> Track:
        cfg = self.config
        ekf = EKF.from_measurement(z, sensor_pos, R, v_sigma=cfg.v_sigma, q=cfg.q)
        t = Track(self._next_id, ekf)
        t.history.append(True)  # the birth measurement counts as the first hit
        self._next_id += 1
        if cfg.confirm_m == 1:
            t.status = TrackStatus.CONFIRMED
        self.tracks.append(t)
        return t
