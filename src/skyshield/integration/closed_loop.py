"""Closed-loop simulation: scenario -> radars -> jammable backhaul -> delay buffer -> tracker -> OSPA.

Timeline per fixed step k (t = k * dt):

1. Every radar scans the true targets at t (detections + clutter).
2. Each radar sends its scan as ONE report packet over its backhaul link. The packet size grows
   with the number of detections, so clutter makes reports longer and easier to lose.
   ``BackhaulLink.send`` applies the active jammers, packet errors and stop-and-wait ARQ.
3. Delivered reports enter the ``DelayLine`` with their arrival time.
4. The fusion centre runs at t + ``fusion_latency_s``: it predicts tracks to t, fuses every
   report that has arrived and was measured at t (sensor by sensor), then runs track management.
   A report that arrives after its step's fusion deadline is stale: the tracker has moved on,
   so it is counted and dropped (no out-of-sequence retrodiction).
5. OSPA between the confirmed tracks and the truth at t.

Sensors and comms draw from separate random generators, so two runs that differ only in the
link (waveform, jammer power) see exactly the same targets, detections and clutter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from skyshield.comms.jammers import Jammer
from skyshield.comms.link import BackhaulLink, BernoulliLink, DelayLine
from skyshield.comms.modulation import BCH_63_45, BlockCode, PacketSpec
from skyshield.evaluation.metrics import ospa
from skyshield.scenario.multi_target import MultiTargetScenario
from skyshield.sensors.radar import Radar
from skyshield.tracking.manager import MultiTargetTracker, TrackerConfig

HEADER_BITS = 96  # sensor id, timestamp, detection count, CRC
BITS_PER_DETECTION = 48  # range + bearing, 24 bits each


def report_packet(n_detections: int, code: BlockCode = BCH_63_45) -> PacketSpec:
    return PacketSpec(HEADER_BITS + BITS_PER_DETECTION * n_detections, code)


@dataclass
class SensorReport:
    sensor_id: int
    t_meas: float
    measurements: list[np.ndarray]


@dataclass
class SensorNode:
    sensor_id: int
    radar: Radar
    link: BackhaulLink | BernoulliLink | None = None  # None = wired / lossless


@dataclass
class JammerSchedule:
    """Jammers that are on between t_on and t_off (seconds)."""

    jammers: list[Jammer] = field(default_factory=list)
    t_on: float = 0.0
    t_off: float = np.inf

    def active(self, t: float) -> list[Jammer]:
        return self.jammers if self.t_on <= t < self.t_off else []


@dataclass
class ClosedLoopConfig:
    fusion_latency_s: float = 0.2
    ospa_c: float = 100.0
    ospa_p: float = 2.0
    code: BlockCode = BCH_63_45
    tracker: TrackerConfig = field(default_factory=TrackerConfig)


@dataclass
class Frame:
    """Snapshot of one step for replays (GIFs, dashboard). Positions are (n, 2) arrays in metres."""

    t: float
    truth_ids: list[int]
    truth_xy: np.ndarray
    det_xy: list[np.ndarray]  # per sensor: every detection in the scan, Cartesian
    det_clutter: list[np.ndarray]  # per sensor: bool, True for a false alarm
    delivered: np.ndarray  # (n_sensors,) bool: the scan reached the tracker this step
    track_ids: list[int]
    track_xy: np.ndarray
    track_cov: np.ndarray  # (n_tracks, 2, 2) position covariance
    jammer_on: bool
    ospa: float


@dataclass
class RunLog:
    """Per-step series (length n_steps) plus per-sensor delivery counts."""

    ospa: np.ndarray
    ospa_loc: np.ndarray
    ospa_card: np.ndarray
    n_true: np.ndarray
    n_est: np.ndarray
    delivered: np.ndarray  # (n_steps, n_sensors) bool: report fused this step
    sent: np.ndarray  # (n_steps, n_sensors) bool
    stale: int
    latencies_s: list[float]
    frames: list[Frame] | None = None  # only when run(..., record=True)

    @property
    def mean_ospa(self) -> float:
        return float(np.mean(self.ospa))

    @property
    def pdr(self) -> float:
        return float(self.delivered.sum() / max(self.sent.sum(), 1))

    @property
    def card_correct_frac(self) -> float:
        return float(np.mean(self.n_true == self.n_est))

    def summary(self) -> dict:
        return {
            "mean_ospa": self.mean_ospa,
            "mean_ospa_loc": float(np.mean(self.ospa_loc)),
            "mean_ospa_card": float(np.mean(self.ospa_card)),
            "pdr": self.pdr,
            "pdr_per_sensor": (self.delivered.sum(0) / np.maximum(self.sent.sum(0), 1)).tolist(),
            "card_correct_frac": self.card_correct_frac,
            "stale_reports": self.stale,
            "mean_latency_ms": 1e3 * float(np.mean(self.latencies_s)) if self.latencies_s else None,
        }


class ClosedLoopSim:
    def __init__(
        self,
        scenario: MultiTargetScenario,
        nodes: list[SensorNode],
        jamming: JammerSchedule | None = None,
        config: ClosedLoopConfig | None = None,
    ) -> None:
        self.scenario = scenario
        self.nodes = nodes
        self.jamming = jamming or JammerSchedule()
        self.config = config or ClosedLoopConfig()

    def run(
        self,
        rng_sensors: np.random.Generator,
        rng_comms: np.random.Generator,
        record: bool = False,
    ) -> RunLog:
        """Run the scenario. ``record=True`` also keeps a ``Frame`` per step (draws no extra
        random numbers, so the series are identical with or without it)."""
        sc, cfg = self.scenario, self.config
        n, m = sc.n_steps, len(self.nodes)
        tracker = MultiTargetTracker(cfg.tracker)
        line = DelayLine()
        out = {k: np.zeros(n) for k in ("ospa", "ospa_loc", "ospa_card", "n_true", "n_est")}
        delivered = np.zeros((n, m), dtype=bool)
        sent = np.zeros((n, m), dtype=bool)
        stale, latencies = 0, []
        col = {node.sensor_id: i for i, node in enumerate(self.nodes)}
        frames: list[Frame] | None = [] if record else None

        for k in range(n):
            t = k * sc.dt
            truth_ids, states = sc.truth_at(k)
            jammers = self.jamming.active(t)
            scans = []

            # Sensors scan and transmit.
            for node in self.nodes:
                zs, origins = node.radar.measure_multi(states[:, :2], rng_sensors)
                if record:
                    xy = np.array([node.radar.to_cartesian(z) for z in zs]).reshape(-1, 2)
                    scans.append((xy, np.asarray(origins) < 0))
                report = SensorReport(node.sensor_id, t, zs)
                sent[k, col[node.sensor_id]] = True
                if node.link is None:
                    line.push(report, t)
                    continue
                d = node.link.send(t, jammers, rng_comms, packet=report_packet(len(zs), cfg.code))
                if d.delivered:
                    latencies.append(d.delay_s)
                    line.push(report, t + d.delay_s)

            # Fusion centre.
            tracker.begin_step(t)
            for rep in sorted(line.pop_ready(t + cfg.fusion_latency_s),
                              key=lambda r: (r.t_meas, r.sensor_id)):
                if rep.t_meas < t:
                    stale += 1
                    continue
                radar = self.nodes[col[rep.sensor_id]].radar
                tracker.update(rep.measurements, radar.position, radar.R)
                delivered[k, col[rep.sensor_id]] = True
            confirmed = tracker.end_step()

            est = np.array([tr.position for tr in confirmed]).reshape(-1, 2)
            res = ospa(est, states[:, :2], c=cfg.ospa_c, p=cfg.ospa_p)
            out["ospa"][k], out["ospa_loc"][k], out["ospa_card"][k] = (
                res.distance, res.localisation, res.cardinality)
            out["n_true"][k], out["n_est"][k] = len(states), len(confirmed)
            if frames is not None:
                frames.append(Frame(
                    t=t, truth_ids=list(truth_ids), truth_xy=states[:, :2].copy(),
                    det_xy=[a for a, _ in scans], det_clutter=[b for _, b in scans],
                    delivered=delivered[k].copy(),
                    track_ids=[tr.track_id for tr in confirmed], track_xy=est.copy(),
                    track_cov=np.array([tr.filter.P[:2, :2] for tr in confirmed]).reshape(-1, 2, 2),
                    jammer_on=bool(jammers), ospa=float(res.distance)))

        return RunLog(out["ospa"], out["ospa_loc"], out["ospa_card"], out["n_true"].astype(int),
                      out["n_est"].astype(int), delivered, sent, stale, latencies, frames)
