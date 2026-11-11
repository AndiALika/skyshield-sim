import numpy as np
import pytest

from skyshield.evaluation.metrics import ospa
from skyshield.scenario.multi_target import crossing_scenario
from skyshield.sensors.radar import Radar
from skyshield.tracking.manager import MultiTargetTracker, TrackerConfig

DT = 1.0


def straight(start, vel, n):
    start, vel = np.asarray(start, float), np.asarray(vel, float)
    k = np.arange(n)[:, None]
    return np.hstack([start + k * vel * DT, np.tile(vel, (n, 1))])


def run(tracker, radar, truths, n_steps, rng):
    """truths: list of (start_step, states). Returns confirmed-track count per step."""
    counts = []
    for k in range(n_steps):
        xy = [s[k - k0, :2] for k0, s in truths if 0 <= k - k0 < len(s)]
        zs, _ = radar.measure_multi(np.array(xy).reshape(-1, 2), rng)
        counts.append(len(tracker.step(zs, radar.position, radar.R, DT)))
    return counts


def test_config_validation():
    with pytest.raises(ValueError):
        TrackerConfig(confirm_m=4, confirm_n=3)
    with pytest.raises(ValueError):
        TrackerConfig(max_misses=0)


def test_confirms_after_m_hits_not_before():
    rng = np.random.default_rng(0)
    radar = Radar(position=np.zeros(2), pd=1.0)
    tracker = MultiTargetTracker(TrackerConfig(confirm_m=3, confirm_n=4))
    counts = run(tracker, radar, [(0, straight([1000, 0], [10, 0], 10))], 10, rng)
    assert counts[:2] == [0, 0]
    assert counts[2] == 1  # third consecutive hit
    assert all(c == 1 for c in counts[2:])


def test_two_separated_targets_tracked_accurately():
    rng = np.random.default_rng(1)
    radar = Radar(position=np.zeros(2), pd=1.0)
    a = straight([1000, 500], [-10, 0], 60)
    b = straight([-800, -900], [0, 12], 60)
    tracker = MultiTargetTracker(TrackerConfig(q=0.1))
    run(tracker, radar, [(0, a), (0, b)], 60, rng)
    conf = tracker.confirmed_tracks
    assert len(conf) == 2
    est = np.array(sorted([t.position for t in conf], key=lambda p: p[0]))
    true = np.array(sorted([a[-1, :2], b[-1, :2]], key=lambda p: p[0]))
    assert np.all(np.linalg.norm(est - true, axis=1) < 25.0)
    # Velocity should be learned too.
    vels = sorted([t.filter.x[2:] for t in conf], key=lambda v: v[0])
    assert np.allclose(vels[0], [-10, 0], atol=3.0)
    assert np.allclose(vels[1], [0, 12], atol=3.0)


def test_track_ids_stable_while_target_persists():
    rng = np.random.default_rng(2)
    radar = Radar(position=np.zeros(2), pd=1.0)
    tracker = MultiTargetTracker()
    truth = straight([1500, 0], [-12, 3], 80)
    ids = set()
    for k in range(80):
        zs, _ = radar.measure_multi(truth[k : k + 1, :2], rng)
        for t in tracker.step(zs, radar.position, radar.R, DT):
            ids.add(t.track_id)
    assert len(ids) == 1


def test_confirmed_track_deleted_after_target_leaves():
    rng = np.random.default_rng(3)
    radar = Radar(position=np.zeros(2), pd=1.0)
    cfg = TrackerConfig(max_misses=4)
    tracker = MultiTargetTracker(cfg)
    counts = run(tracker, radar, [(0, straight([800, 0], [5, 5], 20))], 30, rng)
    assert counts[19] == 1
    # Misses start at step 20; deleted on the 4th consecutive miss (step 23).
    assert counts[20:23] == [1, 1, 1]
    assert counts[23] == 0
    assert tracker.tracks == []


def test_survives_missed_detections():
    rng = np.random.default_rng(4)
    radar = Radar(position=np.zeros(2), pd=0.8)
    tracker = MultiTargetTracker(TrackerConfig(max_misses=5))
    counts = run(tracker, radar, [(0, straight([1200, 300], [-8, -4], 120))], 120, rng)
    assert sum(c == 1 for c in counts[10:]) >= 0.95 * len(counts[10:])


def test_clutter_alone_rarely_confirms():
    rng = np.random.default_rng(5)
    radar = Radar(position=np.zeros(2), pd=1.0, clutter_rate=20.0)
    tracker = MultiTargetTracker()
    counts = run(tracker, radar, [], 200, rng)
    # 4000 clutter points over the run; M/N logic should almost never confirm one.
    assert np.mean(counts) < 0.05
    # Tentative clutter tracks are pruned, so the track list stays bounded.
    assert len(tracker.tracks) < 100


def test_crossing_scenario_with_clutter_ospa():
    """End-to-end: 4 drones (2 crossing), Pd 0.9, 10 clutter/scan."""
    rng = np.random.default_rng(42)
    sc = crossing_scenario(dt=DT, rng=rng)
    radar = Radar(position=np.zeros(2), pd=0.9, clutter_rate=10.0)
    tracker = MultiTargetTracker()
    ospas, card_err = [], []
    for k in range(sc.n_steps):
        _, states = sc.truth_at(k)
        zs, _ = radar.measure_multi(states[:, :2], rng)
        conf = tracker.step(zs, radar.position, radar.R, sc.dt)
        est = np.array([t.position for t in conf]).reshape(-1, 2)
        ospas.append(ospa(est, states[:, :2], c=100.0).distance)
        card_err.append(abs(len(conf) - len(states)))
    ospas = np.array(ospas)
    assert np.mean(ospas) < 30.0, np.mean(ospas)
    assert np.mean(np.array(card_err) == 0) > 0.85


def test_tentative_track_pruned_once_m_of_n_unreachable():
    radar = Radar(position=np.zeros(2))
    tracker = MultiTargetTracker(TrackerConfig(confirm_m=4, confirm_n=5))
    tracker.step([np.array([1000.0, 0.3])], radar.position, radar.R, DT)  # birth (hit 1)
    assert len(tracker.tracks) == 1 and not tracker.tracks[0].confirmed
    tracker.step([], radar.position, radar.R, DT)  # 1 miss: 4/5 still reachable
    assert len(tracker.tracks) == 1
    tracker.step([], radar.position, radar.R, DT)  # 2 misses: at most 3/5 -> pruned
    assert tracker.tracks == []


def test_confirmed_tracks_get_measurements_before_tentative():
    rng = np.random.default_rng(6)
    radar = Radar(position=np.zeros(2), pd=1.0)
    tracker = MultiTargetTracker(TrackerConfig(confirm_m=2, confirm_n=2))
    truth = straight([900, 100], [10, 0], 10)
    run(tracker, radar, [(0, truth)], 5, rng)
    (conf,) = tracker.confirmed_tracks
    # Plant a tentative track right on the next predicted position; it must not steal the hit.
    tracker._spawn(radar.measure_multi(truth[5:6, :2], rng)[0][0], radar.position, radar.R)
    z, _ = radar.measure_multi(truth[5:6, :2], rng)
    tracker.step(z, radar.position, radar.R, DT)
    assert conf.consecutive_misses == 0
