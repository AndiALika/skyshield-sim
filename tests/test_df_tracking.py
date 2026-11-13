import numpy as np
import pytest

from skyshield.tracking.ekf import EKF
from skyshield.tracking.gnn import gnn_associate_bearings
from skyshield.tracking.manager import MultiTargetTracker, TrackerConfig
from skyshield.tracking.triangulation import PendingBearing, find_fixes, intersect, refine

SITES = [np.array([600.0, 300.0]), np.array([-500.0, 600.0]), np.array([100.0, -700.0])]
SIG = np.deg2rad(1.0)


def bearing(xy, site):
    return float(np.arctan2(xy[1] - site[1], xy[0] - site[0]))


def test_intersect_and_refine_are_exact_without_noise():
    p = np.array([1200.0, 900.0])
    xy, r1, r2 = intersect(SITES[0], bearing(p, SITES[0]), SITES[1], bearing(p, SITES[1]))
    np.testing.assert_allclose(xy, p, atol=1e-6)
    assert r1 > 0 and r2 > 0
    assert intersect(SITES[0], 0.3, SITES[0] + [0, 10], 0.3) is None  # parallel
    xy2, cov = refine(p + [80.0, -60.0], SITES, [bearing(p, s) for s in SITES], [SIG] * 3)
    # Degenerate geometry: one bearing constrains only one direction, so N is singular.
    assert refine(SITES[0], SITES[:1], [0.1], [SIG]) is None
    np.testing.assert_allclose(xy2, p, atol=1e-3)
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_ghosts_need_a_third_site():
    targets = [np.array([1500.0, 200.0]), np.array([-800.0, 1500.0])]
    pend = [PendingBearing(i, s, bearing(t, s), SIG) for i, s in enumerate(SITES) for t in targets]
    fixes = find_fixes(pend)
    assert len(fixes) == 2
    for f in fixes:
        assert min(np.linalg.norm(f.xy - t) for t in targets) < 1.0
    two_sites = [p for p in pend if p.sensor_id < 2]
    assert find_fixes(two_sites) == []  # could be ghosts: no third site to confirm
    # Without confirmation, crossings (real or ghost) are accepted greedily.
    assert len(find_fixes(two_sites, min_support=0)) >= 1


def test_bearing_gnn_prefers_consistent_track_and_gates():
    a = EKF(np.array([1000.0, 0.0, 0, 0]), np.diag([400.0, 400.0, 100, 100]))
    b = EKF(np.array([0.0, 1000.0, 0, 0]), np.diag([400.0, 400.0, 100, 100]))
    site = np.zeros(2)
    res = gnn_associate_bearings([a, b], [np.pi / 2 + 0.005, 0.002, -2.0], [SIG] * 3, site)
    assert sorted(res.pairs) == [(0, 1), (1, 0)]
    assert res.unassigned_meas == [2]


def test_bearing_ekf_is_consistent():
    """Mean NEES near 4 with three DF sites and truth from the filter's own motion model."""
    q, dt = 0.1, 1.0
    Q = EKF(np.zeros(4), np.eye(4), q=q).Q(dt)
    L = np.linalg.cholesky(Q + 1e-12 * np.eye(4))
    nees = []
    for seed in range(60):
        rng = np.random.default_rng(seed)
        s = np.array([1500.0, 800.0, rng.normal(0, 5), rng.normal(0, 5)])
        f = EKF(s + np.r_[rng.normal(0, 50, 2), rng.normal(0, 5, 2)],
                np.diag([2500.0, 2500.0, 25.0, 25.0]), q=q)
        for k in range(120):
            if k:
                s = EKF.F(dt) @ s + L @ rng.standard_normal(4)
                f.predict(dt)
            for site in SITES:
                f.update_bearing(bearing(s, site) + SIG * rng.standard_normal(), site, SIG)
            if k > 20:
                e = s - f.x
                nees.append(e @ np.linalg.solve(f.P, e))
    assert 3.4 < np.mean(nees) < 4.8, np.mean(nees)


def run_tracker(sites, n_steps=12, seed=0, config=None):
    rng = np.random.default_rng(seed)
    tr = MultiTargetTracker(config or TrackerConfig(q=0.5))
    p, v = np.array([1200.0, 900.0]), np.array([-10.0, 5.0])
    for k in range(n_steps):
        tr.begin_step(float(k))
        xy = p + k * v
        for i, site in enumerate(sites):
            tr.update_bearings([bearing(xy, site) + SIG * rng.standard_normal()], [SIG], site, i)
        tr.end_step()
    return tr, p + (n_steps - 1) * v


def test_single_df_site_never_starts_a_track():
    tr, _ = run_tracker(SITES[:1])
    assert tr.tracks == []


def test_three_df_sites_build_a_confirmed_track():
    tr, truth = run_tracker(SITES)
    conf = tr.confirmed_tracks
    assert len(conf) == 1 and len(tr.tracks) == 1
    assert np.linalg.norm(conf[0].position - truth) < 40.0


def test_df_initiation_can_be_disabled():
    tr, _ = run_tracker(SITES, config=TrackerConfig(df_initiate=False))
    assert tr.tracks == []
    with pytest.raises(RuntimeError):
        MultiTargetTracker().update_bearings([0.0], [SIG], SITES[0], 0)


def test_radar_is_the_authority_where_it_reports():
    """Bearings alone keep a track alive only where no covering radar reported."""
    def run(radar_reports: bool, rule_on: bool = True):
        tr, _ = run_tracker(SITES, config=TrackerConfig(q=0.5,
                                                        df_hits_in_radar_coverage=not rule_on))
        assert len(tr.confirmed_tracks) == 1
        radar = np.array([1100.0, 900.0])  # right next to the target: it would see it
        for k in range(12, 24):
            tr.begin_step(float(k))
            xy = np.array([1200.0, 900.0]) + k * np.array([-10.0, 5.0])
            if radar_reports:  # empty scan: the radar looked and saw nothing
                tr.update([], radar, np.diag([25.0, 3e-4]), coverage_m=3000.0)
            for i, site in enumerate(SITES):
                tr.update_bearings([bearing(xy, site)], [SIG], site, i)
            tr.end_step()
        return len(tr.tracks)

    assert run(radar_reports=False) == 1  # radar cut off: DF keeps the track
    assert run(radar_reports=True) == 0  # radar heard and saw nothing: track dropped
    assert run(radar_reports=True, rule_on=False) == 1  # rule off: DF props it up


def test_adding_df_leaves_radar_draws_unchanged():
    from skyshield.integration.network import build_sim
    from skyshield.scenario.multi_target import MultiTargetScenario

    logs = {}
    for sensors in ("radar", "radar+df"):
        sim = build_sim("plain", None, 0.0, sensors=sensors)
        sc = sim.scenario
        sim.scenario = MultiTargetScenario(dt=sc.dt, targets=sc.targets, n_steps=25)
        logs[sensors] = sim.run(np.random.default_rng(3), np.random.default_rng(4), record=True)
    for a, b in zip(logs["radar"].frames, logs["radar+df"].frames):
        for xa, xb in zip(a.det_xy, b.det_xy):
            np.testing.assert_array_equal(xa, xb)
    f = logs["radar+df"].frames[-1]
    assert len(f.df_bearings) == 3 and f.df_delivered.all()
    assert logs["radar+df"].delivered.shape == (25, 6)
    with pytest.raises(ValueError):
        build_sim("plain", None, 0.0, sensors="sonar")
