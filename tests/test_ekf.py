import numpy as np

from skyshield.scenario.trajectories import loiter_trajectory, waypoint_trajectory
from skyshield.sensors.radar import Radar
from skyshield.tracking.ekf import EKF


def test_waypoint_trajectory_constant_speed():
    traj = waypoint_trajectory(np.array([[0.0, 0.0], [1000.0, 0.0]]), speed=10.0, dt=1.0)
    speeds = np.hypot(traj[:, 2], traj[:, 3])
    assert np.allclose(speeds, 10.0)
    assert traj[-1, 0] > 950.0


def test_loiter_stays_on_circle():
    traj = loiter_trajectory(np.array([0.0, 0.0]), radius=200.0, speed=10.0, dt=0.5, duration=100.0)
    assert np.allclose(np.hypot(traj[:, 0], traj[:, 1]), 200.0)


def test_radar_noise_statistics():
    rng = np.random.default_rng(0)
    radar = Radar(position=np.array([0.0, 0.0]), pd=1.0)
    target = np.array([1000.0, 0.0])
    zs = np.array([radar.measure(target, rng)[0] for _ in range(5000)])
    assert abs(zs[:, 0].mean() - 1000.0) < 1.0
    assert abs(zs[:, 0].std() - radar.sigma_r) < 0.5


def test_radar_misses_out_of_range():
    rng = np.random.default_rng(0)
    radar = Radar(position=np.array([0.0, 0.0]), pd=1.0, max_range=500.0)
    assert radar.measure(np.array([1000.0, 0.0]), rng) == []


def test_ekf_nees_consistent():
    """NEES consistency check.

    Truth must be simulated from the same constant-velocity (white-noise-acceleration) model the
    filter assumes. If truth has no process noise, any q > 0 makes the filter conservative and the
    mean NEES falls well below the state dimension (4) even though the filter is correct.
    """
    dt, q, v_sigma = 1.0, 0.1, 30.0
    Q = EKF(np.zeros(4), np.eye(4), q=q).Q(dt)
    L = np.linalg.cholesky(Q + 1e-12 * np.eye(4))
    F = EKF.F(dt)
    radar = Radar(position=np.array([0.0, 0.0]), pd=1.0, max_range=1e9)  # no range limit here
    nees_all = []
    for seed in range(100):
        rng = np.random.default_rng(seed)
        s = np.array([-1500.0, 1200.0, rng.normal(0, v_sigma / 3), rng.normal(0, v_sigma / 3)])
        ekf = None
        for k in range(150):
            if k > 0:
                s = F @ s + L @ rng.standard_normal(4)
            z = radar.measure(s[:2], rng)[0]
            if ekf is None:
                ekf = EKF.from_measurement(z, radar.position, radar.R, v_sigma=v_sigma, q=q)
                continue
            ekf.predict(dt)
            ekf.update(z, radar.position, radar.R)
            if k > 20:  # skip convergence transient
                e = s - ekf.x
                nees_all.append(e @ np.linalg.solve(ekf.P, e))
    mean_nees = float(np.mean(nees_all))
    assert 3.5 < mean_nees < 4.5, mean_nees


def test_ekf_improves_over_raw_measurements():
    rng = np.random.default_rng(1)
    dt = 1.0
    truth = waypoint_trajectory(np.array([[-1500.0, 1000.0], [1500.0, -800.0]]), 15.0, dt)
    radar = Radar(position=np.array([0.0, 0.0]), pd=1.0)
    ekf, raw_err, ekf_err = None, [], []
    for k, s in enumerate(truth):
        z = radar.measure(s[:2], rng)[0]
        pos = radar.position + z[0] * np.array([np.cos(z[1]), np.sin(z[1])])
        if ekf is None:
            ekf = EKF.from_measurement(z, radar.position, radar.R, q=0.05)
            continue
        ekf.predict(dt)
        ekf.update(z, radar.position, radar.R)
        if k > 15:
            raw_err.append(np.linalg.norm(pos - s[:2]))
            ekf_err.append(np.linalg.norm(ekf.x[:2] - s[:2]))
    assert np.mean(ekf_err) < np.mean(raw_err)
