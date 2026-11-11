import numpy as np
import pytest

from skyshield.tracking.ekf import EKF
from skyshield.tracking.gnn import gate_threshold, gnn_associate, mahalanobis_matrix

SENSOR = np.zeros(2)
R = np.diag([5.0**2, np.deg2rad(1.0) ** 2])


def make_track(x, y, pos_std=10.0):
    P = np.diag([pos_std**2, pos_std**2, 10.0, 10.0])
    return EKF(np.array([x, y, 0.0, 0.0]), P)


def z_of(x, y):
    return np.array([np.hypot(x, y), np.arctan2(y, x)])


def test_gate_threshold_matches_chi2():
    assert gate_threshold(0.99) == pytest.approx(9.2103, abs=1e-3)


def test_empty_inputs():
    a = gnn_associate([], [z_of(1, 1)], SENSOR, R)
    assert a.pairs == [] and a.unassigned_meas == [0]
    a = gnn_associate([make_track(100, 0)], [], SENSOR, R)
    assert a.pairs == [] and a.unassigned_tracks == [0]


def test_close_measurement_assigned_far_one_not():
    t = make_track(1000.0, 0.0)
    a = gnn_associate([t], [z_of(1003.0, 2.0), z_of(1500.0, 300.0)], SENSOR, R)
    assert a.pairs == [(0, 0)]
    assert a.unassigned_meas == [1]


def test_outside_gate_is_a_miss():
    t = make_track(1000.0, 0.0)
    a = gnn_associate([t], [z_of(1200.0, 0.0)], SENSOR, R)
    assert a.pairs == []
    assert a.unassigned_tracks == [0] and a.unassigned_meas == [0]


def test_measurement_order_does_not_matter():
    tracks = [make_track(1000.0, 0.0), make_track(0.0, 1000.0)]
    zs = [z_of(2.0, 1004.0), z_of(998.0, -3.0)]
    a = gnn_associate(tracks, zs, SENSOR, R)
    assert sorted(a.pairs) == [(0, 1), (1, 0)]


def test_global_optimum_beats_greedy():
    # Range std: track 0 ~20.6 m, track 1 ~7.1 m.
    # d2(t0, z0) = 26^2/425 = 1.6 < d2(t0, z1) = 35^2/425 = 2.9, so greedy from track 0 takes z0,
    # which is track 1's only feasible measurement (z1 is 65 m / 9 sigma from track 1).
    # The global optimum gives both tracks a measurement.
    tracks = [make_track(1000.0, 0.0, pos_std=20.0), make_track(1030.0, 0.0, pos_std=5.0)]
    zs = [z_of(1026.0, 0.0), z_of(965.0, 0.0)]
    d2, _ = mahalanobis_matrix(tracks, zs, SENSOR, R)
    assert d2[0, 0] < d2[0, 1] and d2[1, 1] > gate_threshold(0.99)  # the greedy trap is real
    a = gnn_associate(tracks, zs, SENSOR, R)
    assert sorted(a.pairs) == [(0, 1), (1, 0)]


def test_conflict_goes_to_closer_track():
    tracks = [make_track(1000.0, 0.0), make_track(1020.0, 0.0)]
    a = gnn_associate(tracks, [z_of(1017.0, 0.0)], SENSOR, R)
    assert a.pairs == [(1, 0)]
    assert a.unassigned_tracks == [0]


def test_bearing_wraparound_in_distance():
    # Target just across the -pi/+pi bearing cut must still be close in Mahalanobis distance.
    t = make_track(-1000.0, 1.0)
    z = z_of(-1000.0, -1.0)
    d2, _ = mahalanobis_matrix([t], [z], SENSOR, R)
    assert d2[0, 0] < 1.0
