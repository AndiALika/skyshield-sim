import json

import numpy as np
import pytest

from skyshield.sensors.rf_df import DetectionCurve, RFDFSensor, path_loss_db

CURVE = DetectionCurve(np.array([-10.0, 0.0, 10.0]), np.array([0.0, 0.5, 1.0]), pfa=0.01)


def test_detection_curve_interpolates_and_clamps():
    assert CURVE(-50.0) == 0.0
    assert CURVE(5.0) == pytest.approx(0.75)
    assert CURVE(99.0) == 1.0


def test_detection_curve_validation():
    with pytest.raises(ValueError):
        DetectionCurve(np.array([0.0, 0.0]), np.array([0.1, 0.2]))
    with pytest.raises(ValueError):
        DetectionCurve(np.array([0.0, 1.0]), np.array([0.1, 1.2]))


def test_detection_curve_from_metrics(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"snr_grid_db": [-5, 0, 5], "pfa": 0.02,
                             "cnn": {"pd_vs_snr": {"mixed": [0.1, 0.6, 0.95]}}}))
    c = DetectionCurve.from_metrics(p)
    assert c.pfa == 0.02 and c(0.0) == pytest.approx(0.6)


def test_free_space_path_loss_textbook_value():
    # FSPL at 2.44 GHz, 1 km is ~100.2 dB.
    assert path_loss_db(1000.0, 2.44e9, 2.0) == pytest.approx(100.2, abs=0.1)
    # Steeper exponent -> more loss beyond 1 m.
    assert path_loss_db(1000.0, 2.44e9, 3.0) > path_loss_db(1000.0, 2.44e9, 2.0)


def test_snr_drops_6db_per_distance_doubling_in_free_space():
    s = RFDFSensor(np.zeros(2), CURVE, path_loss_exp=2.0)
    a = s.snr_db(np.array([500.0, 0.0]), 20.0)
    b = s.snr_db(np.array([1000.0, 0.0]), 20.0)
    assert a - b == pytest.approx(6.02, abs=0.01)


def test_noise_floor():
    s = RFDFSensor(np.zeros(2), CURVE, bandwidth_hz=1e6, noise_figure_db=6.0)
    assert s.noise_dbm == pytest.approx(-108.0)


def test_bearing_sigma_scales_and_clips():
    s = RFDFSensor(np.zeros(2), CURVE)
    assert s.bearing_sigma(s.snr_ref_db) == pytest.approx(s.sigma_b_ref)
    assert s.bearing_sigma(s.snr_ref_db - 6.0) == pytest.approx(2 * s.sigma_b_ref, rel=0.01)
    assert s.bearing_sigma(100.0) == s.sigma_b_min
    assert s.bearing_sigma(-100.0) == s.sigma_b_max


def test_measure_pd_follows_curve_and_bearing_is_unbiased():
    rng = np.random.default_rng(0)
    s = RFDFSensor(np.zeros(2), CURVE, false_alarm_rate=0.0)
    target = np.array([[0.0, 1500.0]])
    eirp = 0.0 - float(s.snr_db(target[0], 0.0)) + 5.0  # puts the target at SNR = 5 dB
    dets = [s.measure_multi(target, eirp, rng) for _ in range(4000)]
    pd = np.mean([len(d) for d in dets])
    assert pd == pytest.approx(0.75, abs=0.03)
    b = np.array([d[0].bearing for d in dets if d])
    assert b.mean() == pytest.approx(np.pi / 2, abs=0.005)
    assert b.std() == pytest.approx(s.bearing_sigma(5.0), rel=0.1)


def test_false_alarms_marked_and_counted():
    rng = np.random.default_rng(1)
    s = RFDFSensor(np.zeros(2), CURVE, false_alarm_rate=0.5)
    n = [sum(d.origin == -1 for d in s.measure_multi(np.empty((0, 2)), 20.0, rng))
         for _ in range(4000)]
    assert np.mean(n) == pytest.approx(0.5, abs=0.05)
