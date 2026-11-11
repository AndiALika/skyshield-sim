import numpy as np

from skyshield.scenario.multi_target import MultiTargetScenario, TargetTruth, crossing_scenario
from skyshield.sensors.radar import Radar


def test_target_truth_lifetime():
    t = TargetTruth(7, 5, np.arange(12.0).reshape(3, 4))
    assert t.end_step == 8
    assert t.state_at(4) is None
    assert np.array_equal(t.state_at(5), [0, 1, 2, 3])
    assert np.array_equal(t.state_at(7), [8, 9, 10, 11])
    assert t.state_at(8) is None


def test_truth_at_and_cardinality():
    a = TargetTruth(0, 0, np.zeros((5, 4)))
    b = TargetTruth(1, 3, np.ones((4, 4)))
    sc = MultiTargetScenario(dt=1.0, targets=[a, b])
    assert sc.n_steps == 7
    ids, states = sc.truth_at(3)
    assert ids == [0, 1] and states.shape == (2, 4)
    ids, states = sc.truth_at(6)
    assert ids == [1]
    assert list(sc.cardinality()) == [1, 1, 1, 2, 2, 1, 1]


def test_empty_step_returns_empty_array():
    sc = MultiTargetScenario(dt=1.0, targets=[TargetTruth(0, 2, np.zeros((1, 4)))])
    ids, states = sc.truth_at(0)
    assert ids == [] and states.shape == (0, 4)


def test_crossing_scenario_is_reproducible_and_has_births_deaths():
    a = crossing_scenario(rng=np.random.default_rng(1))
    b = crossing_scenario(rng=np.random.default_rng(1))
    for ta, tb in zip(a.targets, b.targets):
        assert np.array_equal(ta.states, tb.states)
    card = a.cardinality()
    assert card.max() == 4
    assert card[0] == 2  # only T0 and T3 at start
    assert card[-1] < card.max()  # targets leave before the end


def test_measure_multi_origins_and_clutter():
    rng = np.random.default_rng(0)
    radar = Radar(position=np.zeros(2), pd=1.0, clutter_rate=5.0)
    targets = np.array([[100.0, 0.0], [0.0, 200.0], [5000.0, 0.0]])  # last one out of range
    n_clutter = []
    for _ in range(500):
        zs, origins = radar.measure_multi(targets, rng)
        assert len(zs) == len(origins)
        assert origins[:2] == [0, 1]
        assert 2 not in origins
        n_clutter.append(origins.count(-1))
    assert abs(np.mean(n_clutter) - 5.0) < 0.3


def test_measure_multi_no_targets():
    rng = np.random.default_rng(0)
    radar = Radar(position=np.zeros(2), clutter_rate=0.0)
    assert radar.measure_multi(np.empty((0, 2)), rng) == ([], [])


def test_to_cartesian_roundtrip():
    radar = Radar(position=np.array([10.0, -20.0]))
    z = np.array([100.0, np.pi / 2])
    assert np.allclose(radar.to_cartesian(z), [10.0, 80.0])
