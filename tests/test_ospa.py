import numpy as np
import pytest

from skyshield.evaluation.metrics import ospa


def test_both_empty_is_zero():
    assert ospa(np.empty((0, 2)), np.empty((0, 2)), c=100.0).distance == 0.0


def test_one_empty_is_cutoff():
    X = np.array([[0.0, 0.0], [10.0, 10.0]])
    res = ospa(X, np.empty((0, 2)), c=100.0)
    assert res.distance == pytest.approx(100.0)
    assert res.localisation == 0.0
    assert res.cardinality == pytest.approx(100.0)


def test_identical_sets_zero_regardless_of_order():
    X = np.array([[0.0, 0.0], [100.0, 50.0], [-30.0, 7.0]])
    assert ospa(X, X[::-1], c=100.0).distance == pytest.approx(0.0)


def test_known_value_single_pair():
    # One point each, 50 m apart: OSPA = 50 for any p (below the cutoff).
    X = np.array([[0.0, 0.0]])
    Y = np.array([[30.0, 40.0]])
    assert ospa(X, Y, c=100.0, p=1).distance == pytest.approx(50.0)
    assert ospa(X, Y, c=100.0, p=2).distance == pytest.approx(50.0)


def test_known_value_with_cardinality_error():
    # X = {a}, Y = {a', b} with |a - a'| = 10, c = 100, p = 1: (10 + 100) / 2 = 55.
    X = np.array([[0.0, 0.0]])
    Y = np.array([[10.0, 0.0], [500.0, 500.0]])
    res = ospa(X, Y, c=100.0, p=1)
    assert res.distance == pytest.approx(55.0)
    assert res.localisation == pytest.approx(5.0)
    assert res.cardinality == pytest.approx(50.0)


def test_uses_optimal_assignment():
    # p=2 minimises the sum of squared distances:
    # (x0-y1, x1-y0) -> 1 + 100 = 101   vs   (x0-y0, x1-y1) -> 100 + 361 = 461.
    X = np.array([[0.0, 0.0], [20.0, 0.0]])
    Y = np.array([[10.0, 0.0], [1.0, 0.0]])
    res = ospa(X, Y, c=1000.0, p=2)
    assert res.distance == pytest.approx(np.sqrt(101 / 2))


def test_symmetric_and_bounded():
    rng = np.random.default_rng(3)
    for _ in range(50):
        X = rng.uniform(-500, 500, size=(rng.integers(0, 5), 2))
        Y = rng.uniform(-500, 500, size=(rng.integers(0, 5), 2))
        a = ospa(X, Y, c=200.0).distance
        b = ospa(Y, X, c=200.0).distance
        assert a == pytest.approx(b)
        assert 0.0 <= a <= 200.0 + 1e-9


def test_distances_cut_off_at_c():
    X = np.array([[0.0, 0.0]])
    Y = np.array([[1e6, 0.0]])
    assert ospa(X, Y, c=100.0).distance == pytest.approx(100.0)


def test_components_recombine_to_total():
    X = np.array([[0.0, 0.0], [50.0, 50.0]])
    Y = np.array([[3.0, 4.0], [60.0, 50.0], [900.0, 900.0]])
    res = ospa(X, Y, c=100.0, p=2)
    assert res.localisation**2 + res.cardinality**2 == pytest.approx(res.distance**2)


def test_rejects_bad_params():
    with pytest.raises(ValueError):
        ospa(np.zeros((1, 2)), np.zeros((1, 2)), c=0.0)
    with pytest.raises(ValueError):
        ospa(np.zeros((1, 2)), np.zeros((1, 2)), c=10.0, p=0.5)
