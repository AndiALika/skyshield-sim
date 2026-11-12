import numpy as np
import pytest

torch = pytest.importorskip("torch")

from skyshield.rf_detection.cnn import (
    SpectrogramCNN,
    TrainConfig,
    load_model,
    predict_proba,
    save_model,
    train_detector,
)
from skyshield.rf_detection.features import N_FRAMES, NFFT, DatasetSpec, random_dataset


def test_forward_shape_and_proba_range():
    model = SpectrogramCNN()
    X = np.random.default_rng(0).standard_normal((5, 1, NFFT, N_FRAMES)).astype(np.float32)
    assert model(torch.from_numpy(X)).shape == (5,)
    p = predict_proba(model, X)
    assert p.shape == (5,) and np.all((p >= 0) & (p <= 1))


def test_learns_easy_high_snr_detection():
    ds = DatasetSpec(n=400, snr_range_db=(10.0, 20.0), p_wifi=0.0, p_bt=0.0)
    X, y, _, _ = random_dataset(ds, np.random.default_rng(1))
    Xv, yv, _, _ = random_dataset(DatasetSpec(n=200, snr_range_db=(10.0, 20.0), p_wifi=0.0,
                                              p_bt=0.0), np.random.default_rng(2))
    model, hist = train_detector(X, y, Xv, yv, TrainConfig(epochs=4, batch_size=64, seed=0))
    assert len(hist.val_acc) == 4
    assert max(hist.val_acc) > 0.9, hist.val_acc
    assert np.mean((predict_proba(model, Xv) > 0.5) == yv) == pytest.approx(
        hist.val_acc[hist.best_epoch])


def test_save_load_roundtrip(tmp_path):
    model = SpectrogramCNN()
    model.eval()
    X = np.random.default_rng(3).standard_normal((3, 1, NFFT, N_FRAMES)).astype(np.float32)
    path = tmp_path / "m.pt"
    save_model(model, path)
    assert np.allclose(predict_proba(load_model(path), X), predict_proba(model, X))
