import numpy as np
import pytest

from skyshield.evaluation.metrics import auc, roc_curve, threshold_at_pfa
from skyshield.rf_detection.features import (
    N_FRAMES,
    NFFT,
    DatasetSpec,
    baseline_scores,
    power_spectrogram,
    random_dataset,
    spectrogram,
)
from skyshield.rf_detection.signals import (
    FS,
    N_SAMPLES,
    PROFILES,
    SceneSpec,
    bandlimited_noise,
    gfsk,
    make_scene,
    snr_amplitude,
)


def inband_snr_db(x, noise, bw):
    f = np.fft.fftfreq(len(x), 1 / FS)
    m = np.abs(f) < bw / 2
    return 10 * np.log10(np.sum(np.abs(np.fft.fft(x))[m] ** 2) / np.sum(np.abs(np.fft.fft(noise))[m] ** 2))


def test_waveforms_unit_power():
    rng = np.random.default_rng(0)
    assert np.mean(np.abs(gfsk(20000, 250e3, 1.0, rng)) ** 2) == pytest.approx(1.0)
    assert np.mean(np.abs(bandlimited_noise(20000, 2e6, rng)) ** 2) == pytest.approx(1.0)


def test_bandlimited_noise_stays_in_band():
    rng = np.random.default_rng(1)
    x = bandlimited_noise(1 << 15, 2e6, rng)
    f = np.fft.fftfreq(len(x), 1 / FS)
    P = np.abs(np.fft.fft(x)) ** 2
    assert P[np.abs(f) > 1.2e6].sum() / P.sum() < 1e-3


@pytest.mark.parametrize("snr_db", [-5.0, 0.0, 10.0])
def test_inband_snr_convention(snr_db):
    rng = np.random.default_rng(2)
    n, bw = 1 << 16, 1e6
    x = snr_amplitude(snr_db, bw) * bandlimited_noise(n, bw, rng)
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    assert inband_snr_db(x, noise, bw) == pytest.approx(snr_db, abs=0.3)


def test_gfsk_bandwidth_close_to_carson():
    rng = np.random.default_rng(3)
    rs, h = 250e3, 1.0
    x = gfsk(1 << 16, rs, h, rng)
    f = np.fft.fftfreq(len(x), 1 / FS)
    P = np.abs(np.fft.fft(x)) ** 2
    in_carson = P[np.abs(f) < (h + 1) * rs / 2].sum() / P.sum()
    assert in_carson > 0.95


def test_scene_is_reproducible_and_noise_only_has_unit_power():
    a = make_scene(SceneSpec(drone="B", interferers=("wifi",)), np.random.default_rng(4))
    b = make_scene(SceneSpec(drone="B", interferers=("wifi",)), np.random.default_rng(4))
    assert np.array_equal(a, b)
    n = make_scene(SceneSpec(), np.random.default_rng(5))
    assert len(n) == N_SAMPLES
    assert np.mean(np.abs(n) ** 2) == pytest.approx(1.0, abs=0.02)


def test_every_profile_and_interferer_adds_energy():
    for key in PROFILES:
        x = make_scene(SceneSpec(drone=key, drone_snr_db=20.0), np.random.default_rng(6))
        assert np.mean(np.abs(x) ** 2) > 1.05, key
    for name in ("wifi", "bt"):
        x = make_scene(SceneSpec(interferers=(name,), interferer_snr_db={name: 20.0}),
                       np.random.default_rng(7))
        assert np.mean(np.abs(x) ** 2) > 1.05, name
    with pytest.raises(ValueError):
        make_scene(SceneSpec(interferers=("radar",)), np.random.default_rng(0))


def test_power_spectrogram_noise_calibration():
    rng = np.random.default_rng(8)
    n = make_scene(SceneSpec(), rng)
    P = power_spectrogram(n)
    assert P.shape == (NFFT, N_FRAMES)
    assert P.mean() * NFFT == pytest.approx(1.0, abs=0.05)


def test_spectrogram_shows_narrowband_burst_where_expected():
    rng = np.random.default_rng(9)
    x = make_scene(SceneSpec(drone="C", drone_snr_db=20.0), rng)
    S = spectrogram(x)
    assert np.median(S) == pytest.approx(0.0, abs=1e-5)
    assert S.max() > 15.0  # burst stands well above the noise floor


def test_baselines_respond_to_signal():
    rng = np.random.default_rng(10)
    noise = baseline_scores(make_scene(SceneSpec(), rng))
    sig = baseline_scores(make_scene(SceneSpec(drone="A", drone_snr_db=15.0), rng))
    assert abs(noise[0]) < 0.1  # 0 dB total power for unit noise
    assert sig[1] > noise[1] + 5.0


def test_random_dataset_shapes_and_balance():
    X, y, B, specs = random_dataset(DatasetSpec(n=60), np.random.default_rng(11))
    assert X.shape == (60, 1, NFFT, N_FRAMES) and X.dtype == np.float32
    assert B.shape == (60, 2)
    assert set(np.unique(y)) <= {0, 1}
    assert all((s.drone is not None) == bool(lab) for s, lab in zip(specs, y))
    assert 10 < y.sum() < 50


def test_roc_and_auc():
    labels = np.array([0, 0, 1, 1])
    assert auc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == pytest.approx(1.0)
    assert auc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == pytest.approx(0.0)
    assert auc(np.ones(4), labels) == pytest.approx(0.5)  # all tied -> chance
    pfa, pd = roc_curve(np.array([0.1, 0.4, 0.35, 0.8]), labels)
    assert pfa[0] == pd[0] == 0.0 and pfa[-1] == pd[-1] == 1.0
    rng = np.random.default_rng(0)
    s = rng.standard_normal(4000)
    lab = rng.integers(0, 2, 4000)
    assert auc(s, lab) == pytest.approx(0.5, abs=0.03)


def test_threshold_at_pfa():
    neg = np.arange(1000, dtype=float)
    t = threshold_at_pfa(neg, 0.01)
    assert np.mean(neg > t) <= 0.01
    assert np.mean(neg > t - 1) > 0.01
    with pytest.raises(ValueError):
        threshold_at_pfa(neg, 0.0)
