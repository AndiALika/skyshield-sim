"""Spectrogram features and dataset generation for the RF detector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skyshield.rf_detection.signals import (
    HELD_OUT_MODELS,
    N_SAMPLES,
    TRAIN_MODELS,
    SceneSpec,
    make_scene,
)

NFFT = 64  # frequency bins (156 kHz each at 10 MS/s)
N_FRAMES = 128  # time frames (51 us each for 65536 samples)


def power_spectrogram(iq: np.ndarray, nfft: int = NFFT, n_frames: int = N_FRAMES) -> np.ndarray:
    """Linear power spectrogram, shape (nfft, n_frames), frequency axis centred (fftshift).

    Non-overlapping Hann-windowed FFTs; consecutive FFTs are averaged in groups so that every
    sample contributes (Welch-style variance reduction). Scaled so that unit-power white noise
    has an expected value of 1 / nfft in every cell (i.e. power per bin).
    """
    iq = np.asarray(iq)
    group = max(len(iq) // nfft // n_frames, 1)
    usable = n_frames * group * nfft
    if usable > len(iq):
        raise ValueError("observation too short for the requested spectrogram size")
    w = np.hanning(nfft).astype(np.float32)
    blocks = iq[:usable].reshape(n_frames * group, nfft) * w
    P = np.abs(np.fft.fftshift(np.fft.fft(blocks, axis=1), axes=1)) ** 2 / (nfft * np.sum(w**2))
    return P.reshape(n_frames, group, nfft).mean(axis=1).T


def spectrogram(iq: np.ndarray, nfft: int = NFFT, n_frames: int = N_FRAMES) -> np.ndarray:
    """CNN input: log-power spectrogram in dB relative to the per-example median.

    Subtracting the median removes the unknown receiver gain, so the network has to rely on
    spectro-temporal structure rather than absolute level.
    """
    S = 10 * np.log10(power_spectrogram(iq, nfft, n_frames) + 1e-12)
    return (S - np.median(S)).astype(np.float32)


def baseline_scores(iq: np.ndarray) -> np.ndarray:
    """Classical detector statistics (noise power is known and equal to 1).

    [0] energy detector: total received power over the whole band and observation (dB).
    [1] time-frequency peak: strongest spectrogram cell relative to the noise level (dB),
        i.e. an energy detector matched to short narrowband bursts.
    """
    energy = 10 * np.log10(np.mean(np.abs(iq) ** 2))
    peak = 10 * np.log10(power_spectrogram(iq).max() * NFFT)
    return np.array([energy, peak], dtype=np.float32)


@dataclass
class DatasetSpec:
    n: int
    models: tuple[str, ...] = TRAIN_MODELS
    snr_range_db: tuple[float, float] = (-10.0, 20.0)
    p_drone: float = 0.5
    p_wifi: float = 0.4
    p_bt: float = 0.4
    interferer_snr_range_db: tuple[float, float] = (0.0, 25.0)


def sample_spec(ds: DatasetSpec, rng: np.random.Generator) -> SceneSpec:
    drone = rng.choice(ds.models) if rng.random() < ds.p_drone else None
    interf = []
    if rng.random() < ds.p_wifi:
        interf.append("wifi")
    if rng.random() < ds.p_bt:
        interf.append("bt")
    return SceneSpec(
        drone=None if drone is None else str(drone),
        drone_snr_db=float(rng.uniform(*ds.snr_range_db)),
        interferers=tuple(interf),
        interferer_snr_db={k: float(rng.uniform(*ds.interferer_snr_range_db)) for k in interf},
    )


def build_dataset(
    specs: list[SceneSpec], rng: np.random.Generator, n_samples: int = N_SAMPLES
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X (n, 1, NFFT, N_FRAMES) float32 spectrograms, y (n,) with 1 = drone present, and
    B (n, 2) classical baseline scores (see ``baseline_scores``)."""
    X = np.empty((len(specs), 1, NFFT, N_FRAMES), dtype=np.float32)
    y = np.empty(len(specs), dtype=np.int64)
    B = np.empty((len(specs), 2), dtype=np.float32)
    for i, s in enumerate(specs):
        iq = make_scene(s, rng, n_samples)
        X[i, 0] = spectrogram(iq)
        B[i] = baseline_scores(iq)
        y[i] = int(s.drone is not None)
    return X, y, B


def random_dataset(ds: DatasetSpec, rng: np.random.Generator):
    specs = [sample_spec(ds, rng) for _ in range(ds.n)]
    X, y, B = build_dataset(specs, rng)
    return X, y, B, specs


__all__ = [
    "HELD_OUT_MODELS",
    "NFFT",
    "N_FRAMES",
    "TRAIN_MODELS",
    "DatasetSpec",
    "baseline_scores",
    "build_dataset",
    "power_spectrogram",
    "random_dataset",
    "sample_spec",
    "spectrogram",
]
