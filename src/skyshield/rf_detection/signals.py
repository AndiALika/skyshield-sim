"""Synthetic complex-baseband RF scenes for drone detection.

The receiver observes a slice of the 2.4 GHz ISM band: complex baseband at ``FS`` samples/s for
``N_SAMPLES`` samples (6.55 ms at the defaults). Bandwidths are scaled to fit this slice: a Wi-Fi
burst occupies most of it, as a 20 MHz channel would in a real 20-25 MHz capture.

Emitters (all generic, no real product is modelled):

- Drone control link: frequency-hopping bursts, GFSK or chirp (CSS) modulated.
- Drone video downlink: continuous wideband OFDM-like signal (band-limited Gaussian).
- Wi-Fi: bursty wideband OFDM-like packets with random gaps (CSMA-like).
- Bluetooth: 1 Msym/s GFSK hopping on a 625 us slot grid.

SNR convention
--------------
Noise is complex AWGN with unit total power over FS, so the noise power in bandwidth B is B/FS.
Every emitter's SNR is **in-band**: signal power while transmitting divided by the noise power in
its own occupied bandwidth. That is the quantity a link budget gives (P_rx / (k T B F)).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FS = 10e6  # sample rate (Hz)
N_SAMPLES = 65536  # samples per observation (6.55 ms)


# ----- primitive waveforms (unit power) -------------------------------------------------------


def gfsk(n: int, sym_rate: float, h: float, rng: np.random.Generator, fs: float = FS, bt: float = 0.5):
    """Gaussian frequency-shift keying with modulation index h. Unit-power complex baseband."""
    sps = max(round(fs / sym_rate), 1)
    n_sym = n // sps + 2
    bits = rng.integers(0, 2, n_sym) * 2 - 1
    nrz = np.repeat(bits, sps).astype(float)
    # Gaussian pulse-shaping filter on the frequency pulse.
    span = 3 * sps
    t = np.arange(-span, span + 1) / sps
    sigma = np.sqrt(np.log(2)) / (2 * np.pi * bt)
    g = np.exp(-(t**2) / (2 * sigma**2))
    g /= g.sum()
    freq = np.convolve(nrz, g, mode="same")[:n] * (h * sym_rate / 2)
    phase = 2 * np.pi * np.cumsum(freq) / fs + rng.uniform(0, 2 * np.pi)
    return np.exp(1j * phase)


def chirp(n: int, bw: float, sym_rate: float, rng: np.random.Generator, fs: float = FS):
    """Chirp spread spectrum: repeated linear up-chirps across bw, random cyclic shift per symbol."""
    sps = max(round(fs / sym_rate), 2)
    k = np.arange(sps)
    out = np.empty(n, dtype=complex)
    for start in range(0, n, sps):
        shift = rng.integers(0, sps)
        f = (((k + shift) % sps) / sps - 0.5) * bw
        ph = 2 * np.pi * np.cumsum(f) / fs
        seg = np.exp(1j * ph)
        out[start : start + sps] = seg[: min(sps, n - start)]
    return out


def bandlimited_noise(n: int, bw: float, rng: np.random.Generator, fs: float = FS):
    """Complex Gaussian with a flat spectrum over |f| < bw/2 (OFDM-like). Unit power."""
    X = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    f = np.fft.fftfreq(n, 1 / fs)
    edge = 0.05 * bw
    mask = np.clip((bw / 2 + edge - np.abs(f)) / (2 * edge), 0.0, 1.0)  # short linear roll-off
    x = np.fft.ifft(np.fft.fft(X) * mask)
    return x / np.sqrt(np.mean(np.abs(x) ** 2))


def shift(x: np.ndarray, f0: float, start: int = 0, fs: float = FS) -> np.ndarray:
    t = (start + np.arange(len(x))) / fs
    return x * np.exp(2j * np.pi * f0 * t)


def snr_amplitude(snr_db: float, bw: float, fs: float = FS) -> float:
    """Amplitude for a unit-power waveform to reach the given in-band SNR (unit-power noise)."""
    return float(np.sqrt(10 ** (snr_db / 10) * bw / fs))


# ----- emitters -------------------------------------------------------------------------------


@dataclass(frozen=True)
class DroneProfile:
    """A synthetic drone RF signature. Parameters are generic, not taken from real products."""

    name: str
    ctrl_mod: str  # "gfsk" or "chirp"
    ctrl_sym_rate: float  # symbols/s
    ctrl_bw: float  # occupied bandwidth (Hz)
    hop_period: float  # s between hops
    burst_len: float  # s on-air per hop
    video_bw: float | None = None  # Hz, None = no video downlink
    video_snr_offset_db: float = 3.0


# Training models A-D and two held-out models: E interpolates the training parameters,
# F uses a modulation (chirp) never seen in training.
PROFILES: dict[str, DroneProfile] = {
    "A": DroneProfile("A", "gfsk", 250e3, 0.5e6, 2.0e-3, 0.5e-3),
    "B": DroneProfile("B", "gfsk", 500e3, 1.0e6, 1.0e-3, 0.3e-3, video_bw=4e6),
    "C": DroneProfile("C", "gfsk", 100e3, 0.25e6, 4.0e-3, 1.5e-3),
    "D": DroneProfile("D", "gfsk", 1e6, 1.5e6, 0.8e-3, 0.4e-3, video_bw=8e6),
    "E": DroneProfile("E", "gfsk", 300e3, 0.6e6, 1.5e-3, 0.6e-3, video_bw=5e6),
    "F": DroneProfile("F", "chirp", 4e3, 0.5e6, 3.0e-3, 1.0e-3),
}
TRAIN_MODELS = ("A", "B", "C", "D")
HELD_OUT_MODELS = ("E", "F")


def add_drone(buf: np.ndarray, p: DroneProfile, snr_db: float, rng: np.random.Generator) -> None:
    n = len(buf)
    hop = int(p.hop_period * FS)
    burst = int(p.burst_len * FS)
    t0 = -int(rng.integers(0, hop))  # random hop phase relative to the capture
    a = snr_amplitude(snr_db, p.ctrl_bw)
    max_f = FS / 2 - p.ctrl_bw
    for start in range(t0, n, hop):
        s, e = max(start, 0), min(start + burst, n)
        if e <= s:
            continue
        if p.ctrl_mod == "gfsk":
            h = p.ctrl_bw / p.ctrl_sym_rate - 1.0  # Carson-style: BW ~ (h + 1) * Rs
            w = gfsk(e - s, p.ctrl_sym_rate, max(h, 0.3), rng)
        else:
            w = chirp(e - s, p.ctrl_bw, p.ctrl_sym_rate, rng)
        buf[s:e] += a * shift(w, rng.uniform(-max_f, max_f), s)
    if p.video_bw is not None:
        fc = rng.uniform(-(FS - p.video_bw) / 2, (FS - p.video_bw) / 2)
        v = bandlimited_noise(n, p.video_bw, rng)
        buf += snr_amplitude(snr_db + p.video_snr_offset_db, p.video_bw) * shift(v, fc)


def add_wifi(buf: np.ndarray, snr_db: float, rng: np.random.Generator, bw: float = 8e6) -> None:
    n = len(buf)
    a = snr_amplitude(snr_db, bw)
    fc = rng.uniform(-(FS - bw) / 2, (FS - bw) / 2)
    pos = -int(rng.uniform(0, 1.5e-3) * FS)
    while pos < n:
        length = int(rng.uniform(0.1e-3, 1.5e-3) * FS)
        s, e = max(pos, 0), min(pos + length, n)
        if e > s:
            buf[s:e] += a * shift(bandlimited_noise(e - s, bw, rng), fc, s)
        pos += length + int(rng.exponential(0.6e-3) * FS)


def add_bluetooth(buf: np.ndarray, snr_db: float, rng: np.random.Generator, occupancy: float = 0.6):
    n = len(buf)
    slot, pkt, bw = int(625e-6 * FS), int(366e-6 * FS), 1e6
    a = snr_amplitude(snr_db, bw)
    for start in range(-int(rng.integers(0, slot)), n, slot):
        if rng.random() > occupancy:
            continue
        s, e = max(start, 0), min(start + pkt, n)
        if e > s:
            w = gfsk(e - s, 1e6, 0.32, rng)
            buf[s:e] += a * shift(w, rng.uniform(-FS / 2 + bw, FS / 2 - bw), s)


# ----- scenes ---------------------------------------------------------------------------------


@dataclass
class SceneSpec:
    drone: str | None = None  # profile key, or None for no drone
    drone_snr_db: float = 10.0
    interferers: tuple[str, ...] = ()  # subset of ("wifi", "bt")
    interferer_snr_db: dict[str, float] = field(default_factory=dict)


def make_scene(spec: SceneSpec, rng: np.random.Generator, n: int = N_SAMPLES) -> np.ndarray:
    """Complex baseband IQ for one observation: unit-power AWGN plus the requested emitters."""
    buf = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    if spec.drone is not None:
        add_drone(buf, PROFILES[spec.drone], spec.drone_snr_db, rng)
    for name in spec.interferers:
        snr = spec.interferer_snr_db.get(name, 10.0)
        if name == "wifi":
            add_wifi(buf, snr, rng)
        elif name == "bt":
            add_bluetooth(buf, snr, rng)
        else:
            raise ValueError(f"unknown interferer {name!r}")
    return buf.astype(np.complex64)
