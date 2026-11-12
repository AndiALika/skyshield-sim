"""Baseband Monte Carlo checks for the analytic link models in ``modulation`` / ``waveforms``.

- ``bpsk_ber``: BPSK over AWGN, symbol level.
- ``dsss_tone_ber``: DSSS at chip level (random +-1 spreading code, Gp chips per bit) with a CW
  tone jammer. The analytic model treats the despread tone as Gaussian noise of power J/Gp; for a
  tone that is only approximately true, so this is the check that matters.
- ``fh_partial_band_per``: slow FH (one packet per hop) against partial-band noise, packet level.

Conventions: chip/symbol energy normalised to 1, noise is complex AWGN, decisions are coherent
on the real part.
"""

from __future__ import annotations

import numpy as np

_CHUNK = 2_000_000  # max complex samples per batch


def _awgn(shape, n0: float, rng: np.random.Generator) -> np.ndarray:
    return np.sqrt(n0 / 2) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def bpsk_ber(ebn0_db: float, n_bits: int, rng: np.random.Generator) -> float:
    n0 = 10 ** (-ebn0_db / 10)
    bits = rng.integers(0, 2, n_bits) * 2 - 1
    r = bits + _awgn(n_bits, n0, rng)
    return float(np.mean(np.sign(r.real) != bits))


def dsss_tone_ber(
    ebn0_db: float,
    js_db: float,
    gp: int,
    n_bits: int,
    rng: np.random.Generator,
    tone_offset: float = 0.01,
) -> float:
    """BER of DSSS-BPSK with a CW tone jammer.

    js_db: jammer-to-signal power ratio at the receiver input (both measured over the spread band).
    tone_offset: tone frequency offset from the carrier in cycles per chip.
    """
    ec = 1.0
    eb = gp * ec
    n0 = eb / 10 ** (ebn0_db / 10)
    a_j = np.sqrt(10 ** (js_db / 10) * ec)
    errors, done = 0, 0
    per_chunk = max(1, _CHUNK // gp)
    while done < n_bits:
        nb = min(per_chunk, n_bits - done)
        bits = rng.integers(0, 2, nb) * 2 - 1
        code = rng.integers(0, 2, (nb, gp)) * 2 - 1  # long random spreading code
        k = (done * gp + np.arange(nb * gp)).reshape(nb, gp)
        phase0 = rng.uniform(0, 2 * np.pi)
        tone = a_j * np.exp(1j * (2 * np.pi * tone_offset * k + phase0))
        r = bits[:, None] * code * np.sqrt(ec) + tone + _awgn((nb, gp), n0, rng)
        z = np.sum(r * code, axis=1).real
        errors += int(np.sum(np.sign(z) != bits))
        done += nb
    return errors / n_bits


def fh_partial_band_per(
    ebn0_db: float,
    ebj0_db: float,
    rho: float,
    n_channels: int,
    packet_bits: int,
    n_packets: int,
    rng: np.random.Generator,
) -> float:
    """Uncoded PER of slow FH under partial-band noise.

    ebj0_db: Eb/J0 with J0 = total jammer power / total band (the barrage-equivalent density).
    The jammer covers k = round(rho * n_channels) channels with density J0 * n_channels / k.
    """
    k = max(1, round(rho * n_channels))
    n0 = 10 ** (-ebn0_db / 10)
    j0_hit = 10 ** (-ebj0_db / 10) * n_channels / k
    hit = rng.random(n_packets) < k / n_channels
    var = n0 + hit * j0_hit
    errors = 0
    per_chunk = max(1, _CHUNK // packet_bits)
    for s in range(0, n_packets, per_chunk):
        v = var[s : s + per_chunk, None]
        bits = rng.integers(0, 2, (len(v), packet_bits)) * 2 - 1
        noise = np.sqrt(v / 2) * rng.standard_normal((len(v), packet_bits))
        errors += int(np.sum(np.any(np.sign(bits + noise) != bits, axis=1)))
    return errors / n_packets
