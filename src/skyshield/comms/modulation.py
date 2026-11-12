"""Bit and packet error models.

All SINRs here are per information bit: Eb / (N0 + J0_eff), linear. Jamming that survives the
receiver (after despreading, or in a hit FH channel) is treated as additional Gaussian noise;
``skyshield.comms.simulate`` checks that approximation with baseband Monte Carlo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import erfc
from scipy.stats import binom


def qfunc(x: float | np.ndarray) -> np.ndarray:
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2.0))


def ber_bpsk(sinr: float | np.ndarray) -> np.ndarray:
    """Coherent BPSK (and Gray-coded QPSK per bit): Pb = Q(sqrt(2 Eb/N0))."""
    return qfunc(np.sqrt(2.0 * np.maximum(np.asarray(sinr, dtype=float), 0.0)))


def db2lin(x: float | np.ndarray) -> np.ndarray:
    return 10.0 ** (np.asarray(x, dtype=float) / 10.0)


def lin2db(x: float | np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.asarray(x, dtype=float))


@dataclass(frozen=True)
class BlockCode:
    """(n, k) block code that corrects any pattern of up to t bit errors (bounded-distance
    decoding). Decoding failures beyond t are counted as codeword errors."""

    n: int
    k: int
    t: int

    def __post_init__(self) -> None:
        if not (1 <= self.k <= self.n and 0 <= self.t < self.n):
            raise ValueError("need 1 <= k <= n and 0 <= t < n")

    @property
    def rate(self) -> float:
        return self.k / self.n

    def codeword_error(self, ber: float | np.ndarray) -> np.ndarray:
        """P(more than t of the n coded bits are wrong)."""
        return binom.sf(self.t, self.n, np.asarray(ber, dtype=float))


UNCODED = BlockCode(1, 1, 0)
HAMMING_7_4 = BlockCode(7, 4, 1)
BCH_63_45 = BlockCode(63, 45, 3)


@dataclass(frozen=True)
class PacketSpec:
    info_bits: int = 512  # one sensor report: a few measurements plus header
    code: BlockCode = UNCODED

    @property
    def n_codewords(self) -> int:
        return math.ceil(self.info_bits / self.code.k)

    @property
    def coded_bits(self) -> int:
        return self.n_codewords * self.code.n

    def per(self, sinr: float | np.ndarray) -> np.ndarray:
        """Packet error rate for a per-information-bit SINR (linear).

        Coded bits carry energy R * Eb, so their BER is evaluated at R * SINR.
        """
        p = ber_bpsk(self.code.rate * np.asarray(sinr, dtype=float))
        p_cw = self.code.codeword_error(p)
        return 1.0 - (1.0 - p_cw) ** self.n_codewords
