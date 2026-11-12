"""Jammer models for the backhaul band.

The band is split into ``n_channels`` channels, each as wide as one plain/FH channel. A jammer
describes how its received power is spread over those channels at time t: ``channel_fractions``
returns a vector that sums to 1 (all power inside the band). For a fixed total power, spreading
it thinner hits more channels at lower density; that trade-off is the core of anti-jam analysis.

These are models of interference for defensive analysis: they describe received power
distributions only and contain no signal generation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skyshield.sensors.rf_df import path_loss_db


@dataclass
class Jammer:
    position: np.ndarray
    eirp_dbm: float = 40.0
    path_loss_exp: float = 2.5
    freq_hz: float = 2.44e9
    name: str = "jammer"
    # True if channel_fractions is random for a given t (expected values then need averaging).
    stochastic = False

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)

    def received_mw(self, rx_pos: np.ndarray) -> float:
        d = float(np.linalg.norm(np.asarray(rx_pos, float) - self.position))
        return float(10 ** ((self.eirp_dbm - path_loss_db(d, self.freq_hz, self.path_loss_exp)) / 10))

    def channel_fractions(self, n_channels: int, t: float, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


@dataclass
class BarrageJammer(Jammer):
    """Noise spread evenly over the whole band."""

    name: str = "barrage"

    def channel_fractions(self, n_channels, t, rng):
        return np.full(n_channels, 1.0 / n_channels)


@dataclass
class PartialBandJammer(Jammer):
    """Noise concentrated on a random fraction rho of the channels (re-drawn every call).

    Against FH the jammer trades coverage for density; the worst case for an uncoded FH link is
    rho ~ 1 / (Eb/J0), where it jams just enough of the band to make every hit hurt.
    """

    rho: float = 0.1
    name: str = "partial-band"
    stochastic = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 < self.rho <= 1.0:
            raise ValueError("rho must be in (0, 1]")

    def n_jammed(self, n_channels: int) -> int:
        return max(1, round(self.rho * n_channels))

    def channel_fractions(self, n_channels, t, rng):
        k = self.n_jammed(n_channels)
        f = np.zeros(n_channels)
        f[rng.choice(n_channels, size=k, replace=False)] = 1.0 / k
        return f

    def all_states(self, n_channels: int) -> list[np.ndarray]:
        """Every placement of a contiguous jammed block, equally likely.

        Plain, FH and DSSS outcomes depend only on how many channels are jammed and whether the
        link's own channel is one of them, so averaging over these n_channels cyclic shifts gives
        the exact expectation over random placements.
        """
        k = self.n_jammed(n_channels)
        base = np.zeros(n_channels)
        base[:k] = 1.0 / k
        return [np.roll(base, i) for i in range(n_channels)]


@dataclass
class SpotJammer(Jammer):
    """All power on one channel (e.g. a fixed-frequency link's known channel)."""

    channel: int = 0
    name: str = "spot"

    def channel_fractions(self, n_channels, t, rng):
        f = np.zeros(n_channels)
        f[self.channel % n_channels] = 1.0
        return f


@dataclass
class SweepJammer(Jammer):
    """A narrow jamming band swept across the whole band with a fixed period."""

    width_channels: int = 4
    period_s: float = 10e-3
    name: str = "sweep"

    def channel_fractions(self, n_channels, t, rng):
        start = int((t % self.period_s) / self.period_s * n_channels)
        idx = (start + np.arange(self.width_channels)) % n_channels
        f = np.zeros(n_channels)
        f[idx] = 1.0 / self.width_channels
        return f
