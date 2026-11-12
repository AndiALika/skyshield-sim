"""Link waveforms: how much of the jammer's power ends up as interference per information bit.

Common frame: a band of ``n_channels`` channels, each B = bit rate wide. Every waveform carries
the same bit rate and sees the same thermal noise N = k T B F in the information bandwidth.

- Plain: fixed channel c. Interference = jammer power in channel c.
- FH (slow): one packet attempt per hop, channel drawn uniformly. Interference = jammer power in
  the hop's channel. Averaged over hops, FH turns a concentrated jammer into occasional bad hops.
- DSSS: spread over all n_channels (processing gain Gp = n_channels). Despreading divides
  in-band jammer power by Gp, whatever its shape (Gaussian approximation; checked against a
  chip-level simulation in ``skyshield.comms.simulate``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Plain:
    channel: int = 0
    name: str = "plain"

    def interference_options(self, j_ch: np.ndarray) -> np.ndarray:
        """Possible interference powers for one attempt (equally likely)."""
        return np.array([j_ch[self.channel % len(j_ch)]])


@dataclass(frozen=True)
class FrequencyHopping:
    name: str = "FH"

    def interference_options(self, j_ch: np.ndarray) -> np.ndarray:
        return np.asarray(j_ch, dtype=float)


@dataclass(frozen=True)
class DSSS:
    name: str = "DSSS"

    def interference_options(self, j_ch: np.ndarray) -> np.ndarray:
        return np.array([np.sum(j_ch) / len(j_ch)])  # J_in / Gp with Gp = n_channels


WAVEFORMS = {"plain": Plain(), "FH": FrequencyHopping(), "DSSS": DSSS()}
