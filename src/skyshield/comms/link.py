"""Sensor-to-fusion backhaul link: link budget, jamming, packet errors, ARQ, delivery delay.

Per packet the jammer's channel occupancy is drawn once and held for every retransmission of
that packet (a slow jammer). An FH link redraws its channel on each attempt, a plain or DSSS link
does not. So retries help FH against partial-band jamming, but not a plain link that is being
spot-jammed: every retry lands in the same jammed channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from skyshield.comms.jammers import Jammer
from skyshield.comms.modulation import PacketSpec
from skyshield.comms.waveforms import Plain
from skyshield.sensors.rf_df import BOLTZMANN_DBM_HZ, path_loss_db


@dataclass
class Delivery:
    delivered: bool
    attempts: int
    delay_s: float  # time from first transmission to arrival (inf if dropped)


@dataclass
class BackhaulLink:
    tx_pos: np.ndarray
    rx_pos: np.ndarray
    waveform: object = field(default_factory=Plain)
    tx_power_dbm: float = 20.0
    freq_hz: float = 2.44e9
    bit_rate: float = 1e6
    n_channels: int = 64
    noise_figure_db: float = 6.0
    path_loss_exp: float = 2.5
    packet: PacketSpec = field(default_factory=PacketSpec)
    max_retries: int = 3
    ack_overhead_s: float = 0.5e-3  # turnaround + ACK per attempt
    base_latency_s: float = 1e-3  # processing / queuing, paid once

    def __post_init__(self) -> None:
        self.tx_pos = np.asarray(self.tx_pos, dtype=float)
        self.rx_pos = np.asarray(self.rx_pos, dtype=float)

    # ----- budget ------------------------------------------------------------------------------

    @property
    def distance_m(self) -> float:
        return float(np.linalg.norm(self.tx_pos - self.rx_pos))

    @property
    def signal_mw(self) -> float:
        pl = path_loss_db(self.distance_m, self.freq_hz, self.path_loss_exp)
        return float(10 ** ((self.tx_power_dbm - pl) / 10))

    @property
    def noise_mw(self) -> float:
        return float(10 ** ((BOLTZMANN_DBM_HZ + 10 * np.log10(self.bit_rate)
                             + self.noise_figure_db) / 10))

    @property
    def snr_db(self) -> float:
        return float(10 * np.log10(self.signal_mw / self.noise_mw))

    @property
    def attempt_s(self) -> float:
        return self.attempt_s_for(self.packet)

    def attempt_s_for(self, packet: PacketSpec) -> float:
        """Air time of one attempt plus ACK turnaround."""
        return packet.coded_bits / self.bit_rate + self.ack_overhead_s

    def jammer_channels_mw(self, jammers: list[Jammer], t: float, rng) -> np.ndarray:
        j = np.zeros(self.n_channels)
        for jam in jammers:
            j += jam.received_mw(self.rx_pos) * jam.channel_fractions(self.n_channels, t, rng)
        return j

    def js_db(self, jammers: list[Jammer]) -> float:
        """Total received jammer power over signal power (dB)."""
        j = sum(jam.received_mw(self.rx_pos) for jam in jammers)
        return float(10 * np.log10(j / self.signal_mw)) if j > 0 else -np.inf

    # ----- packet level ------------------------------------------------------------------------

    def attempt_pers(self, j_ch: np.ndarray, packet: PacketSpec | None = None) -> np.ndarray:
        """Packet error rate for each equally likely attempt outcome under one jammer state."""
        packet = packet or self.packet
        i = self.waveform.interference_options(j_ch)
        return packet.per(self.signal_mw / (self.noise_mw + i))

    def send(
        self,
        t: float,
        jammers: list[Jammer],
        rng: np.random.Generator,
        packet: PacketSpec | None = None,
    ) -> Delivery:
        """Simulate one packet with stop-and-wait ARQ. ``packet`` overrides the link default
        (sensor reports grow with the number of detections they carry)."""
        packet = packet or self.packet
        j_ch = self.jammer_channels_mw(jammers, t, rng)
        pers = self.attempt_pers(j_ch, packet)
        for a in range(1, self.max_retries + 2):
            per = pers[rng.integers(len(pers))] if len(pers) > 1 else pers[0]
            if rng.random() >= per:
                return Delivery(True, a, self.base_latency_s + a * self.attempt_s_for(packet))
        return Delivery(False, self.max_retries + 1, np.inf)

    def expected_pdr(
        self, jammers: list[Jammer], rng: np.random.Generator, t: float = 0.0, n_states: int = 200
    ) -> float:
        """Packet delivery ratio with ARQ, averaged over jammer states (exact over FH hops).

        PDR = 1 - E_state[ (E_hop[PER | state])^(retries+1) ]. Hops are independent per attempt;
        the jammer state is shared by all attempts of one packet.
        """
        exponent = self.max_retries + 1
        stochastic = [j for j in jammers if j.stochastic]
        if len(stochastic) == 1 and hasattr(stochastic[0], "all_states"):
            # Exact: enumerate the random jammer's placements, add the deterministic ones.
            jam = stochastic[0]
            fixed = self.jammer_channels_mw([j for j in jammers if j is not jam], t, rng)
            p_rx = jam.received_mw(self.rx_pos)
            fails = [float(np.mean(self.attempt_pers(fixed + p_rx * f))) ** exponent
                     for f in jam.all_states(self.n_channels)]
            return 1.0 - float(np.mean(fails))
        n = n_states if stochastic else 1
        fail = 0.0
        for _ in range(n):
            p = float(np.mean(self.attempt_pers(self.jammer_channels_mw(jammers, t, rng))))
            fail += p**exponent
        return 1.0 - fail / n


@dataclass
class BernoulliLink:
    """Simplified link: each packet independently delivered with probability ``pdr`` after a
    fixed latency. Used to measure tracking quality as a function of delivery ratio alone,
    without any particular jammer or waveform behind it."""

    pdr: float
    latency_s: float = 2e-3

    def __post_init__(self) -> None:
        if not 0.0 <= self.pdr <= 1.0:
            raise ValueError("pdr must be in [0, 1]")

    def send(self, t, jammers, rng: np.random.Generator, packet=None) -> Delivery:
        if rng.random() < self.pdr:
            return Delivery(True, 1, self.latency_s)
        return Delivery(False, 1, np.inf)


class DelayLine:
    """Delivers payloads at their arrival time. The fusion centre drains it each tracker step."""

    def __init__(self) -> None:
        self._items: list[tuple[float, int, object]] = []
        self._seq = 0

    def push(self, payload: object, arrival_s: float) -> None:
        if np.isfinite(arrival_s):
            self._items.append((arrival_s, self._seq, payload))
            self._seq += 1

    def pop_ready(self, t: float) -> list[object]:
        ready = sorted(x for x in self._items if x[0] <= t)
        self._items = [x for x in self._items if x[0] > t]
        return [p for _, _, p in ready]

    def __len__(self) -> int:
        return len(self._items)
