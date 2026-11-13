"""Passive RF direction-finding (DF) sensor: detects a drone's control link and reports a bearing.

Chain per scan and target:
    EIRP -> log-distance path loss (+ optional shadowing) -> in-band SNR
         -> P(detect) from the RF detector's measured Pd(SNR) curve -> noisy bearing

The Pd(SNR) curve is not assumed: it is the curve the CNN actually achieves at a fixed false-alarm
rate (experiments/phase2_rf_detection.py writes it to results/phase2_metrics.json). Bearing noise
follows the usual DF behaviour, sigma ~ 1/sqrt(SNR), clipped to a floor set by array calibration.

The sensor sees no range. Bearings use the same convention as the radar: radians from the +x axis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPEED_OF_LIGHT = 299_792_458.0
BOLTZMANN_DBM_HZ = -174.0  # kT at 290 K, dBm/Hz


@dataclass(frozen=True)
class DetectionCurve:
    """Pd as a function of in-band SNR, linearly interpolated and clamped at the ends."""

    snr_db: np.ndarray
    pd: np.ndarray
    pfa: float = 0.01  # false-alarm probability per observation the curve was measured at

    def __post_init__(self) -> None:
        s, p = np.asarray(self.snr_db, float), np.asarray(self.pd, float)
        if s.ndim != 1 or s.shape != p.shape or len(s) < 2 or np.any(np.diff(s) <= 0):
            raise ValueError("snr_db must be strictly increasing and match pd")
        if np.any((p < 0) | (p > 1)):
            raise ValueError("pd must lie in [0, 1]")
        object.__setattr__(self, "snr_db", s)
        object.__setattr__(self, "pd", p)

    def __call__(self, snr_db: float | np.ndarray) -> np.ndarray:
        return np.interp(snr_db, self.snr_db, self.pd)

    @classmethod
    def from_metrics(cls, path: Path, condition: str = "mixed") -> DetectionCurve:
        m = json.loads(Path(path).read_text())
        c = m["cnn"]["pd_vs_snr"]
        return cls(np.array(m["snr_grid_db"]), np.array(c[condition]), pfa=m["pfa"])


def path_loss_db(d: float | np.ndarray, freq_hz: float, exponent: float = 2.0) -> np.ndarray:
    """Log-distance path loss, anchored to free space at 1 m. exponent = 2 gives free space."""
    d = np.maximum(np.asarray(d, float), 1.0)
    fspl_1m = 20 * np.log10(4 * np.pi * freq_hz / SPEED_OF_LIGHT)
    return fspl_1m + 10 * exponent * np.log10(d)


@dataclass
class RFDetection:
    bearing: float  # rad
    sigma: float  # bearing std (rad) for this report
    snr_db: float  # true in-band SNR (NaN for a false alarm)
    origin: int  # target index, -1 for a false alarm


@dataclass
class RFDFSensor:
    position: np.ndarray
    detection_curve: DetectionCurve
    freq_hz: float = 2.44e9
    bandwidth_hz: float = 1e6  # in-band noise bandwidth of the control link
    noise_figure_db: float = 6.0
    rx_gain_dbi: float = 3.0
    path_loss_exp: float = 2.5  # near-ground, partially obstructed
    shadowing_db: float = 0.0  # log-normal shadowing std, drawn per scan and target
    sigma_b_ref: float = np.deg2rad(2.0)  # bearing std at snr_ref_db
    snr_ref_db: float = 10.0
    sigma_b_min: float = np.deg2rad(1.0)
    sigma_b_max: float = np.deg2rad(15.0)
    false_alarm_rate: float | None = None  # expected false bearings per scan; default: curve pfa

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)
        if self.false_alarm_rate is None:
            self.false_alarm_rate = self.detection_curve.pfa

    @property
    def noise_dbm(self) -> float:
        return BOLTZMANN_DBM_HZ + 10 * np.log10(self.bandwidth_hz) + self.noise_figure_db

    def snr_db(self, target_xy: np.ndarray, eirp_dbm: float) -> np.ndarray:
        """Mean in-band SNR (no shadowing) for target(s) at target_xy, shape (..., 2)."""
        d = np.linalg.norm(np.asarray(target_xy, float) - self.position, axis=-1)
        p_rx = eirp_dbm + self.rx_gain_dbi - path_loss_db(d, self.freq_hz, self.path_loss_exp)
        return p_rx - self.noise_dbm

    def bearing_sigma(self, snr_db: float) -> float:
        s = self.sigma_b_ref * 10 ** ((self.snr_ref_db - snr_db) / 20)  # 1/sqrt(SNR)
        return float(np.clip(s, self.sigma_b_min, self.sigma_b_max))

    def measure_multi(
        self,
        targets_xy: np.ndarray,
        eirp_dbm: float | np.ndarray,
        rng: np.random.Generator,
        channel_interference_mw: np.ndarray | None = None,
    ) -> list[RFDetection]:
        """One scan. Each target is detected with Pd(SNR); false bearings arrive as Poisson.

        channel_interference_mw: received interference (e.g. a jammer) in each of the band's
        channels, each ``bandwidth_hz`` wide. The drone's control link hops, so each target is
        observed on one uniformly drawn channel and that channel's interference adds to the
        noise floor. None means no interference and draws no channel.
        """
        targets_xy = np.asarray(targets_xy, dtype=float).reshape(-1, 2)
        eirp = np.broadcast_to(np.asarray(eirp_dbm, float), (len(targets_xy),))
        out: list[RFDetection] = []
        noise_mw = 10 ** (self.noise_dbm / 10)
        for i, xy in enumerate(targets_xy):
            snr = float(self.snr_db(xy, eirp[i]))
            if channel_interference_mw is not None:
                ch = rng.integers(len(channel_interference_mw))
                snr -= 10 * np.log10(1 + channel_interference_mw[ch] / noise_mw)
            if self.shadowing_db > 0:
                snr += self.shadowing_db * rng.standard_normal()
            if rng.random() < float(self.detection_curve(snr)):
                d = xy - self.position
                sig = self.bearing_sigma(snr)
                b = np.arctan2(d[1], d[0]) + sig * rng.standard_normal()
                out.append(RFDetection(float((b + np.pi) % (2 * np.pi) - np.pi), sig, snr, i))
        for _ in range(rng.poisson(self.false_alarm_rate)):
            out.append(RFDetection(float(rng.uniform(-np.pi, np.pi)), self.sigma_b_max, np.nan, -1))
        return out
