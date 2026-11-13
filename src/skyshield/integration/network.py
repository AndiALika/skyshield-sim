"""Reference network used by the Phase 4 experiment, the README GIFs and the dashboard.

Three radars with backhaul links of 1.1 / 2.3 / 3.2 km to a fusion centre at the origin, and a
ground jammer 3.5 km away. The links are deliberately unequal: with equal links every link
failed at the same jammer power and the resilience curve was a single step (see CLAUDE.md,
"Integration design").
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from skyshield.comms.jammers import BarrageJammer, Jammer, PartialBandJammer, SpotJammer
from skyshield.comms.link import BackhaulLink, BernoulliLink
from skyshield.comms.waveforms import DSSS, FrequencyHopping, Plain
from skyshield.integration.closed_loop import (
    ClosedLoopConfig,
    ClosedLoopSim,
    DFNode,
    JammerSchedule,
    SensorNode,
)
from skyshield.scenario.multi_target import crossing_scenario
from skyshield.sensors.radar import Radar
from skyshield.sensors.rf_df import DetectionCurve, RFDFSensor

FUSION = np.array([0.0, 0.0])
RADARS = [np.array([-1000.0, -400.0]), np.array([2200.0, -800.0]), np.array([-400.0, 3200.0])]
JAMMER_POS = np.array([-2500.0, 2500.0])
BACKHAUL_PL_EXP = 2.0  # masts, line of sight
PD, CLUTTER = 0.9, 5.0
# Passive RF DF sites close to the fusion centre (links of 0.7-0.8 km), spread out so their
# bearings cross at useful angles over the area the drones fly in.
DF_SITES = [np.array([700.0, 400.0]), np.array([-600.0, 500.0]), np.array([100.0, -700.0])]
DF_ID0 = 10  # DF node ids are DF_ID0, DF_ID0 + 1, ...
PHASE2_METRICS = Path(__file__).resolve().parents[3] / "results" / "phase2_metrics.json"

WAVEFORMS: dict[str, Callable] = {
    "plain": lambda: Plain(channel=0),
    "FH": FrequencyHopping,
    "DSSS": DSSS,
}
JAMMERS: dict[str, Callable[[float], Jammer]] = {
    "spot": lambda e: SpotJammer(JAMMER_POS, eirp_dbm=e, channel=0),
    "partial-band 0.1": lambda e: PartialBandJammer(JAMMER_POS, eirp_dbm=e, rho=0.1),
    "partial-band 0.5": lambda e: PartialBandJammer(JAMMER_POS, eirp_dbm=e, rho=0.5),
    "barrage": lambda e: BarrageJammer(JAMMER_POS, eirp_dbm=e),
}


def build_sim(
    waveform: str,
    jammer: str | None,
    eirp_dbm: float,
    window: tuple[float, float] = (0.0, np.inf),
    *,
    scenario_seed: int = 2024,
    pd: float = PD,
    clutter: float = CLUTTER,
    config: ClosedLoopConfig | None = None,
    sensors: str = "radar",
) -> ClosedLoopSim:
    """Closed loop on the reference network.

    waveform: a key of ``WAVEFORMS``, or ``"iid:<pdr>"`` for a ``BernoulliLink`` on every radar.
    jammer: a key of ``JAMMERS`` or None; it is on during ``window`` (seconds).
    sensors: "radar", "radar+df" or "df". DF nodes use the same backhaul waveform as the radars
    and the CNN's measured Pd(SNR) curve ("mixed" environment) from Phase 2.
    """
    if sensors not in ("radar", "radar+df", "df"):
        raise ValueError(f"unknown sensors: {sensors}")
    sc = crossing_scenario(rng=np.random.default_rng(scenario_seed))
    if waveform.startswith("iid:"):
        def make_link(p):
            return BernoulliLink(float(waveform[4:]))
    else:
        def make_link(p):
            return BackhaulLink(p, FUSION, waveform=WAVEFORMS[waveform](),
                                path_loss_exp=BACKHAUL_PL_EXP)
    nodes = [SensorNode(i, Radar(position=p, pd=pd, clutter_rate=clutter), make_link(p))
             for i, p in enumerate(RADARS)] if sensors != "df" else []
    df_nodes = []
    if sensors != "radar":
        curve = DetectionCurve.from_metrics(PHASE2_METRICS, "mixed")
        df_nodes = [DFNode(DF_ID0 + i, RFDFSensor(p, curve), make_link(p))
                    for i, p in enumerate(DF_SITES)]
    jams = [] if jammer is None else [JAMMERS[jammer](eirp_dbm)]
    return ClosedLoopSim(sc, nodes, JammerSchedule(jams, *window), config, df_nodes)
