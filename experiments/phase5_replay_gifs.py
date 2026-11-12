"""Phase 5: animated replays for the README.

1. phase5_burst_replay.gif: the Phase 4 jamming burst (spot jammer, 50 dBm, on 100-200 s),
   plain vs FH backhaul side by side, same detections and clutter in both panels.
2. phase5_tracking_replay.gif: the unjammed network tracking the four drones for the whole run.

Runtime: about 2 min (most of it is GIF encoding).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from skyshield.dashboard.replay import save_gif
from skyshield.integration.network import FUSION, JAMMER_POS, RADARS, build_sim

SEED = 2024
BURST = {"t_on": 100.0, "t_off": 200.0, "eirp_dbm": 50.0, "jammer": "spot"}
OUT = Path(__file__).resolve().parents[1] / "results"


def run(waveform: str, jammer: str | None, window=(0.0, np.inf)):
    sim = build_sim(waveform, jammer, BURST["eirp_dbm"], window, scenario_seed=SEED)
    return sim.run(np.random.default_rng([SEED, 0, 1]), np.random.default_rng([SEED, 0, 2]),
                   record=True)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    window = (BURST["t_on"], BURST["t_off"])
    logs = {f"{wf} backhaul": run(wf, BURST["jammer"], window) for wf in ("plain", "FH")}
    save_gif(logs, OUT / "phase5_burst_replay.gif", RADARS, FUSION, JAMMER_POS,
             steps=range(85, 216), fps=10, jam_window=window,
             title=f"{BURST['jammer']} jammer, {BURST['eirp_dbm']:g} dBm, on 100-200 s: "
                   "red = report lost on the backhaul")
    for name, log in logs.items():
        print(f"{name}: mean OSPA {log.mean_ospa:.1f} m")

    clean = {"no jamming": run("plain", None)}
    save_gif(clean, OUT / "phase5_tracking_replay.gif", RADARS, FUSION, None,
             steps=range(0, 300, 2), fps=12,
             title="3 radars, 4 drones, 5 clutter/scan per radar: x = truth, green = track")
    print(f"no jamming: mean OSPA {clean['no jamming'].mean_ospa:.1f} m")


if __name__ == "__main__":
    main()
