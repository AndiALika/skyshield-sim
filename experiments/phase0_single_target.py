"""Phase 0: single drone, single radar, EKF tracking.

Produces results/phase0_tracking.png and prints RMSE and mean NEES.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from skyshield.evaluation.metrics import nees, position_rmse
from skyshield.scenario.trajectories import waypoint_trajectory
from skyshield.sensors.radar import Radar
from skyshield.tracking.ekf import EKF

SEED = 42
DT = 1.0  # radar scan period (s)


def run(seed: int = SEED, pd: float = 0.9, make_plot: bool = True) -> dict:
    rng = np.random.default_rng(seed)

    truth = waypoint_trajectory(
        waypoints=np.array([[-2000.0, 1500.0], [-200.0, 800.0], [300.0, -600.0], [1500.0, -1200.0]]),
        speed=15.0,
        dt=DT,
        accel_noise=1.0,
        rng=rng,
    )
    radar = Radar(position=np.array([0.0, 0.0]), pd=pd, clutter_rate=0.0)

    track: EKF | None = None
    est, nees_vals, meas_xy = [], [], []
    for k, state in enumerate(truth):
        z_list = radar.measure(state[:2], rng)
        if track is None:
            if z_list:
                track = EKF.from_measurement(z_list[0], radar.position, radar.R, q=1.0)
            else:
                est.append(np.full(4, np.nan))
                continue
        else:
            track.predict(DT)
            if z_list:
                track.update(z_list[0], radar.position, radar.R)
        if z_list:
            r, b = z_list[0]
            meas_xy.append(radar.position + r * np.array([np.cos(b), np.sin(b)]))
        est.append(track.x.copy())
        nees_vals.append(nees(track.x, state, track.P))

    est = np.array(est)
    valid = ~np.isnan(est[:, 0])
    rmse = position_rmse(est[valid], truth[valid])
    mean_nees = float(np.mean(nees_vals))

    if make_plot:
        out = Path(__file__).resolve().parents[1] / "results"
        out.mkdir(exist_ok=True)
        meas = np.array(meas_xy)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(truth[:, 0], truth[:, 1], "k-", label="truth")
        ax.scatter(meas[:, 0], meas[:, 1], s=8, c="tab:gray", alpha=0.5, label="radar measurements")
        ax.plot(est[valid, 0], est[valid, 1], "tab:red", label="EKF estimate")
        ax.plot(*radar.position, "b^", ms=10, label="radar")
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"Phase 0: EKF tracking (RMSE {rmse:.1f} m, mean NEES {mean_nees:.2f})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "phase0_tracking.png", dpi=150)

    return {"rmse": rmse, "mean_nees": mean_nees, "steps": len(truth)}


if __name__ == "__main__":
    res = run()
    print(f"steps={res['steps']}  position RMSE={res['rmse']:.2f} m  mean NEES={res['mean_nees']:.2f} (ideal ~4)")
