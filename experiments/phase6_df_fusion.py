"""Phase 6: passive RF direction finding (DF) in the closed loop.

Three DF sites near the fusion centre (backhaul links of 0.7-0.8 km, same radio as the radars)
report bearings to the drones' control links (20 dBm EIRP). Detection uses the CNN's measured
Pd(SNR) curve from Phase 2 ("mixed" environment) and bearing noise ~ 1/sqrt(SNR). The backhaul
jammer shares the 2.4 GHz band, so while it is on it also raises the DF receivers' noise floor.

Question: when the jammer cuts the radar backhaul, how much of the track picture can passive DF
on short links keep alive, and what does DF cost or add when the radars are fine?

Experiments
-----------
1. Sweep: OSPA vs jammer EIRP for sensors = radar / radar+df / df, every waveform and jammer
   shape (same grid, seeds and common random numbers as Phase 4; the radar rows reproduce it).
2. Resilience: worst case over jammer shapes, per waveform and sensor set.
3. Ablations on the radar+df network:
   a. "bearings always count": DF hits count everywhere, not only where no radar is heard.
   b. "DF immune to the jammer": the jammer does not raise the DF noise floor.
4. Burst replay GIF: plain backhaul, spot jammer 46 dBm on 100-200 s, radar vs radar+df.

Outputs in results/: phase6_resilience.png, phase6_ablation.png, phase6_burst_replay.gif,
phase6_metrics.json. Runtime: ~25 min on 16 cores (5058 runs).
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from skyshield.dashboard.replay import save_gif
from skyshield.integration.closed_loop import ClosedLoopConfig
from skyshield.integration.network import (
    DF_SITES,
    FUSION,
    JAMMER_POS,
    JAMMERS,
    RADARS,
    WAVEFORMS,
    build_sim,
)
from skyshield.tracking.manager import TrackerConfig

SEED = 2024  # same as Phase 4
N_MC = 6
EIRP_GRID = np.arange(30.0, 70.1, 2.0)
SENSORS = ("radar", "radar+df", "df")
ABLATIONS = {
    "bearings always count": {"pairs": [("plain", "spot"), ("FH", "barrage")],
                              "config": {"tracker": {"df_hits_in_radar_coverage": True}}},
    "DF immune to jammer": {"pairs": [("FH", "barrage"), ("DSSS", "barrage")],
                            "config": {"df_jamming": False}},
}
BURST = {"t_on": 100.0, "t_off": 200.0, "eirp_dbm": 46.0, "jammer": "spot", "waveform": "plain"}
OUT = Path(__file__).resolve().parents[1] / "results"
COLORS = {"plain": "tab:gray", "FH": "tab:blue", "DSSS": "tab:orange"}
LS = {"radar": "--", "radar+df": "-", "df": ":"}


def make_config(variant: str | None) -> ClosedLoopConfig | None:
    if variant is None:
        return None
    c = ABLATIONS[variant]["config"]
    return ClosedLoopConfig(tracker=TrackerConfig(**c.get("tracker", {})),
                            df_jamming=c.get("df_jamming", True))


def one_run(args) -> dict:
    sensors, waveform, jammer, eirp, mc, variant = args
    sim = build_sim(waveform, jammer, eirp, scenario_seed=SEED, sensors=sensors,
                    config=make_config(variant))
    log = sim.run(np.random.default_rng([SEED, mc, 1]), np.random.default_rng([SEED, mc, 2]))
    s = log.summary()
    s.update(sensors=sensors, waveform=waveform, jammer=jammer, eirp_dbm=eirp, mc=mc,
             variant=variant, mean_n_est_minus_true=float(np.mean(log.n_est - log.n_true)))
    return s


def cell(rows, **key) -> dict:
    out = {"ospa": [], "ospa_ci95": [], "card_ok": [], "pdr": []}
    for e in EIRP_GRID:
        r = [x for x in rows if x["eirp_dbm"] == e and all(x[k] == v for k, v in key.items())]
        o = np.array([x["mean_ospa"] for x in r])
        out["ospa"].append(float(o.mean()))
        out["ospa_ci95"].append(float(1.96 * o.std(ddof=1) / np.sqrt(len(o))))
        out["card_ok"].append(float(np.mean([x["card_correct_frac"] for x in r])))
        out["pdr"].append(float(np.mean([x["pdr"] for x in r])))
    return out


def worst_case(agg: dict, sensors: str, wf: str) -> tuple[list[float], list[str]]:
    stack = np.array([agg[sensors][wf][jm]["ospa"] for jm in JAMMERS])
    return stack.max(0).tolist(), [list(JAMMERS)[i] for i in stack.argmax(0)]


def first_above(curve, thr) -> float | None:
    idx = np.nonzero(np.asarray(curve) > thr)[0]
    return float(EIRP_GRID[idx[0]]) if len(idx) else None


# ----- figures --------------------------------------------------------------------------------


def plot_resilience(res: dict, baseline: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, wf in zip(axes, WAVEFORMS):
        for s in SENSORS:
            ax.plot(EIRP_GRID, res[s][wf]["worst_ospa"], LS[s], color=COLORS[wf], lw=2.2,
                    marker="o", ms=3, label=f"{s} (baseline {baseline[s]:.1f} m)")
        ax.axhline(100, color="0.6", lw=0.8)
        ax.set_title(f"{wf} backhaul: worst jammer shape per EIRP")
        ax.set_xlabel("jammer EIRP (dBm)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("mean OSPA (m), c = 100 m")
    axes[0].set_ylim(0, 108)
    fig.suptitle(f"Adding 3 passive DF sites on short links ({N_MC} Monte Carlo runs per point)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ablation(agg: dict, abl: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (name, spec) in zip(axes, ABLATIONS.items()):
        for wf, jm in spec["pairs"]:
            ax.plot(EIRP_GRID, agg["radar+df"][wf][jm]["ospa"], "-", color=COLORS[wf], lw=2,
                    label=f"radar+df, {wf} / {jm}")
            ax.plot(EIRP_GRID, abl[name][f"{wf}|{jm}"]["ospa"], ":", color=COLORS[wf], lw=2,
                    label=f"{name}, {wf} / {jm}")
            ax.plot(EIRP_GRID, agg["radar"][wf][jm]["ospa"], "--", color=COLORS[wf], lw=1,
                    alpha=0.6, label=f"radar only, {wf} / {jm}")
        ax.set_title(f"Ablation: {name}")
        ax.set_xlabel("jammer EIRP (dBm)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("mean OSPA (m)")
    axes[0].set_ylim(0, 108)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_burst() -> dict:
    window = (BURST["t_on"], BURST["t_off"])
    burst_logs = {}
    for s in ("radar", "radar+df"):
        sim = build_sim(BURST["waveform"], BURST["jammer"], BURST["eirp_dbm"], window,
                        scenario_seed=SEED, sensors=s)
        burst_logs[s] = sim.run(np.random.default_rng([SEED, 0, 1]),
                                np.random.default_rng([SEED, 0, 2]), record=True)
    save_gif(burst_logs, OUT / "phase6_burst_replay.gif", RADARS, FUSION, JAMMER_POS,
             steps=range(85, 216), fps=10, jam_window=window, df_sites=DF_SITES,
             title=f"plain backhaul, {BURST['jammer']} jammer {BURST['eirp_dbm']:g} dBm on "
                   "100-200 s: radar only vs radar + 3 passive DF sites")
    return burst_logs


def main() -> dict:
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    jobs = [(s, wf, jm, float(e), mc, None) for s in SENSORS for wf in WAVEFORMS
            for jm in JAMMERS for e in EIRP_GRID for mc in range(N_MC)]
    jobs += [(s, "plain", None, 0.0, mc, None) for s in SENSORS for mc in range(N_MC)]
    jobs += [("radar+df", wf, jm, float(e), mc, name) for name, spec in ABLATIONS.items()
             for wf, jm in spec["pairs"] for e in EIRP_GRID for mc in range(N_MC)]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        rows = list(pool.map(one_run, jobs, chunksize=4))
    print(f"{len(rows)} runs in {time.time() - t0:.0f} s", flush=True)

    main_rows = [r for r in rows if r["variant"] is None and r["jammer"] is not None]
    agg = {s: {wf: {jm: cell(main_rows, sensors=s, waveform=wf, jammer=jm) for jm in JAMMERS}
               for wf in WAVEFORMS} for s in SENSORS}
    base_rows = [r for r in rows if r["jammer"] is None]
    baseline = {s: float(np.mean([r["mean_ospa"] for r in base_rows if r["sensors"] == s]))
                for s in SENSORS}
    baseline_card = {s: float(np.mean([r["card_correct_frac"] for r in base_rows
                                       if r["sensors"] == s])) for s in SENSORS}
    res = {}
    for s in SENSORS:
        res[s] = {}
        for wf in WAVEFORMS:
            worst, which = worst_case(agg, s, wf)
            res[s][wf] = {"worst_ospa": worst, "worst_jammer": which,
                          "eirp_for_2x_radar_baseline_dbm": first_above(worst,
                                                                        2 * baseline["radar"]),
                          "eirp_blind_dbm": first_above(worst, 95.0)}
    abl = {name: {f"{wf}|{jm}": cell([r for r in rows if r["variant"] == name],
                                     waveform=wf, jammer=jm)
                  for wf, jm in spec["pairs"]} for name, spec in ABLATIONS.items()}
    plot_resilience(res, baseline, OUT / "phase6_resilience.png")
    plot_ablation(agg, abl, OUT / "phase6_ablation.png")

    burst_logs = render_burst()
    on = slice(int(BURST["t_on"]), int(BURST["t_off"]))

    metrics = {
        "seed": SEED, "n_mc": N_MC, "eirp_grid_dbm": EIRP_GRID.tolist(),
        "df_sites": [p.tolist() for p in DF_SITES],
        "baseline_ospa": baseline, "baseline_card_correct": baseline_card,
        "sweep": agg, "resilience": res, "ablations": abl,
        "burst": {**BURST, **{s: {"mean_ospa": lg.mean_ospa,
                                  "ospa_during": float(lg.ospa[on].mean()),
                                  "card_ok_during": float(np.mean(lg.n_true[on] == lg.n_est[on]))}
                              for s, lg in burst_logs.items()}},
        "max_stale_reports": int(max(r["stale_reports"] for r in rows)),
        "runtime_s": time.time() - t0,
    }
    (OUT / "phase6_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    import sys

    if "--gif-only" in sys.argv:
        render_burst()
        raise SystemExit
    m = main()
    print(f"runtime {m['runtime_s'] / 60:.1f} min")
    print("baseline OSPA:", {k: round(v, 2) for k, v in m["baseline_ospa"].items()})
    for s in SENSORS:
        print(s, {wf: (v["eirp_for_2x_radar_baseline_dbm"], v["eirp_blind_dbm"])
                  for wf, v in m["resilience"][s].items()})
    print("burst:", json.dumps({k: m["burst"][k] for k in ("radar", "radar+df")}, indent=1))
