"""Phase 4: closed loop. Jammer power -> backhaul packet loss -> tracking error (OSPA).

Network: three radars whose backhaul links to the fusion centre (origin) are 1.1, 2.3 and 3.2 km
long (20 dBm, mast-to-mast line of sight; SNR 27 / 20 / 18 dB). A ground jammer 3.5 km from the
fusion centre attacks the receiver, so every link sees the same jammer power but a different
signal power: links drop out one at a time as the jammer gets stronger. (A first version with
three equal-length links failed all at once, which hid the degradation in a single step.)
Scenario: the Phase 1 crossing scenario (4 drones), Pd 0.9, 5 clutter/scan per radar.

Experiments
-----------
1. Sweep: OSPA vs jammer EIRP for every (waveform, jammer shape) pair. Monte Carlo runs use
   common random numbers: run i sees the same truth, detections and clutter for every pair.
2. Resilience curve (flagship): for each waveform, the worst OSPA over jammer shapes at each
   power level, i.e. against a jammer that picks the best shape for the link it faces.
3. Jamming burst: one run with the jammer on from 100 s to 200 s, OSPA and delivery over time.
4. Coupling: OSPA vs packet delivery ratio. A simplified i.i.d.-loss link (``BernoulliLink``)
   sweeps PDR directly; the jammer-sweep runs are overlaid on the same axes.

Outputs in results/: phase4_ospa_vs_eirp.png, phase4_resilience.png, phase4_burst.png,
phase4_pdr_vs_ospa.png, phase4_metrics.json
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

from skyshield.integration.network import JAMMER_POS, JAMMERS, RADARS, WAVEFORMS, build_sim

SEED = 2024
N_MC = 6
EIRP_GRID = np.arange(30.0, 70.1, 2.0)
PD, CLUTTER = 0.9, 5.0
BURST = {"t_on": 100.0, "t_off": 200.0, "eirp_dbm": 50.0, "jammer": "spot"}
PDR_GRID = np.round(np.arange(0.0, 1.001, 0.05), 2)
OUT = Path(__file__).resolve().parents[1] / "results"

COLORS = {"plain": "tab:gray", "FH": "tab:blue", "DSSS": "tab:orange"}
LS = {"spot": "-", "partial-band 0.1": "--", "partial-band 0.5": "-.", "barrage": ":"}


def build(waveform: str, jammer: str | None, eirp: float, window=(0.0, np.inf)):
    return build_sim(waveform, jammer, eirp, window, scenario_seed=SEED, pd=PD, clutter=CLUTTER)


def one_run(args) -> dict:
    waveform, jammer, eirp, mc = args
    log = build(waveform, jammer, eirp).run(
        np.random.default_rng([SEED, mc, 1]),  # sensors: shared across all link configurations
        np.random.default_rng([SEED, mc, 2]),  # comms
    )
    s = log.summary()
    s.update(waveform=waveform, jammer=jammer, eirp_dbm=eirp, mc=mc)
    return s


def aggregate(rows: list[dict]) -> dict:
    out: dict = {}
    for wf in WAVEFORMS:
        out[wf] = {}
        for jm in JAMMERS:
            cell = {"ospa": [], "ospa_ci95": [], "pdr": [], "card_ok": [], "pdr_per_sensor": []}
            for e in EIRP_GRID:
                r = [x for x in rows if x["waveform"] == wf and x["jammer"] == jm
                     and x["eirp_dbm"] == e]
                o = np.array([x["mean_ospa"] for x in r])
                cell["ospa"].append(float(o.mean()))
                cell["ospa_ci95"].append(float(1.96 * o.std(ddof=1) / np.sqrt(len(o))))
                cell["pdr"].append(float(np.mean([x["pdr"] for x in r])))
                cell["card_ok"].append(float(np.mean([x["card_correct_frac"] for x in r])))
                cell["pdr_per_sensor"].append(np.mean([x["pdr_per_sensor"] for x in r], 0).tolist())
            out[wf][jm] = cell
    return out


def resilience(agg: dict, baseline: dict) -> dict:
    res = {}
    for wf in WAVEFORMS:
        stack = np.array([agg[wf][jm]["ospa"] for jm in JAMMERS])
        worst = stack.max(axis=0)
        which = [list(JAMMERS)[i] for i in stack.argmax(axis=0)]
        # Jammer EIRP at which OSPA doubles relative to the unjammed baseline.
        thr = 2 * baseline[wf]
        idx = np.nonzero(worst > thr)[0]
        res[wf] = {"worst_ospa": worst.tolist(), "worst_jammer": which,
                   "eirp_for_2x_ospa_dbm": float(EIRP_GRID[idx[0]]) if len(idx) else None}
    return res


# ----- figures --------------------------------------------------------------------------------


def plot_sweep(agg: dict, baseline: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for ax, wf in zip(axes, WAVEFORMS):
        for jm in JAMMERS:
            c = agg[wf][jm]
            ax.errorbar(EIRP_GRID, c["ospa"], yerr=c["ospa_ci95"], ls=LS[jm], color=COLORS[wf],
                        capsize=2, lw=1.8, label=f"{jm} jammer")
        ax.axhline(baseline[wf], color="k", lw=0.8, label="no jammer")
        ax.set_title(f"{wf} backhaul")
        ax.set_xlabel("jammer EIRP (dBm), 3.5 km from fusion centre")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("mean OSPA (m), c = 100 m")
    fig.suptitle(f"Closed loop: 3 radars, 4 drones, {N_MC} Monte Carlo runs per point (95% CI)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_resilience(res: dict, agg: dict, baseline: dict, path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for wf in WAVEFORMS:
        ax[0].plot(EIRP_GRID, res[wf]["worst_ospa"], "o-", color=COLORS[wf], lw=2.2, ms=4,
                   label=f"{wf} (worst jammer shape)")
        stack = np.array([agg[wf][jm]["pdr"] for jm in JAMMERS])
        ax[1].plot(EIRP_GRID, stack.min(axis=0), "o-", color=COLORS[wf], lw=2.2, ms=4,
                   label=wf)
    ax[0].axhline(np.mean(list(baseline.values())), color="k", lw=0.8, ls="--",
                  label="no jamming")
    ax[0].axhline(100.0, color="0.6", lw=0.8)
    ax[0].text(EIRP_GRID[0], 102, "OSPA cut-off: no track picture", fontsize=8, color="0.4")
    ax[0].set_ylabel("mean OSPA (m)")
    ax[0].set_title("Resilience: tracking error vs jammer power\n"
                    "(jammer chooses spot / partial-band / barrage per link type)")
    ax[1].set_ylabel("packet delivery ratio")
    ax[1].set_title("Backhaul delivery under the same worst-case jammer")
    for a in ax:
        a.set_xlabel("jammer EIRP (dBm)")
        a.grid(alpha=0.3)
        a.legend(fontsize=9)
    ax[0].set_ylim(0, 110)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_burst(logs: dict, path: Path) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    for wf, log in logs.items():
        t = np.arange(len(log.ospa))
        ax[0].plot(t, log.ospa, color=COLORS[wf], lw=1.4, label=f"{wf}: mean {log.mean_ospa:.1f} m")
        ax[1].step(t, log.delivered.mean(axis=1), where="post", color=COLORS[wf], lw=1.4)
    for a in ax:
        a.axvspan(BURST["t_on"], BURST["t_off"], color="tab:red", alpha=0.08)
        a.grid(alpha=0.3)
    ax[0].text(BURST["t_on"] + 2, 92, f"{BURST['jammer']} jammer on, {BURST['eirp_dbm']:g} dBm",
               color="tab:red", fontsize=9)
    ax[0].set_ylabel("OSPA (m)")
    ax[0].set_ylim(0, 105)
    ax[0].legend(fontsize=9, loc="upper right")
    ax[1].set_ylabel("reports fused\n(fraction of radars)")
    ax[1].set_xlabel("time (s)")
    fig.suptitle("Jamming burst: tracking collapses on the plain link and recovers after")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_coupling(rows: list[dict], iid: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for wf in WAVEFORMS:
        r = [x for x in rows if x["waveform"] == wf]
        ax.scatter([x["pdr"] for x in r], [x["mean_ospa"] for x in r], s=10, alpha=0.45,
                   color=COLORS[wf], label=f"jammer sweep, {wf}")
    ax.errorbar(iid["pdr"], iid["ospa"], yerr=iid["ospa_ci95"], fmt="k-", lw=1.8, capsize=2,
                label="i.i.d. packet loss (BernoulliLink)")
    ax.set_xlabel("packet delivery ratio")
    ax.set_ylabel("mean OSPA (m)")
    ax.set_title("How much delivery does the tracker need?")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> dict:
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    jobs = [(wf, jm, float(e), mc) for wf in WAVEFORMS for jm in JAMMERS for e in EIRP_GRID
            for mc in range(N_MC)]
    jobs += [(wf, None, 0.0, mc) for wf in WAVEFORMS for mc in range(N_MC)]
    jobs += [(f"iid:{p}", None, 0.0, mc) for p in PDR_GRID for mc in range(N_MC)]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        rows = list(pool.map(one_run, jobs, chunksize=4))
    print(f"{len(rows)} runs in {time.time() - t0:.0f} s", flush=True)

    sweep = [r for r in rows if r["jammer"] is not None]
    iid = {"pdr": [], "ospa": [], "ospa_ci95": [], "card_ok": []}
    for p in PDR_GRID:
        r = [x for x in rows if x["waveform"] == f"iid:{p}"]
        o = np.array([x["mean_ospa"] for x in r])
        iid["pdr"].append(float(p))
        iid["ospa"].append(float(o.mean()))
        iid["ospa_ci95"].append(float(1.96 * o.std(ddof=1) / np.sqrt(len(o))))
        iid["card_ok"].append(float(np.mean([x["card_correct_frac"] for x in r])))
    baseline = {wf: float(np.mean([r["mean_ospa"] for r in rows
                                   if r["jammer"] is None and r["waveform"] == wf]))
                for wf in WAVEFORMS}
    agg = aggregate(sweep)
    res = resilience(agg, baseline)
    plot_sweep(agg, baseline, OUT / "phase4_ospa_vs_eirp.png")
    plot_resilience(res, agg, baseline, OUT / "phase4_resilience.png")
    plot_coupling(sweep, iid, OUT / "phase4_pdr_vs_ospa.png")

    burst_logs = {}
    for wf in WAVEFORMS:
        sim = build(wf, BURST["jammer"], BURST["eirp_dbm"], (BURST["t_on"], BURST["t_off"]))
        burst_logs[wf] = sim.run(np.random.default_rng([SEED, 0, 1]),
                                 np.random.default_rng([SEED, 0, 2]))
    plot_burst(burst_logs, OUT / "phase4_burst.png")

    on = slice(int(BURST["t_on"]), int(BURST["t_off"]))
    metrics = {
        "seed": SEED, "n_mc": N_MC, "eirp_grid_dbm": EIRP_GRID.tolist(),
        "radars": [p.tolist() for p in RADARS], "jammer_pos": JAMMER_POS.tolist(),
        "pd": PD, "clutter_per_scan": CLUTTER,
        "baseline_ospa": baseline,
        "sweep": agg,
        "resilience": res,
        "ospa_vs_iid_pdr": iid,
        "burst": {**BURST, **{wf: {"mean_ospa": log.mean_ospa,
                                   "ospa_during": float(log.ospa[on].mean()),
                                   "pdr_during": float(log.delivered[on].mean()),
                                   "stale": log.stale}
                              for wf, log in burst_logs.items()}},
        "max_stale_reports": int(max(r["stale_reports"] for r in rows)),
        "runtime_s": time.time() - t0,
    }
    (OUT / "phase4_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    m = main()
    print(f"runtime {m['runtime_s'] / 60:.1f} min")
    print("baseline OSPA (no jammer):", {k: round(v, 1) for k, v in m["baseline_ospa"].items()})
    print("jammer EIRP where worst-case OSPA doubles:",
          {k: v["eirp_for_2x_ospa_dbm"] for k, v in m["resilience"].items()})
    print("burst:", json.dumps({k: m["burst"][k] for k in WAVEFORMS}, indent=1))
    print("max stale reports in any run:", m["max_stale_reports"])
