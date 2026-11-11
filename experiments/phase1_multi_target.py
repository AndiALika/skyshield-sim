"""Phase 1: multi-target tracking in clutter (EKF + GNN + M/N track management), scored with OSPA.

Outputs (in results/):
- phase1_tracking.png       one run: truth, clutter, confirmed tracks, OSPA and cardinality over time
- phase1_ospa_sweep.png     Monte Carlo mean OSPA vs clutter rate and vs probability of detection
- phase1_metrics.json       all numbers behind the figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from skyshield.evaluation.metrics import ospa
from skyshield.scenario.multi_target import MultiTargetScenario, crossing_scenario
from skyshield.sensors.radar import Radar
from skyshield.tracking.manager import MultiTargetTracker, TrackerConfig

SEED = 42
DT = 1.0
OSPA_C = 100.0  # m, cut-off: a missed or false track costs 100 m
OSPA_P = 2
N_MC = 20
CLUTTER_RATES = [0.0, 5.0, 10.0, 20.0, 40.0]
PDS = [0.6, 0.7, 0.8, 0.9, 1.0]
BASE_PD, BASE_CLUTTER = 0.9, 10.0
OUT = Path(__file__).resolve().parents[1] / "results"


def run_once(
    seed: int,
    pd: float = BASE_PD,
    clutter_rate: float = BASE_CLUTTER,
    config: TrackerConfig | None = None,
    keep_history: bool = False,
) -> dict:
    rng = np.random.default_rng(seed)
    sc: MultiTargetScenario = crossing_scenario(dt=DT, rng=rng)
    radar = Radar(position=np.zeros(2), pd=pd, clutter_rate=clutter_rate)
    tracker = MultiTargetTracker(config)

    d, loc, card, n_true, n_est = [], [], [], [], []
    hist: dict = {"tracks": {}, "clutter": [], "detections": []}
    for k in range(sc.n_steps):
        _, states = sc.truth_at(k)
        zs, origins = radar.measure_multi(states[:, :2], rng)
        conf = tracker.step(zs, radar.position, radar.R, sc.dt)

        est = np.array([t.position for t in conf]).reshape(-1, 2)
        res = ospa(est, states[:, :2], c=OSPA_C, p=OSPA_P)
        d.append(res.distance)
        loc.append(res.localisation)
        card.append(res.cardinality)
        n_true.append(len(states))
        n_est.append(len(conf))

        if keep_history:
            for t in conf:
                hist["tracks"].setdefault(t.track_id, []).append((k, *t.position))
            for z, o in zip(zs, origins):
                (hist["clutter"] if o < 0 else hist["detections"]).append(radar.to_cartesian(z))

    n_true, n_est = np.array(n_true), np.array(n_est)
    out = {
        "ospa": np.array(d),
        "ospa_loc": np.array(loc),
        "ospa_card": np.array(card),
        "n_true": n_true,
        "n_est": n_est,
        "mean_ospa": float(np.mean(d)),
        "card_correct_frac": float(np.mean(n_true == n_est)),
        "n_track_ids": len({tid for tid in hist["tracks"]}) if keep_history else None,
    }
    if keep_history:
        out["scenario"] = sc
        out["history"] = hist
        out["radar"] = radar
    return out


def monte_carlo(pd: float, clutter_rate: float, n: int = N_MC) -> dict:
    runs = [run_once(SEED + 1000 * i, pd=pd, clutter_rate=clutter_rate) for i in range(n)]
    m = np.array([r["mean_ospa"] for r in runs])
    c = np.array([r["card_correct_frac"] for r in runs])
    return {
        "pd": pd,
        "clutter_rate": clutter_rate,
        "mean_ospa": float(m.mean()),
        "ospa_ci95": float(1.96 * m.std(ddof=1) / np.sqrt(n)),
        "mean_loc": float(np.mean([r["ospa_loc"].mean() for r in runs])),
        "mean_card": float(np.mean([r["ospa_card"].mean() for r in runs])),
        "card_correct_frac": float(c.mean()),
    }


def plot_single(r: dict, path: Path) -> None:
    sc, hist, radar = r["scenario"], r["history"], r["radar"]
    fig = plt.figure(figsize=(13, 6.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[:, 0])
    if hist["clutter"]:
        cl = np.array(hist["clutter"])
        ax.scatter(cl[:, 0], cl[:, 1], s=2, c="0.8", label="clutter", rasterized=True)
    for i, t in enumerate(sc.targets):
        st = t.states[: max(0, sc.n_steps - t.start_step)]  # only the part inside the run
        ax.plot(st[:, 0], st[:, 1], "k-", lw=1.2, label="truth" if i == 0 else None)
        ax.annotate(f"T{t.target_id}", t.states[0, :2], fontsize=9, weight="bold")
    cmap = plt.get_cmap("tab10")
    for j, (tid, pts) in enumerate(sorted(hist["tracks"].items())):
        pts = np.array(pts)
        ax.plot(pts[:, 1], pts[:, 2], "-", color=cmap(j % 10), lw=2, alpha=0.85,
                label=f"track {tid}")
    ax.plot(*radar.position, "b^", ms=10, label="radar")
    ax.set_aspect("equal")
    ax.set_xlim(-radar.max_range, radar.max_range)
    ax.set_ylim(-radar.max_range, radar.max_range)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Pd={radar.pd}, clutter={radar.clutter_rate:g}/scan")
    ax.legend(fontsize=7, loc="lower left", ncol=2)

    k = np.arange(len(r["ospa"])) * DT
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(k, r["ospa"], "k", lw=1.2, label="OSPA")
    ax1.plot(k, r["ospa_loc"], "tab:blue", lw=0.9, label="localisation")
    ax1.plot(k, r["ospa_card"], "tab:red", lw=0.9, label="cardinality")
    ax1.set_ylabel(f"OSPA (m), c={OSPA_C:g}, p={OSPA_P}")
    ax1.set_title(f"mean OSPA {r['mean_ospa']:.1f} m")
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, OSPA_C * 1.05)

    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    ax2.step(k, r["n_true"], "k", where="post", label="true targets")
    ax2.step(k, r["n_est"], "tab:orange", where="post", label="confirmed tracks")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("count")
    ax2.set_title(f"cardinality correct {100 * r['card_correct_frac']:.0f}% of scans")
    ax2.legend(fontsize=8)

    fig.suptitle("Phase 1: EKF + GNN + M/N track management, crossing scenario")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sweep(clutter: list[dict], pd: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, rows, key, xlabel, fixed in [
        (axes[0], clutter, "clutter_rate", "clutter rate (false alarms / scan)", f"Pd={BASE_PD}"),
        (axes[1], pd, "pd", "probability of detection", f"clutter={BASE_CLUTTER:g}/scan"),
    ]:
        x = [r[key] for r in rows]
        ax.errorbar(x, [r["mean_ospa"] for r in rows], yerr=[r["ospa_ci95"] for r in rows],
                    fmt="o-", color="k", capsize=3, label="OSPA")
        ax.plot(x, [r["mean_loc"] for r in rows], "s--", color="tab:blue", label="localisation")
        ax.plot(x, [r["mean_card"] for r in rows], "^--", color="tab:red", label="cardinality")
        ax.set_xlabel(xlabel)
        ax.set_title(f"{fixed}, {N_MC} Monte Carlo runs (95% CI)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean OSPA (m)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> dict:
    OUT.mkdir(exist_ok=True)
    single = run_once(SEED, keep_history=True)
    plot_single(single, OUT / "phase1_tracking.png")

    clutter_rows = [monte_carlo(BASE_PD, lam) for lam in CLUTTER_RATES]
    pd_rows = [monte_carlo(p, BASE_CLUTTER) for p in PDS]
    plot_sweep(clutter_rows, pd_rows, OUT / "phase1_ospa_sweep.png")

    metrics = {
        "config": {
            "seed": SEED, "dt": DT, "ospa_c": OSPA_C, "ospa_p": OSPA_P, "n_mc": N_MC,
            "tracker": vars(TrackerConfig()),
        },
        "single_run": {
            "pd": BASE_PD, "clutter_rate": BASE_CLUTTER,
            "mean_ospa": single["mean_ospa"],
            "card_correct_frac": single["card_correct_frac"],
            "n_true_targets": len(single["scenario"].targets),
            "n_confirmed_track_ids": single["n_track_ids"],
        },
        "sweep_clutter": clutter_rows,
        "sweep_pd": pd_rows,
    }
    (OUT / "phase1_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    m = main()
    s = m["single_run"]
    print(f"single run: mean OSPA={s['mean_ospa']:.1f} m, cardinality correct "
          f"{100 * s['card_correct_frac']:.0f}%, {s['n_confirmed_track_ids']} track IDs for "
          f"{s['n_true_targets']} targets")
    print(f"clutter sweep (Pd={BASE_PD:.1f}):")
    for r in m["sweep_clutter"]:
        print(f"  lambda={r['clutter_rate']:>4.0f}  OSPA={r['mean_ospa']:5.1f} +/- {r['ospa_ci95']:.1f}"
              f"  card ok={100 * r['card_correct_frac']:.0f}%")
    print(f"Pd sweep (clutter={BASE_CLUTTER:g}):")
    for r in m["sweep_pd"]:
        print(f"  Pd={r['pd']:.1f}  OSPA={r['mean_ospa']:5.1f} +/- {r['ospa_ci95']:.1f}"
              f"  card ok={100 * r['card_correct_frac']:.0f}%")
