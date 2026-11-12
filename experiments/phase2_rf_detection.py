"""Phase 2: CNN drone detector on synthetic RF spectrograms. Robustness study (headline experiment 1).

Trains a CNN on drone models A-D in a mixed environment (noise, Wi-Fi, Bluetooth), then measures
Pd vs in-band SNR at a fixed false-alarm rate (Pfa = 1 %) in five conditions:

    clean          training drone models, no interferers
    mixed          training drone models, interferers as in training (each with p = 0.4)
    heavy          training drone models, Wi-Fi AND Bluetooth always on, strong (10-25 dB)
    held-out E     unseen drone model, same signal family as training (GFSK + video)
    held-out F     unseen drone model, unseen modulation (chirp spread spectrum)

The detection threshold is set once, on negatives drawn from the "mixed" environment, and is then
held fixed, as a deployed detector would be. Two classical baselines get the same treatment:
a wideband energy detector and a time-frequency peak (narrowband burst) detector.

Finally, the "mixed" CNN Pd(SNR) curve drives an RF direction-finding sensor model to give
Pd vs distance for a 20 dBm control link.

Outputs in results/: phase2_spectrograms.png, phase2_pd_vs_snr.png, phase2_roc.png,
phase2_pd_vs_range.png, phase2_metrics.json, models/rf_cnn.pt
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from skyshield.evaluation.metrics import auc, roc_curve, threshold_at_pfa
from skyshield.rf_detection.cnn import TrainConfig, predict_proba, save_model, train_detector
from skyshield.rf_detection.features import (
    DatasetSpec,
    build_dataset,
    random_dataset,
    spectrogram,
)
from skyshield.rf_detection.signals import (
    HELD_OUT_MODELS,
    TRAIN_MODELS,
    SceneSpec,
    make_scene,
)
from skyshield.sensors.rf_df import DetectionCurve, RFDFSensor

SEED = 7
N_TRAIN, N_VAL = 10_000, 2_000
N_NEG = 3_000  # negatives for threshold calibration
N_NEG_EVAL = 1_000  # negatives per evaluation environment
N_POS = 200  # positives per (condition, SNR) cell
PFA = 0.01
SNR_GRID = np.arange(-15.0, 20.1, 2.5)
ROC_SNR = 0.0
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
CACHE = ROOT / "data" / "cache"

CONDITIONS = {
    "clean": {"models": TRAIN_MODELS, "env": "clean"},
    "mixed": {"models": TRAIN_MODELS, "env": "mixed"},
    "heavy": {"models": TRAIN_MODELS, "env": "heavy"},
    "held-out E": {"models": ("E",), "env": "mixed"},
    "held-out F": {"models": ("F",), "env": "mixed"},
}
COLORS = {"clean": "tab:green", "mixed": "k", "heavy": "tab:red",
          "held-out E": "tab:blue", "held-out F": "tab:purple"}


def environment(env: str, rng: np.random.Generator) -> tuple[tuple[str, ...], dict]:
    if env == "clean":
        return (), {}
    if env == "heavy":
        names = ("wifi", "bt")
        return names, {k: float(rng.uniform(10, 25)) for k in names}
    names = tuple(k for k in ("wifi", "bt") if rng.random() < 0.4)
    return names, {k: float(rng.uniform(0, 25)) for k in names}


def scenes(n: int, rng, models=None, snr=None, env="mixed") -> list[SceneSpec]:
    out = []
    for _ in range(n):
        interf, isnr = environment(env, rng)
        drone = None if models is None else str(rng.choice(models))
        out.append(SceneSpec(drone, float(snr if snr is not None else 0.0), interf, isnr))
    return out


def cached(name: str, fn):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.npz"
    if path.exists():
        d = np.load(path)
        return d["X"], d["y"], d["B"]
    X, y, B = fn()
    np.savez(path, X=X, y=y, B=B)
    return X, y, B


def detector_scores(model, X: np.ndarray, B: np.ndarray) -> dict[str, np.ndarray]:
    return {"cnn": predict_proba(model, X), "energy": B[:, 0], "tf_peak": B[:, 1]}


def main() -> dict:
    t0 = time.time()
    OUT.mkdir(exist_ok=True)

    # ----- data --------------------------------------------------------------------------------
    def _train():
        X, y, B, _ = random_dataset(DatasetSpec(n=N_TRAIN), np.random.default_rng(SEED))
        return X, y, B

    def _val():
        X, y, B, _ = random_dataset(DatasetSpec(n=N_VAL), np.random.default_rng(SEED + 1))
        return X, y, B

    X, y, _ = cached(f"train_{SEED}_{N_TRAIN}", _train)
    Xv, yv, _ = cached(f"val_{SEED}_{N_VAL}", _val)
    print(f"data ready ({time.time() - t0:.0f} s)", flush=True)

    # ----- train -------------------------------------------------------------------------------
    model, hist = train_detector(X, y, Xv, yv, TrainConfig(epochs=10, seed=SEED))
    save_model(model, OUT / "models" / "rf_cnn.pt")
    print(f"trained: best epoch {hist.best_epoch}, val acc {hist.val_acc[hist.best_epoch]:.3f} "
          f"({time.time() - t0:.0f} s)", flush=True)
    del X, y

    # ----- thresholds at Pfa on the mixed environment ------------------------------------------
    eval_rng = np.random.default_rng(SEED + 2)
    Xn, _, Bn = build_dataset(scenes(N_NEG, eval_rng, env="mixed"), eval_rng)
    neg = detector_scores(model, Xn, Bn)
    thr = {k: threshold_at_pfa(v, PFA) for k, v in neg.items()}

    # Realised Pfa in each environment with that fixed threshold.
    pfa_by_env = {}
    neg_by_env = {"mixed": neg}
    for env in ("clean", "heavy"):
        Xe, _, Be = build_dataset(scenes(N_NEG_EVAL, eval_rng, env=env), eval_rng)
        neg_by_env[env] = detector_scores(model, Xe, Be)
    for env, sc in neg_by_env.items():
        pfa_by_env[env] = {k: float(np.mean(sc[k] > thr[k])) for k in thr}

    # ----- Pd vs SNR ---------------------------------------------------------------------------
    pd = {det: {c: [] for c in CONDITIONS} for det in thr}
    roc_pos = {}
    for cname, c in CONDITIONS.items():
        for snr in SNR_GRID:
            Xp, _, Bp = build_dataset(scenes(N_POS, eval_rng, c["models"], snr, c["env"]), eval_rng)
            sc = detector_scores(model, Xp, Bp)
            for det, t in thr.items():
                pd[det][cname].append(float(np.mean(sc[det] > t)))
            if snr == ROC_SNR:
                roc_pos[cname] = sc
        print(f"  {cname:<11} CNN Pd: " + " ".join(f"{p:.2f}" for p in pd["cnn"][cname]),
              flush=True)

    # ROC at ROC_SNR: positives of each condition against negatives of the matching environment.
    roc = {}
    for cname, c in CONDITIONS.items():
        negs = neg_by_env[c["env"]]
        roc[cname] = {}
        for det in thr:
            s = np.r_[roc_pos[cname][det], negs[det]]
            lab = np.r_[np.ones(len(roc_pos[cname][det])), np.zeros(len(negs[det]))]
            pfa_c, pd_c = roc_curve(s, lab)
            roc[cname][det] = {"auc": auc(s, lab), "pfa": pfa_c, "pd": pd_c}

    # ----- figures -----------------------------------------------------------------------------
    plot_examples(OUT / "phase2_spectrograms.png")
    plot_pd_vs_snr(pd, pfa_by_env, OUT / "phase2_pd_vs_snr.png")
    plot_roc(roc, OUT / "phase2_roc.png")
    curve = DetectionCurve(SNR_GRID, np.maximum.accumulate(np.array(pd["cnn"]["mixed"])), PFA)
    range_rows = plot_pd_vs_range(curve, OUT / "phase2_pd_vs_range.png")

    snr_at = {c: snr_for_pd(np.array(pd["cnn"][c]), 0.9) for c in CONDITIONS}
    metrics = {
        "seed": SEED, "n_train": N_TRAIN, "n_val": N_VAL, "pfa": PFA,
        "snr_grid_db": SNR_GRID.tolist(),
        "train_models": list(TRAIN_MODELS), "held_out_models": list(HELD_OUT_MODELS),
        "training": {"best_epoch": hist.best_epoch, "val_acc": hist.val_acc,
                     "val_loss": hist.val_loss, "train_loss": hist.train_loss},
        "thresholds": thr,
        "realised_pfa": pfa_by_env,
        "cnn": {"pd_vs_snr": pd["cnn"], "snr_for_pd90_db": snr_at},
        "energy": {"pd_vs_snr": pd["energy"]},
        "tf_peak": {"pd_vs_snr": pd["tf_peak"]},
        "auc_at_0db": {c: {d: roc[c][d]["auc"] for d in thr} for c in CONDITIONS},
        "rf_df_pd_vs_range": range_rows,
        "runtime_s": time.time() - t0,
    }
    (OUT / "phase2_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def snr_for_pd(pd: np.ndarray, target: float) -> float | None:
    """Lowest SNR on the grid (linearly interpolated) where Pd reaches target."""
    above = np.nonzero(pd >= target)[0]
    if len(above) == 0:
        return None
    i = above[0]
    if i == 0:
        return float(SNR_GRID[0])
    x0, x1, y0, y1 = SNR_GRID[i - 1], SNR_GRID[i], pd[i - 1], pd[i]
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))


def plot_examples(path: Path) -> None:
    rng = np.random.default_rng(3)
    cases = [
        ("noise only", SceneSpec()),
        ("Wi-Fi", SceneSpec(interferers=("wifi",), interferer_snr_db={"wifi": 15})),
        ("Bluetooth", SceneSpec(interferers=("bt",), interferer_snr_db={"bt": 15})),
        ("drone A (GFSK hopper)", SceneSpec("A", 12.0)),
        ("drone B (hopper + video)", SceneSpec("B", 12.0)),
        ("drone C (slow hopper)", SceneSpec("C", 12.0)),
        ("drone D (fast hopper + video)", SceneSpec("D", 12.0)),
        ("held-out E", SceneSpec("E", 12.0)),
        ("held-out F (chirp)", SceneSpec("F", 12.0)),
        ("drone A @ 3 dB + Wi-Fi + BT", SceneSpec("A", 3.0, ("wifi", "bt"),
                                                   {"wifi": 15, "bt": 15})),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.2), sharex=True, sharey=True)
    for ax, (title, spec) in zip(axes.flat, cases):
        S = spectrogram(make_scene(spec, rng))
        ax.imshow(S, aspect="auto", origin="lower", vmin=-5, vmax=25, cmap="viridis",
                  extent=[0, 6.55, -5, 5])
        ax.set_title(title, fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("time (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("frequency (MHz)")
    fig.suptitle("Synthetic 10 MHz ISM-band captures (dB above median); drones at 12 dB "
                 "in-band SNR unless noted")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pd_vs_snr(pd: dict, pfa_by_env: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    ax = axes[0]
    for c in CONDITIONS:
        ax.plot(SNR_GRID, pd["cnn"][c], "o-", color=COLORS[c], ms=4, label=c)
    ax.set_title(f"CNN detector, threshold fixed at Pfa = {PFA:.0%} (mixed environment)")
    ax.set_xlabel("in-band SNR (dB)")
    ax.set_ylabel("probability of detection")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    styles = {"cnn": ("CNN", "-"), "tf_peak": ("TF-peak detector", "--"),
              "energy": ("energy detector", ":")}
    for det, (name, ls) in styles.items():
        for c in ("mixed", "heavy"):
            ax.plot(SNR_GRID, pd[det][c], ls, color=COLORS[c], lw=1.8,
                    label=f"{name}, {c} (Pfa {pfa_by_env[c][det]:.1%})")
    ax.set_title("CNN vs classical detectors, same Pfa calibration")
    ax.set_xlabel("in-band SNR (dB)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(roc: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for c in CONDITIONS:
        r = roc[c]["cnn"]
        axes[0].plot(r["pfa"], r["pd"], color=COLORS[c], label=f"{c} (AUC {r['auc']:.3f})")
    axes[0].set_title(f"CNN ROC at {ROC_SNR:g} dB in-band SNR")
    for det, ls in (("cnn", "-"), ("tf_peak", "--"), ("energy", ":")):
        r = roc["mixed"][det]
        axes[1].plot(r["pfa"], r["pd"], ls, color="k", label=f"{det} (AUC {r['auc']:.3f})")
    axes[1].set_title(f"Detectors compared, mixed environment, {ROC_SNR:g} dB")
    for ax in axes:
        ax.plot([0, 1], [0, 1], color="0.7", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1)
        ax.set_xlabel("probability of false alarm")
        ax.set_ylabel("probability of detection")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pd_vs_range(curve: DetectionCurve, path: Path) -> list[dict]:
    d = np.linspace(100, 6000, 120)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rows = []
    for exp, ls in ((2.0, ":"), (2.5, "-"), (3.0, "--")):
        sensor = RFDFSensor(np.zeros(2), curve, path_loss_exp=exp)
        snr = sensor.snr_db(np.stack([d, np.zeros_like(d)], axis=1), 20.0)
        p = curve(snr)
        ax.plot(d / 1000, p, ls, color="k", label=f"path-loss exponent {exp:g}")
        idx = np.nonzero(p >= 0.9)[0]
        beyond = bool(len(idx)) and idx[-1] == len(d) - 1  # still >= 0.9 at the last distance
        rows.append({"path_loss_exp": exp,
                     "range_pd90_m": None if beyond or not len(idx) else float(d[idx[-1]]),
                     "beyond_max_distance_m": float(d[-1]) if beyond else None})
    ax.set_xlabel("distance to drone (km)")
    ax.set_ylabel("probability of detection")
    ax.set_title("RF DF sensor: 20 dBm control link, 1 MHz, NF 6 dB, CNN at Pfa 1 %")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return rows


if __name__ == "__main__":
    m = main()
    print(f"runtime {m['runtime_s'] / 60:.1f} min")
    print("SNR for Pd = 0.9 (CNN):", {k: (None if v is None else round(v, 1))
                                      for k, v in m["cnn"]["snr_for_pd90_db"].items()})
    print("realised Pfa:", json.dumps(m["realised_pfa"]))
    print("AUC at 0 dB:", json.dumps(m["auc_at_0db"]))
    print("RF DF range for Pd 0.9:", m["rf_df_pd_vs_range"])
