"""Phase 3: backhaul link under jamming. BER and PDR curves for plain, FH and DSSS links.

Common frame: 64 channels x 1 MHz, 1 Mbit/s BPSK. Plain uses one fixed channel; slow FH hops one
channel per packet attempt; DSSS spreads over all 64 channels (Gp = 64, 18.1 dB). Every jammer has
the same total received power J, only its spectral shape differs.

Outputs in results/:
  phase3_validation.png     analytic models vs baseband Monte Carlo (BPSK, DSSS + tone, FH + PBN)
  phase3_ber_vs_js.png      BER vs J/S for each waveform and jammer, incl. worst-case partial band
  phase3_partial_band.png   FH BER vs jammed fraction rho: the jammer's optimum
  phase3_pdr.png            packet delivery ratio and latency vs jammer EIRP (link-level, ARQ)
  phase3_metrics.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from skyshield.comms.jammers import BarrageJammer, PartialBandJammer, SpotJammer
from skyshield.comms.link import BackhaulLink
from skyshield.comms.modulation import BCH_63_45, PacketSpec, ber_bpsk, db2lin
from skyshield.comms.simulate import bpsk_ber, dsss_tone_ber, fh_partial_band_per
from skyshield.comms.waveforms import DSSS, FrequencyHopping, Plain

SEED = 11
N_CH = 64
EBN0_DB = 12.0  # for the BER-vs-J/S study
JS_GRID = np.arange(-30.0, 40.1, 1.0)
OUT = Path(__file__).resolve().parents[1] / "results"

# Link-level geometry (metres, same frame as the tracker): sensor node -> fusion centre.
# Backhaul antennas are mast-mounted with line of sight (free space, exponent 2); the jammer is
# near the ground (exponent 2.5, the Jammer default).
BACKHAUL_PL_EXP = 2.0
FUSION = np.array([0.0, 0.0])
SENSOR = np.array([2000.0, 0.0])
JAMMER = np.array([-2500.0, 2500.0])  # ~3.5 km from the fusion centre
EIRP_GRID = np.arange(0.0, 70.1, 1.0)
WF = {"plain": Plain(channel=0), "FH": FrequencyHopping(), "DSSS": DSSS()}
WF_COLORS = {"plain": "tab:gray", "FH": "tab:blue", "DSSS": "tab:orange"}


# ----- analytic BER ---------------------------------------------------------------------------


def mean_ber(wf, fractions_states: list[np.ndarray], js: float, ebn0: float) -> float:
    """Average BER over equally likely jammer states and (for FH) hop channels. S = 1."""
    n0 = 1.0 / ebn0
    vals = [np.mean(ber_bpsk(1.0 / (n0 + wf.interference_options(js * f))))
            for f in fractions_states]
    return float(np.mean(vals))


def jammer_states(kind: str, rho: float = 0.1) -> list[np.ndarray]:
    if kind == "barrage":
        return [np.full(N_CH, 1.0 / N_CH)]
    if kind == "spot":
        f = np.zeros(N_CH)
        f[0] = 1.0  # on the plain link's channel
        return [f]
    if kind == "partial":
        return PartialBandJammer(np.zeros(2), rho=rho).all_states(N_CH)
    raise ValueError(kind)


def fh_ber_rho(rho: float, ebj0: float, ebn0: float) -> float:
    """Continuous-rho FH BER: hit with prob rho, density J0/rho in a hit channel."""
    hit = ber_bpsk(1.0 / (1.0 / ebn0 + 1.0 / (rho * ebj0)))
    return float(rho * hit + (1 - rho) * ber_bpsk(ebn0))


def worst_partial_band(js: float, ebn0: float) -> tuple[float, float]:
    """Max FH BER over the number of jammed channels k = 1..N. Returns (ber, rho)."""
    best = (0.0, 1.0)
    for k in range(1, N_CH + 1):
        b = mean_ber(WF["FH"], jammer_states("partial", k / N_CH), js, ebn0)
        if b > best[0]:
            best = (b, k / N_CH)
    return best


# ----- experiment -----------------------------------------------------------------------------


def validation(rng) -> dict:
    ebn0 = np.arange(0.0, 10.1, 1.0)
    bpsk_sim = [bpsk_ber(e, 1_000_000, rng) for e in ebn0]
    js = np.arange(0.0, 26.1, 2.0)
    ds_sim = [dsss_tone_ber(10.0, j, N_CH, 100_000, rng) for j in js]
    rho = np.array([0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0])
    pkt = PacketSpec(256)
    fh_sim, fh_th = [], []
    for r in rho:
        fh_sim.append(fh_partial_band_per(15.0, 10.0, r, N_CH, 256, 40_000, rng))
        k = max(1, round(r * N_CH))
        hit = 1 / (1 / db2lin(15.0) + N_CH / (k * db2lin(10.0)))
        fh_th.append(float(k / N_CH * pkt.per(hit) + (1 - k / N_CH) * pkt.per(db2lin(15.0))))
    return {
        "bpsk": {"ebn0_db": ebn0.tolist(), "sim": bpsk_sim,
                 "theory": ber_bpsk(db2lin(ebn0)).tolist()},
        "dsss_tone": {"ebn0_db": 10.0, "gp": N_CH, "js_db": js.tolist(), "sim": ds_sim,
                      "theory": ber_bpsk(1 / (1 / db2lin(10.0) + db2lin(js) / N_CH)).tolist()},
        "fh_pbn": {"ebn0_db": 15.0, "ebj0_db": 10.0, "packet_bits": 256, "rho": rho.tolist(),
                   "sim_per": fh_sim, "theory_per": fh_th},
    }


def ber_study() -> dict:
    ebn0 = db2lin(EBN0_DB)
    js = db2lin(JS_GRID)
    curves = {
        "plain / spot": [mean_ber(WF["plain"], jammer_states("spot"), j, ebn0) for j in js],
        "plain / barrage": [mean_ber(WF["plain"], jammer_states("barrage"), j, ebn0) for j in js],
        "FH / barrage": [mean_ber(WF["FH"], jammer_states("barrage"), j, ebn0) for j in js],
        "FH / spot": [mean_ber(WF["FH"], jammer_states("spot"), j, ebn0) for j in js],
        "DSSS / any": [mean_ber(WF["DSSS"], jammer_states("spot"), j, ebn0) for j in js],
    }
    worst = [worst_partial_band(j, ebn0) for j in js]
    curves["FH / worst partial-band"] = [b for b, _ in worst]
    return {"ebn0_db": EBN0_DB, "js_db": JS_GRID.tolist(), "ber": curves,
            "worst_rho": [r for _, r in worst]}


def link(wf_name: str, retries: int = 3) -> BackhaulLink:
    return BackhaulLink(SENSOR, FUSION, waveform=WF[wf_name], n_channels=N_CH,
                        packet=PacketSpec(512, BCH_63_45), max_retries=retries,
                        path_loss_exp=BACKHAUL_PL_EXP)


def pdr_study(rng) -> dict:
    jam_kinds = {
        "spot": lambda e: SpotJammer(JAMMER, eirp_dbm=e, channel=0),
        "partial-band (rho 0.1)": lambda e: PartialBandJammer(JAMMER, eirp_dbm=e, rho=0.1),
        "barrage": lambda e: BarrageJammer(JAMMER, eirp_dbm=e),
    }
    out = {"eirp_dbm": EIRP_GRID.tolist(), "pdr": {}, "pdr_no_arq": {}, "latency_ms": {},
           "js_db": [], "eirp_for_pdr90_dbm": {}}
    ref = link("plain")
    out["snr_db"] = ref.snr_db
    out["js_db"] = [ref.js_db([BarrageJammer(JAMMER, eirp_dbm=e)]) for e in EIRP_GRID]
    for jname, make in jam_kinds.items():
        for wf in WF:
            key = f"{wf} / {jname}"
            pdr = [link(wf).expected_pdr([make(e)], rng) for e in EIRP_GRID]
            out["pdr"][key] = pdr
            out["pdr_no_arq"][key] = [link(wf, 0).expected_pdr([make(e)], rng) for e in EIRP_GRID]
            below = np.nonzero(np.array(pdr) < 0.9)[0]
            out["eirp_for_pdr90_dbm"][key] = float(EIRP_GRID[below[0]]) if len(below) else None
    # Mean latency of delivered packets, simulated, for FH under partial-band.
    lk = link("FH")
    lat = []
    for e in EIRP_GRID:
        jam = [jam_kinds["partial-band (rho 0.1)"](e)]
        d = [lk.send(0.0, jam, rng) for _ in range(800)]
        ok = [x.delay_s for x in d if x.delivered]
        lat.append(1e3 * float(np.mean(ok)) if ok else None)
    out["latency_ms"]["FH / partial-band (rho 0.1)"] = lat
    return out


# ----- figures --------------------------------------------------------------------------------


def plot_validation(v: dict, path: Path) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    def mc(a, x, p, n, label="MC (95% CI)"):
        p = np.asarray(p, float)
        half = 1.96 * np.sqrt(np.maximum(p * (1 - p), 1.0 / n) / n)  # binomial, floor at 1 event
        lo = np.minimum(half, p * 0.999)
        a.errorbar(x, np.maximum(p, 1.0 / n), yerr=[lo, half], fmt="o", color="tab:red",
                   ms=4, capsize=2, label=label)

    b = v["bpsk"]
    ax[0].semilogy(b["ebn0_db"], b["theory"], "k-", label="theory Q(sqrt(2Eb/N0))")
    mc(ax[0], b["ebn0_db"], b["sim"], 1_000_000)
    ax[0].set(xlabel="Eb/N0 (dB)", ylabel="BER", title="BPSK in AWGN (1e6 bits/point)")
    d = v["dsss_tone"]
    ax[1].semilogy(d["js_db"], d["theory"], "k-", label="Gaussian approx., J/Gp")
    ax[1].set_yscale("log")
    mc(ax[1], d["js_db"], d["sim"], 100_000, "chip-level MC (95% CI)")
    ax[1].set(xlabel="J/S (dB)", title=f"DSSS Gp={d['gp']} vs CW tone, Eb/N0 = 10 dB")
    f = v["fh_pbn"]
    ax[2].semilogx(f["rho"], f["theory_per"], "k-", label="analytic")
    mc(ax[2], f["rho"], f["sim_per"], 40_000)
    ax[2].set(xlabel="jammed fraction rho", ylabel="packet error rate",
              title="slow FH vs partial-band noise, Eb/J0 = 10 dB")
    for a in ax:
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ber(b: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        "plain / spot": ("tab:gray", "-"), "plain / barrage": ("tab:gray", "--"),
        "FH / spot": ("tab:blue", ":"), "FH / barrage": ("tab:blue", "--"),
        "FH / worst partial-band": ("tab:blue", "-"), "DSSS / any": ("tab:orange", "-"),
    }
    for k, (c, ls) in styles.items():
        ax.semilogy(b["js_db"], np.maximum(b["ber"][k], 1e-9), ls, color=c, lw=2, label=k)
    ax.set_ylim(1e-8, 0.6)
    ax.annotate("plain, FH and DSSS coincide under barrage:\nspreading the jammer's power\n"
                "gives the same J0 everywhere", xy=(5, 1e-5), xytext=(14, 1e-6), fontsize=8,
                arrowprops={"arrowstyle": "->", "color": "0.4"})
    ax.set_xlabel("J/S at the receiver (dB), same total jammer power for every shape")
    ax.set_ylabel("bit error rate (uncoded BPSK)")
    ax.set_title(f"Anti-jam comparison, 64 x 1 MHz band, Eb/N0 = {b['ebn0_db']:g} dB")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_partial_band(path: Path) -> dict:
    rho = np.logspace(-3, 0, 200)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rows = []
    for ebj0_db in (5, 10, 15, 20, 25):
        ber = [fh_ber_rho(r, db2lin(ebj0_db), db2lin(30.0)) for r in rho]
        i = int(np.argmax(ber))
        rows.append({"ebj0_db": ebj0_db, "worst_rho": float(rho[i]), "worst_ber": float(ber[i]),
                     "barrage_ber": float(ber[-1])})
        ax.loglog(rho, ber, label=f"Eb/J0 = {ebj0_db} dB")
        ax.plot(rho[i], ber[i], "kv", ms=5)
    ax.set_xlabel("fraction of band jammed, rho")
    ax.set_ylabel("BER (FH, uncoded BPSK)")
    ax.set_title("Optimal partial-band jamming vs FH (markers = worst rho)\nEb/N0 = 30 dB")
    ax.set_ylim(1e-8, 1)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"curves": rows}


def plot_pdr(p: dict, path: Path) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), sharey=False)
    e = np.array(p["eirp_dbm"])
    jams = ["spot", "partial-band (rho 0.1)", "barrage"]
    for a, jname in zip(ax, jams):
        for wf in WF:
            key = f"{wf} / {jname}"
            a.plot(e, p["pdr"][key], "-", color=WF_COLORS[wf], lw=2, label=f"{wf}, 3 retries")
            a.plot(e, p["pdr_no_arq"][key], ":", color=WF_COLORS[wf], lw=1.5,
                   label=f"{wf}, no ARQ")
        a.set_title(f"{jname} jammer")
        a.set_xlabel("jammer EIRP (dBm), 3.5 km from fusion centre")
        a.grid(alpha=0.3)
        top = a.secondary_xaxis("top", functions=(lambda x: x + (p["js_db"][0] - e[0]),
                                                  lambda x: x - (p["js_db"][0] - e[0])))
        top.set_xlabel("J/S at receiver (dB)")
    ax[0].set_ylabel("packet delivery ratio")
    ax[0].legend(fontsize=7, loc="lower left")
    lat = p["latency_ms"]["FH / partial-band (rho 0.1)"]
    ins = ax[1].inset_axes([0.08, 0.1, 0.42, 0.35])
    ins.plot(e, [np.nan if x is None else x for x in lat], color="tab:blue")
    ins.set_title("FH mean latency (ms)", fontsize=7)
    ins.tick_params(labelsize=6)
    fig.suptitle(f"Sensor -> fusion link: 2 km, 20 dBm, SNR {p['snr_db']:.0f} dB, "
                 "512-bit reports, BCH(63,45)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> dict:
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    v = validation(rng)
    plot_validation(v, OUT / "phase3_validation.png")
    b = ber_study()
    plot_ber(b, OUT / "phase3_ber_vs_js.png")
    pb = plot_partial_band(OUT / "phase3_partial_band.png")
    p = pdr_study(rng)
    plot_pdr(p, OUT / "phase3_pdr.png")

    def js_at(curve, target=1e-5):
        idx = np.nonzero(np.array(curve) > target)[0]
        return float(JS_GRID[idx[0]]) if len(idx) else None

    metrics = {
        "seed": SEED, "n_channels": N_CH, "processing_gain_db": 10 * np.log10(N_CH),
        "validation": v, "ber_vs_js": b, "partial_band": pb, "pdr": p,
        "js_for_ber_1e-5_db": {k: js_at(c) for k, c in b["ber"].items()},
        "runtime_s": time.time() - t0,
    }
    (OUT / "phase3_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    m = main()
    print(f"runtime {m['runtime_s']:.0f} s")
    print("J/S (dB) at which BER exceeds 1e-5:", json.dumps(m["js_for_ber_1e-5_db"]))
    print("worst-case rho vs Eb/J0:", [(r["ebj0_db"], round(r["worst_rho"], 3))
                                       for r in m["partial_band"]["curves"]])
    print(f"link SNR {m['pdr']['snr_db']:.1f} dB")
    print("jammer EIRP (dBm) where PDR drops below 0.9:",
          json.dumps(m["pdr"]["eirp_for_pdr90_dbm"], indent=1))
