"""Simulation helpers and Plotly figures for the Streamlit dashboard (no Streamlit imports here,
so this module is testable on its own)."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from skyshield.comms.link import BackhaulLink
from skyshield.dashboard.replay import ellipse_params
from skyshield.integration.closed_loop import Frame, RunLog
from skyshield.integration.network import (
    BACKHAUL_PL_EXP,
    FUSION,
    JAMMER_POS,
    JAMMERS,
    RADARS,
    WAVEFORMS,
    build_sim,
)

LINK_TYPES = tuple(WAVEFORMS)  # plain, FH, DSSS


@dataclass(frozen=True)
class SimParams:
    jammer: str | None  # key of JAMMERS or None
    eirp_dbm: float
    t_on: float
    t_off: float
    pd: float
    clutter: float
    mc_seed: int
    scenario_seed: int = 2024


def _run_one(args: tuple[str, SimParams]) -> RunLog:
    waveform, p = args
    sim = build_sim(waveform, p.jammer, p.eirp_dbm, (p.t_on, p.t_off),
                    scenario_seed=p.scenario_seed, pd=p.pd, clutter=p.clutter)
    return sim.run(np.random.default_rng([p.mc_seed, 1]), np.random.default_rng([p.mc_seed, 2]),
                   record=True)


def run_all(p: SimParams, parallel: bool = True) -> dict[str, RunLog]:
    """One recorded run per link type plus an unjammed reference ("no jamming"). All runs share
    the sensor random stream, so they see identical detections and clutter."""
    jobs = [(wf, p) for wf in LINK_TYPES]
    jobs.append(("plain", SimParams(None, p.eirp_dbm, p.t_on, p.t_off, p.pd, p.clutter,
                                    p.mc_seed, p.scenario_seed)))
    if parallel:
        with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
            logs = list(pool.map(_run_one, jobs))
    else:
        logs = [_run_one(j) for j in jobs]
    return dict(zip([*LINK_TYPES, "no jamming"], logs))


def link_table(p: SimParams) -> list[dict]:
    """Link budget per radar: length, SNR, and jammer-to-signal ratio while the jammer is on."""
    jams = [] if p.jammer is None else [JAMMERS[p.jammer](p.eirp_dbm)]
    rows = []
    for i, pos in enumerate(RADARS):
        link = BackhaulLink(pos, FUSION, path_loss_exp=BACKHAUL_PL_EXP)
        js = link.js_db(jams)
        rows.append({"radar": f"R{i}", "link (km)": round(link.distance_m / 1e3, 2),
                     "SNR (dB)": round(link.snr_db, 1),
                     "J/S when on (dB)": None if not np.isfinite(js) else round(js, 1)})
    return rows


# ----- Plotly replay -----------------------------------------------------------------------------

def _lines(segments: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[list, list]:
    xs: list = []
    ys: list = []
    for a, b in segments:
        xs += [a[0], b[0], None]
        ys += [a[1], b[1], None]
    return xs, ys


def _ellipses(f: Frame, n: int = 24) -> tuple[list, list]:
    xs: list = []
    ys: list = []
    th = np.linspace(0, 2 * np.pi, n)
    for xy, cov in zip(f.track_xy, f.track_cov):
        w, h, a = ellipse_params(cov)
        w, h, a = max(w, 40.0) / 2, max(h, 40.0) / 2, np.radians(a)
        ex = xy[0] + w * np.cos(th) * np.cos(a) - h * np.sin(th) * np.sin(a)
        ey = xy[1] + w * np.cos(th) * np.sin(a) + h * np.sin(th) * np.cos(a)
        xs += [*ex.tolist(), None]
        ys += [*ey.tolist(), None]
    return xs, ys


def frame_traces(f: Frame) -> list[dict]:
    """The per-step traces, always in the same order (Plotly animation needs that)."""
    ok = [i for i, d in enumerate(f.delivered) if d]
    lost = [i for i, d in enumerate(f.delivered) if not d]

    def dets(idx):
        pts = [f.det_xy[i] for i in idx if len(f.det_xy[i])]
        return np.vstack(pts) if pts else np.zeros((0, 2))

    d_ok, d_lost = dets(ok), dets(lost)
    l_ok = _lines([(RADARS[i], FUSION) for i in ok])
    l_lost = _lines([(RADARS[i], FUSION) for i in lost])
    ex, ey = _ellipses(f)
    return [
        {"type": "scatter", "x": l_ok[0], "y": l_ok[1], "mode": "lines",
         "line": {"color": "#2ca02c", "width": 3}, "name": "link up", "hoverinfo": "skip"},
        {"type": "scatter", "x": l_lost[0], "y": l_lost[1], "mode": "lines",
         "line": {"color": "#d62728", "width": 3, "dash": "dot"}, "name": "report lost",
         "hoverinfo": "skip"},
        {"type": "scatter", "x": d_ok[:, 0].tolist(), "y": d_ok[:, 1].tolist(), "mode": "markers",
         "marker": {"color": "#1f77b4", "size": 5, "opacity": 0.6}, "name": "detection (fused)"},
        {"type": "scatter", "x": d_lost[:, 0].tolist(), "y": d_lost[:, 1].tolist(),
         "mode": "markers", "marker": {"color": "#d62728", "size": 5, "opacity": 0.5},
         "name": "detection (lost)"},
        {"type": "scatter", "x": f.truth_xy[:, 0].tolist(), "y": f.truth_xy[:, 1].tolist(),
         "mode": "markers", "marker": {"symbol": "x", "color": "#ff7f0e", "size": 10},
         "name": "drone (truth)", "text": [f"drone {i}" for i in f.truth_ids]},
        {"type": "scatter", "x": ex, "y": ey, "mode": "lines",
         "line": {"color": "#2ca02c", "width": 1.5}, "name": "track 2σ", "hoverinfo": "skip"},
        {"type": "scatter", "x": f.track_xy[:, 0].tolist(), "y": f.track_xy[:, 1].tolist(),
         "mode": "markers+text", "marker": {"color": "#2ca02c", "size": 6},
         "text": [f"T{i}" for i in f.track_ids], "textposition": "top right",
         "name": "confirmed track"},
        {"type": "scatter", "x": [JAMMER_POS[0]], "y": [JAMMER_POS[1]], "mode": "markers+text",
         "marker": {"symbol": "star", "size": 22 if f.jammer_on else 12,
                    "color": "#d62728" if f.jammer_on else "#aaa"},
         "text": ["jammer ON" if f.jammer_on else "jammer off"], "textposition": "bottom right",
         "name": "jammer"},
    ]


def replay_figure(log: RunLog, stride: int = 2, height: int = 620) -> dict:
    """Plotly figure dict with a play button and a time slider."""
    frames = log.frames
    if frames is None:
        raise ValueError("run must be recorded")
    steps = list(range(0, len(frames), stride))
    paths: dict[int, list] = {}
    for g in frames:
        for tid, xy in zip(g.truth_ids, g.truth_xy):
            paths.setdefault(tid, []).append(xy)
    px: list = []
    py: list = []
    for pts in paths.values():
        a = np.array(pts)
        px += [*a[:, 0].tolist(), None]
        py += [*a[:, 1].tolist(), None]
    static = [
        {"type": "scatter", "x": px, "y": py, "mode": "lines", "hoverinfo": "skip",
         "line": {"color": "#999", "width": 1}, "opacity": 0.4, "name": "drone paths"},
        {"type": "scatter", "x": [r[0] for r in RADARS], "y": [r[1] for r in RADARS],
         "mode": "markers+text", "marker": {"symbol": "triangle-up", "size": 14, "color": "#9467bd"},
         "text": [f"R{i}" for i in range(len(RADARS))], "textposition": "bottom center",
         "name": "radar"},
        {"type": "scatter", "x": [FUSION[0]], "y": [FUSION[1]], "mode": "markers+text",
         "marker": {"symbol": "square", "size": 12, "color": "#9467bd"}, "text": ["fusion"],
         "textposition": "top right", "name": "fusion centre"},
    ]
    n_static = len(static)
    first = frame_traces(frames[steps[0]])
    anim = [{"name": str(k), "data": frame_traces(frames[k]),
             "traces": list(range(n_static, n_static + len(first))),
             "layout": {"title": {"text": _title(frames[k])}}} for k in steps]
    allp = np.vstack([np.array(RADARS), JAMMER_POS[None], FUSION[None],
                      *[g.truth_xy for g in frames if len(g.truth_xy)]])
    lo, hi = allp.min(0) - 300, allp.max(0) + 300
    slider = {"active": 0, "currentvalue": {"prefix": "t = ", "suffix": " s"},
              "x": 0.16, "len": 0.84, "y": 0, "yanchor": "top", "pad": {"t": 50},
              "steps": [{"label": f"{frames[k].t:.0f}", "method": "animate",
                         "args": [[str(k)], {"mode": "immediate", "frame": {"duration": 0},
                                             "transition": {"duration": 0}}]} for k in steps]}
    play = {"type": "buttons", "showactive": False, "direction": "left", "x": 0.15, "y": 0,
            "xanchor": "right", "yanchor": "top", "pad": {"t": 60, "r": 10},
            "buttons": [
                {"label": "▶ play", "method": "animate",
                 "args": [None, {"frame": {"duration": 120, "redraw": True},
                                 "transition": {"duration": 0}, "fromcurrent": True}]},
                {"label": "❚❚ pause", "method": "animate",
                 "args": [[None], {"mode": "immediate", "frame": {"duration": 0}}]}]}
    layout = {
        "height": height, "title": {"text": _title(frames[steps[0]])},
        "xaxis": {"range": [lo[0], hi[0]], "title": "x (m)", "zeroline": False,
                  "constrain": "domain"},
        "yaxis": {"range": [lo[1], hi[1]], "title": "y (m)", "scaleanchor": "x",
                  "zeroline": False, "constrain": "domain"},
        "legend": {"orientation": "h", "y": 1.08, "x": 0},
        "margin": {"l": 50, "r": 10, "t": 90, "b": 90},
        "updatemenus": [play], "sliders": [slider],
    }
    return {"data": static + first, "layout": layout, "frames": anim}


def _title(f: Frame) -> str:
    return (f"t = {f.t:.0f} s · OSPA {f.ospa:.0f} m · {len(f.track_ids)} tracks / "
            f"{len(f.truth_ids)} drones · {int(f.delivered.sum())}/{len(f.delivered)} reports fused")
