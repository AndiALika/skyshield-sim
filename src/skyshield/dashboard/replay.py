"""Matplotlib replay of recorded closed-loop runs, saved as an animated GIF.

Each panel shows one run: true drone paths, radar detections (blue if the report reached the
fusion centre, red if the backhaul lost it), confirmed tracks with 2-sigma ellipses, and each
radar's link drawn green or red. A strip below plots OSPA so far.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse

from skyshield.integration.closed_loop import Frame, RunLog

TRUTH_C, TRACK_C, DET_C, LOST_C = "0.15", "tab:green", "tab:blue", "tab:red"


def ellipse_params(cov: np.ndarray, n_sigma: float = 2.0) -> tuple[float, float, float]:
    """(width, height, angle_deg) of the n-sigma ellipse of a 2x2 covariance."""
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0.0, None)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    return 2 * n_sigma * np.sqrt(vals[1]), 2 * n_sigma * np.sqrt(vals[0]), float(angle)


def draw_map(
    ax: plt.Axes,
    frames: Sequence[Frame],
    k: int,
    radars: Sequence[np.ndarray],
    fusion: np.ndarray,
    jammer_pos: np.ndarray | None,
    trail: int = 40,
    df_sites: Sequence[np.ndarray] | None = None,
) -> None:
    f = frames[k]
    ax.clear()
    # DF sites and their bearing rays (dashed: blue if fused, red if lost on the backhaul).
    for i, site in enumerate(df_sites if len(f.df_delivered) else []):  # run without DF: skip
        ok = bool(f.df_delivered[i]) if i < len(f.df_delivered) else False
        if i < len(f.df_bearings):
            for b in f.df_bearings[i]:
                end = site + 6000.0 * np.array([np.cos(b), np.sin(b)])
                ax.plot(*np.c_[site, end], color=DET_C if ok else LOST_C, lw=0.7, ls="--",
                        alpha=0.55, zorder=1)
        ax.plot(*np.c_[site, fusion], color=TRACK_C if ok else LOST_C, lw=1.2,
                ls="-" if ok else ":", zorder=1)
        ax.scatter(*site, marker="D", s=45, color="tab:purple", zorder=5)
        ax.annotate(f"DF{i}", site, xytext=(5, 4), textcoords="offset points", fontsize=7)
    # Truth trails.
    paths: dict[int, list[np.ndarray]] = {}
    for g in frames[max(0, k - trail): k + 1]:
        for tid, xy in zip(g.truth_ids, g.truth_xy):
            paths.setdefault(tid, []).append(xy)
    for pts in paths.values():
        p = np.array(pts)
        ax.plot(p[:, 0], p[:, 1], color=TRUTH_C, lw=0.8, alpha=0.5)
    ax.scatter(*f.truth_xy.T, marker="x", s=30, color=TRUTH_C, lw=1.2, zorder=4)
    # Detections and links.
    for i, (rp, xy, clut, ok) in enumerate(zip(radars, f.det_xy, f.det_clutter, f.delivered)):
        c = DET_C if ok else LOST_C
        if len(xy):
            ax.scatter(xy[~clut, 0], xy[~clut, 1], s=12, color=c, alpha=0.8, zorder=3)
            ax.scatter(xy[clut, 0], xy[clut, 1], s=6, color=c, alpha=0.3, zorder=2)
        ax.plot(*np.c_[rp, fusion], color=TRACK_C if ok else LOST_C, lw=1.6,
                ls="-" if ok else ":", zorder=1)
        ax.scatter(*rp, marker="^", s=80, color="k", zorder=5)
        ax.annotate(f"R{i}", rp, xytext=(6, -12), textcoords="offset points", fontsize=8)
    ax.scatter(*fusion, marker="s", s=70, color="k", zorder=5)
    ax.annotate("fusion", fusion, xytext=(6, 4), textcoords="offset points", fontsize=8)
    # Tracks.
    for tid, xy, cov in zip(f.track_ids, f.track_xy, f.track_cov):
        w, h, a = ellipse_params(cov)
        ax.add_patch(Ellipse(xy, max(w, 40), max(h, 40), angle=a, fill=False,
                             color=TRACK_C, lw=1.4, zorder=6))
        ax.annotate(f"T{tid}", xy, xytext=(7, 5), textcoords="offset points", fontsize=7,
                    color=TRACK_C)
    if jammer_pos is not None:
        ax.scatter(*jammer_pos, marker="*", s=260 if f.jammer_on else 120,
                   color=LOST_C if f.jammer_on else "0.6", zorder=5)
        ax.annotate("jammer ON" if f.jammer_on else "jammer off", jammer_pos,
                    xytext=(8, -4), textcoords="offset points", fontsize=8,
                    color=LOST_C if f.jammer_on else "0.5")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def save_gif(
    logs: dict[str, RunLog],
    path: Path,
    radars: Sequence[np.ndarray],
    fusion: np.ndarray,
    jammer_pos: np.ndarray | None = None,
    steps: Sequence[int] | None = None,
    fps: int = 10,
    jam_window: tuple[float, float] | None = None,
    title: str = "",
    dpi: int = 70,
    df_sites: Sequence[np.ndarray] | None = None,
) -> Path:
    """Side-by-side GIF, one panel per recorded run (all runs must share the scenario)."""
    names = list(logs)
    frames = {n: logs[n].frames for n in names}
    if any(fr is None for fr in frames.values()):
        raise ValueError("runs must be recorded with run(..., record=True)")
    n_steps = len(frames[names[0]])
    steps = list(range(n_steps)) if steps is None else list(steps)

    pts = [np.asarray(radars), np.asarray(fusion)[None]]
    pts += [g.truth_xy for g in frames[names[0]] if len(g.truth_xy)]
    if jammer_pos is not None:
        pts.append(np.asarray(jammer_pos)[None])
    allp = np.vstack(pts)
    lo, hi = allp.min(0) - 300, allp.max(0) + 300

    fig = plt.figure(figsize=(5.0 * len(names), 6.4))
    gs = fig.add_gridspec(2, len(names), height_ratios=[4, 1.1], hspace=0.12, wspace=0.04)
    maps = [fig.add_subplot(gs[0, j]) for j in range(len(names))]
    strip = fig.add_subplot(gs[1, :])
    t_all = np.array([g.t for g in frames[names[0]]])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def draw(i: int) -> None:
        k = steps[i]
        for ax, n in zip(maps, names):
            draw_map(ax, frames[n], k, radars, fusion, jammer_pos, df_sites=df_sites)
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            g = frames[n][k]
            ax.set_title(f"{n}   t = {g.t:.0f} s   OSPA {g.ospa:.0f} m", fontsize=10)
        strip.clear()
        for j, n in enumerate(names):
            strip.plot(t_all[: k + 1], logs[n].ospa[: k + 1], color=colors[j % 10], lw=1.4,
                       label=n)
        if jam_window is not None:
            strip.axvspan(*jam_window, color=LOST_C, alpha=0.08)
        strip.set_xlim(t_all[steps[0]], t_all[steps[-1]])
        strip.set_ylim(0, 105)
        strip.set_ylabel("OSPA (m)", fontsize=8)
        strip.set_xlabel("time (s)", fontsize=8)
        strip.tick_params(labelsize=7)
        strip.grid(alpha=0.3)
        strip.legend(fontsize=7, loc="upper left")

    if title:
        fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.07)
    anim = FuncAnimation(fig, draw, frames=len(steps))
    path = Path(path)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path
