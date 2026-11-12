"""SkyShield-Sim dashboard: run the closed loop with your own jammer and watch tracking degrade.

Run:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from skyshield.dashboard.figures import LINK_TYPES, SimParams, link_table, replay_figure, run_all
from skyshield.integration.network import JAMMERS

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

st.set_page_config(page_title="SkyShield-Sim", page_icon=":material/radar:", layout="wide")


@st.cache_data(max_entries=8, show_spinner=False)
def simulate(p: SimParams) -> dict:
    # Sequential on purpose: worker processes spawned from a Streamlit script re-import the
    # script itself, so a process pool here would re-run the app in every worker.
    return run_all(p, parallel=False)


@st.cache_data(max_entries=16, show_spinner=False)
def replay(p: SimParams, link: str, stride: int) -> dict:
    return replay_figure(simulate(p)[link], stride=stride)


# ----- sidebar ---------------------------------------------------------------------------------

with st.sidebar:
    st.header("Scenario")
    with st.form("params"):
        jammer = st.selectbox("Jammer waveform", ["none", *JAMMERS], index=1, key="jammer")
        eirp = st.slider("Jammer EIRP (dBm)", 30, 70, 50, step=2, key="eirp",
                         help="Ground jammer 3.5 km from the fusion centre.")
        window = st.slider("Jammer on (s)", 0, 300, (100, 200), step=10, key="window")
        pd_ = st.slider("Radar Pd", 0.5, 1.0, 0.9, step=0.05, key="pd")
        clutter = st.slider("Clutter (false alarms per scan per radar)", 0.0, 20.0, 5.0, step=1.0,
                            key="clutter")
        seed = st.number_input("Monte Carlo seed", 0, 999, 0, key="seed")
        st.form_submit_button("Run simulation", icon=":material/play_arrow:", width="stretch",
                              key="run")
    st.caption("Each run simulates 300 s for plain, FH and DSSS backhaul plus an unjammed "
               "reference. All four see the same drones, detections and clutter.")

params = SimParams(None if jammer == "none" else jammer, float(eirp), float(window[0]),
                   float(window[1]), float(pd_), float(clutter), int(seed))

# ----- header ----------------------------------------------------------------------------------

st.title("SkyShield-Sim")
st.caption("Counter-UAS sensor network: three radars report to a fusion centre over a radio "
           "backhaul. A jammer attacks the backhaul; lost reports degrade tracking. "
           "Simulation only.")
view = st.segmented_control("View", ["Live simulation", "Monte Carlo study"],
                            default="Live simulation", required=True, key="view",
                            label_visibility="collapsed")

if view == "Monte Carlo study":
    st.subheader("Resilience: tracking error vs jammer power")
    st.markdown("Precomputed by `experiments/phase4_closed_loop.py` (6 Monte Carlo runs per "
                "point; the jammer picks the most damaging waveform for each link type).")
    metrics_path = RESULTS / "phase4_metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        with st.container(horizontal=True):
            for wf, r in m["resilience"].items():
                v = r["eirp_for_2x_ospa_dbm"]
                st.metric(f"{wf}: EIRP that doubles OSPA", f"{v:g} dBm" if v else "> 70 dBm",
                          border=True)
    for name, alt in [("phase4_resilience.png", "OSPA and PDR vs jammer EIRP per link type"),
                      ("phase4_pdr_vs_ospa.png", "OSPA vs packet delivery ratio"),
                      ("phase4_ospa_vs_eirp.png", "OSPA vs EIRP for each jammer waveform")]:
        if (RESULTS / name).exists():
            st.image(str(RESULTS / name), alt=alt)
    st.stop()

with st.spinner("Simulating 4 × 300 s closed loop…"):
    logs = simulate(params)

# ----- KPIs ------------------------------------------------------------------------------------

t = np.array([f.t for f in logs["no jamming"].frames])
on = (t >= params.t_on) & (t < params.t_off) if params.jammer else np.zeros_like(t, bool)
base = logs["no jamming"]


def during(series: np.ndarray) -> float:
    return float(series[on].mean()) if on.any() else float(series.mean())


label = "while jammed" if on.any() else "whole run"
with st.grid(4, gap="small"):
    for wf in LINK_TYPES:
        d = during(logs[wf].ospa) - during(base.ospa)
        st.metric(f"{wf} backhaul: OSPA {label}", f"{during(logs[wf].ospa):.1f} m",
                  0 if abs(d) < 0.05 else f"{d:+.1f} m", delta_color="inverse",
                  delta_description="vs no jamming", border=True, height="stretch")
    st.metric(f"No jamming: OSPA {label}", f"{during(base.ospa):.1f} m", 0,
              delta_description="reference", border=True, height="stretch")
    for wf in LINK_TYPES:
        st.metric(f"{wf}: reports fused {label}", f"{during(logs[wf].delivered.mean(1)):.0%}",
                  border=True, height="stretch")
    st.metric("Stale reports (all runs)", sum(lg.stale for lg in logs.values()), border=True,
              height="stretch")

# ----- replay + time series ----------------------------------------------------------------------

left, right = st.columns([3, 2])
with left, st.container(border=True):
    st.markdown("**Replay**")
    with st.container(horizontal=True, vertical_alignment="bottom"):
        link = st.segmented_control("Backhaul", [*LINK_TYPES, "no jamming"], default="plain",
                                    required=True,
                                    key="link")
        stride = st.segmented_control("Frame step (s)", [1, 2, 5], default=2, required=True,
                                     key="stride")
    fig = replay(params, link or "plain", stride or 2)
    st.plotly_chart(fig, key="map", config={"displaylogo": False})
    st.caption("Orange × = drone (truth) · purple = radars and fusion centre · green dot + "
               "ellipse = confirmed track with 2σ · blue = radar detection that reached the "
               "tracker · red = detection lost on the backhaul.")

with right:
    with st.container(border=True):
        st.markdown("**OSPA over time** (c = 100 m: no track picture)")
        df = pd.DataFrame({k: v.ospa for k, v in logs.items()}, index=pd.Index(t, name="t (s)"))
        st.line_chart(df, y_label="OSPA (m)", alt="OSPA over time for each backhaul type")
    with st.container(border=True):
        st.markdown("**Reports fused** (fraction of the 3 radars, 10 s average)")
        dl = pd.DataFrame({k: pd.Series(logs[k].delivered.mean(1)).rolling(10, 1).mean().values
                           for k in LINK_TYPES}, index=pd.Index(t, name="t (s)"))
        st.line_chart(dl, y_label="fraction fused", alt="Delivered reports over time")
    with st.container(border=True):
        st.markdown("**Backhaul link budget**")
        st.dataframe(pd.DataFrame(link_table(params)), hide_index=True,
                     alt="Per-radar backhaul length, SNR and jammer-to-signal ratio")
