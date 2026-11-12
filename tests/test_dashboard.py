import json

import numpy as np
import pytest

from skyshield.dashboard.figures import SimParams, frame_traces, link_table, replay_figure
from skyshield.dashboard.replay import ellipse_params, save_gif
from skyshield.integration.network import FUSION, JAMMER_POS, RADARS, build_sim
from skyshield.scenario.multi_target import MultiTargetScenario


def short_log(waveform="plain", jammer="spot", n_steps=30):
    sim = build_sim(waveform, jammer, 50.0, (10.0, 20.0))
    sc = sim.scenario
    sim.scenario = MultiTargetScenario(dt=sc.dt, targets=sc.targets, n_steps=n_steps)
    return sim.run(np.random.default_rng(0), np.random.default_rng(1), record=True)


@pytest.fixture(scope="module")
def log():
    return short_log()


def test_ellipse_params_axis_aligned():
    w, h, a = ellipse_params(np.diag([100.0, 25.0]))
    assert (w, h) == pytest.approx((40.0, 20.0))  # 2 sigma each side
    assert abs(np.sin(np.radians(a))) == pytest.approx(0.0, abs=1e-9)


def test_frame_traces_keep_a_fixed_layout(log):
    shapes = {len(frame_traces(f)) for f in log.frames}
    assert shapes == {8}
    jammed = log.frames[15]
    assert jammed.jammer_on and not jammed.delivered.any()  # 50 dBm spot kills plain links
    tr = frame_traces(jammed)
    assert tr[0]["x"] == [] and len(tr[1]["x"]) == 3 * len(RADARS)  # all links drawn as lost


def test_replay_figure_is_valid_json_with_frames(log):
    fig = replay_figure(log, stride=5)
    assert len(fig["frames"]) == 6
    assert all(len(fr["data"]) == len(fr["traces"]) for fr in fig["frames"])
    json.dumps(fig)  # serialisable for st.plotly_chart
    with pytest.raises(ValueError):
        replay_figure(build_sim("plain", None, 0.0).run(np.random.default_rng(0),
                                                        np.random.default_rng(1)))


def test_link_table_js_falls_with_distance():
    rows = link_table(SimParams("barrage", 50.0, 0.0, 300.0, 0.9, 5.0, 0))
    snr = [r["SNR (dB)"] for r in rows]
    js = [r["J/S when on (dB)"] for r in rows]
    # Every link sees the same jammer power, so J/S rises exactly as SNR falls.
    assert np.allclose(np.array(js) + np.array(snr), js[0] + snr[0], atol=0.15)
    assert link_table(SimParams(None, 50.0, 0.0, 300.0, 0.9, 5.0, 0))[0]["J/S when on (dB)"] is None


def test_save_gif(tmp_path, log):
    out = save_gif({"a": log, "b": log}, tmp_path / "x.gif", RADARS, FUSION, JAMMER_POS,
                   steps=range(6), jam_window=(10.0, 20.0))
    assert out.exists() and out.stat().st_size > 1000
