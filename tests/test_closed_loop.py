import numpy as np

from skyshield.comms.jammers import BarrageJammer, SpotJammer
from skyshield.comms.link import BackhaulLink
from skyshield.comms.waveforms import FrequencyHopping, Plain
from skyshield.integration.closed_loop import (
    BITS_PER_DETECTION,
    HEADER_BITS,
    ClosedLoopConfig,
    ClosedLoopSim,
    JammerSchedule,
    SensorNode,
    report_packet,
)
from skyshield.scenario.multi_target import MultiTargetScenario, TargetTruth, crossing_scenario
from skyshield.sensors.radar import Radar

FUSION = np.zeros(2)
RADAR_POS = [np.array([-1500.0, -500.0]), np.array([1500.0, -500.0])]


def scenario(n_steps=120):
    sc = crossing_scenario(rng=np.random.default_rng(0))
    return MultiTargetScenario(dt=sc.dt, targets=sc.targets, n_steps=n_steps)


def nodes(waveform=None):
    out = []
    for i, p in enumerate(RADAR_POS):
        link = None if waveform is None else BackhaulLink(p, FUSION, waveform=waveform,
                                                          path_loss_exp=2.0)
        out.append(SensorNode(i, Radar(position=p, pd=0.9, clutter_rate=3.0), link))
    return out


def run(sim, seed=1):
    return sim.run(np.random.default_rng(seed), np.random.default_rng(seed + 100))


def test_report_packet_grows_with_detections():
    assert report_packet(0).info_bits == HEADER_BITS
    assert report_packet(5).info_bits == HEADER_BITS + 5 * BITS_PER_DETECTION


def test_jammer_schedule_window():
    j = [BarrageJammer(np.zeros(2))]
    s = JammerSchedule(j, t_on=10.0, t_off=20.0)
    assert s.active(9.9) == [] and s.active(10.0) == j and s.active(20.0) == []


def test_wired_network_tracks_well():
    log = run(ClosedLoopSim(scenario(), nodes(None)))
    assert log.pdr == 1.0
    assert log.stale == 0
    assert log.mean_ospa < 30.0
    assert log.delivered.shape == (120, 2)


def test_unjammed_radio_links_match_wired_exactly():
    wired = run(ClosedLoopSim(scenario(), nodes(None)))
    radio = run(ClosedLoopSim(scenario(), nodes(FrequencyHopping())))
    assert radio.pdr == 1.0
    assert np.array_equal(wired.ospa, radio.ospa)  # same sensor RNG stream -> same tracks
    assert 0.5e-3 < np.mean(radio.latencies_s) < 5e-3


def test_total_jamming_blinds_the_tracker():
    jam = JammerSchedule([BarrageJammer(np.array([0.0, 500.0]), eirp_dbm=90.0)])
    log = run(ClosedLoopSim(scenario(), nodes(Plain()), jam))
    assert log.pdr == 0.0
    assert np.all(log.n_est == 0)
    alive = log.n_true > 0
    assert np.allclose(log.ospa[alive], 100.0)


def test_jamming_burst_degrades_then_recovers():
    jam = JammerSchedule([SpotJammer(np.array([0.0, 800.0]), eirp_dbm=60.0, channel=0)],
                         t_on=40.0, t_off=80.0)
    log = run(ClosedLoopSim(scenario(), nodes(Plain(channel=0)), jam))
    before, during, after = log.ospa[20:40], log.ospa[50:80], log.ospa[95:120]
    assert during.mean() > 3 * before.mean()
    assert after.mean() < 0.5 * during.mean()
    assert not log.delivered[45:80].any() and log.delivered[:40].all()


def test_fh_survives_spot_jammer_that_kills_plain():
    jam = JammerSchedule([SpotJammer(np.array([0.0, 800.0]), eirp_dbm=60.0, channel=0)])
    plain = run(ClosedLoopSim(scenario(), nodes(Plain(channel=0)), jam))
    fh = run(ClosedLoopSim(scenario(), nodes(FrequencyHopping()), jam))
    assert plain.pdr < 0.05 and fh.pdr > 0.95
    assert fh.mean_ospa < 0.5 * plain.mean_ospa


def test_reports_after_fusion_deadline_are_stale():
    cfg = ClosedLoopConfig(fusion_latency_s=1e-4)  # shorter than any link delay
    log = run(ClosedLoopSim(scenario(), nodes(FrequencyHopping()), config=cfg))
    assert not log.delivered.any()
    # Every report except the last step's (still in the buffer at the end) went stale.
    assert log.stale == 2 * (120 - 1)


def test_summary_keys():
    s = run(ClosedLoopSim(scenario(40), nodes(None))).summary()
    assert {"mean_ospa", "pdr", "pdr_per_sensor", "stale_reports", "mean_latency_ms"} <= set(s)
    assert s["mean_latency_ms"] is None  # wired: no radio latency recorded


def test_single_target_truth_helper_runs():
    sc = MultiTargetScenario(1.0, [TargetTruth(0, 0, np.tile([500.0, 500.0, 0, 0], (30, 1)))])
    log = run(ClosedLoopSim(sc, nodes(None)))
    assert log.n_true.max() == 1 and log.n_est[-1] == 1


def test_recording_does_not_change_results():
    sim = ClosedLoopSim(scenario(60), nodes(Plain()))
    a = run(sim)
    b = sim.run(np.random.default_rng(1), np.random.default_rng(101), record=True)
    assert a.frames is None and len(b.frames) == 60
    np.testing.assert_array_equal(a.ospa, b.ospa)
    f = b.frames[30]
    assert f.truth_xy.shape == (len(f.truth_ids), 2)
    assert f.track_xy.shape == (len(f.track_ids), 2) and f.track_cov.shape[1:] == (2, 2)
    assert len(f.det_xy) == 2 and f.ospa == b.ospa[30]
    np.testing.assert_array_equal(f.delivered, b.delivered[30])


def test_reference_network_matches_phase4_geometry():
    from skyshield.integration.network import RADARS, build_sim

    sim = build_sim("plain", "spot", 50.0, (10.0, 20.0))
    assert len(sim.nodes) == len(RADARS)
    d = [n.link.distance_m for n in sim.nodes]
    assert np.allclose(sorted(d), [1077.0, 2341.0, 3225.0], atol=2.0)
    assert sim.jamming.active(15.0) and not sim.jamming.active(25.0)
