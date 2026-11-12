import numpy as np
import pytest

from skyshield.comms.jammers import BarrageJammer, PartialBandJammer, SpotJammer, SweepJammer
from skyshield.comms.link import BackhaulLink, DelayLine
from skyshield.comms.modulation import (
    BCH_63_45,
    HAMMING_7_4,
    UNCODED,
    BlockCode,
    PacketSpec,
    ber_bpsk,
    db2lin,
)
from skyshield.comms.simulate import bpsk_ber, dsss_tone_ber, fh_partial_band_per
from skyshield.comms.waveforms import DSSS, FrequencyHopping, Plain

N = 64


# ----- modulation / coding ---------------------------------------------------------------------


def test_bpsk_ber_textbook_values():
    assert ber_bpsk(db2lin(0.0)) == pytest.approx(0.0786, abs=1e-4)
    assert ber_bpsk(db2lin(9.6)) == pytest.approx(1e-5, rel=0.1)
    assert ber_bpsk(0.0) == pytest.approx(0.5)


def test_block_code_error_probability():
    assert UNCODED.codeword_error(0.01) == pytest.approx(0.01)
    # Hamming(7,4) fails with 2+ errors: 1 - (1-p)^7 - 7 p (1-p)^6.
    p = 0.01
    expected = 1 - (1 - p) ** 7 - 7 * p * (1 - p) ** 6
    assert HAMMING_7_4.codeword_error(p) == pytest.approx(expected)
    with pytest.raises(ValueError):
        BlockCode(4, 7, 1)


def test_packet_per_monotone_and_coding_gain():
    sinr = db2lin(np.arange(0, 15, 0.5))
    unc = PacketSpec(512, UNCODED).per(sinr)
    bch = PacketSpec(512, BCH_63_45).per(sinr)
    assert np.all(np.diff(unc) <= 0)
    # SINR needed for PER = 1e-2: coding should save a few dB despite the rate loss.
    def need(per):
        return sinr[np.argmax(per < 1e-2)]
    gain_db = 10 * np.log10(need(unc) / need(bch))
    assert 1.0 < gain_db < 4.0
    assert PacketSpec(512, BCH_63_45).coded_bits == 12 * 63


# ----- jammers ---------------------------------------------------------------------------------


@pytest.mark.parametrize("jam", [
    BarrageJammer(np.zeros(2)),
    PartialBandJammer(np.zeros(2), rho=0.2),
    SpotJammer(np.zeros(2), channel=5),
    SweepJammer(np.zeros(2), width_channels=4),
])
def test_jammer_fractions_sum_to_one(jam):
    f = jam.channel_fractions(N, 0.0123, np.random.default_rng(0))
    assert f.shape == (N,) and f.sum() == pytest.approx(1.0) and np.all(f >= 0)


def test_partial_band_covers_rho_fraction():
    f = PartialBandJammer(np.zeros(2), rho=0.25).channel_fractions(N, 0, np.random.default_rng(1))
    assert np.count_nonzero(f) == 16
    with pytest.raises(ValueError):
        PartialBandJammer(np.zeros(2), rho=0.0)


def test_sweep_moves_with_time():
    j = SweepJammer(np.zeros(2), width_channels=2, period_s=1.0)
    a = np.nonzero(j.channel_fractions(N, 0.0, None))[0]
    b = np.nonzero(j.channel_fractions(N, 0.5, None))[0]
    assert list(a) == [0, 1] and list(b) == [32, 33]


def test_jammer_received_power_falls_with_distance():
    j = BarrageJammer(np.zeros(2), eirp_dbm=30.0)
    assert j.received_mw(np.array([100.0, 0])) > j.received_mw(np.array([1000.0, 0]))


# ----- waveforms -------------------------------------------------------------------------------


def test_waveform_interference_rules():
    j = np.zeros(N)
    j[3] = 64.0
    assert Plain(channel=3).interference_options(j) == pytest.approx([64.0])
    assert Plain(channel=4).interference_options(j) == pytest.approx([0.0])
    fh = FrequencyHopping().interference_options(j)
    assert len(fh) == N and fh.mean() == pytest.approx(1.0)
    assert DSSS().interference_options(j) == pytest.approx([1.0])  # 64 / Gp


# ----- link ------------------------------------------------------------------------------------


def make_link(wf, **kw):
    return BackhaulLink(np.array([1000.0, 0.0]), np.zeros(2), waveform=wf, **kw)


def test_link_budget():
    link = make_link(Plain())
    assert link.noise_mw == pytest.approx(10 ** (-108.0 / 10), rel=1e-3)
    assert link.snr_db == pytest.approx(20 - (40.2 + 75.0) + 108.0, abs=0.2)


def test_no_jammer_gives_near_perfect_delivery():
    rng = np.random.default_rng(0)
    for wf in (Plain(), FrequencyHopping(), DSSS()):
        assert make_link(wf).expected_pdr([], rng) > 0.999


def test_spot_jammer_kills_plain_but_not_fh_or_dsss():
    rng = np.random.default_rng(1)
    jam = [SpotJammer(np.array([1500.0, 500.0]), eirp_dbm=30.0, channel=0)]
    plain = make_link(Plain(channel=0)).expected_pdr(jam, rng)
    fh = make_link(FrequencyHopping()).expected_pdr(jam, rng)
    ds = make_link(DSSS()).expected_pdr(jam, rng)
    assert plain < 0.01
    assert fh > 0.99  # 1/64 hops hit; retries clean that up
    assert ds > 0.99


def test_retries_help_fh_but_not_spot_jammed_plain():
    rng = np.random.default_rng(2)
    jam = [PartialBandJammer(np.array([1500.0, 500.0]), eirp_dbm=40.0, rho=0.3)]
    fh0 = make_link(FrequencyHopping(), max_retries=0).expected_pdr(jam, rng)
    fh3 = make_link(FrequencyHopping(), max_retries=3).expected_pdr(jam, rng)
    assert fh3 > fh0 + 0.1
    spot = [SpotJammer(np.array([1500.0, 500.0]), eirp_dbm=30.0)]
    assert make_link(Plain(), max_retries=5).expected_pdr(spot, rng) < 0.01


def test_send_matches_expected_pdr_and_reports_delay():
    rng = np.random.default_rng(3)
    jam = [PartialBandJammer(np.array([1500.0, 500.0]), eirp_dbm=40.0, rho=0.3)]
    link = make_link(FrequencyHopping(), max_retries=1)
    sent = [link.send(0.0, jam, rng) for _ in range(4000)]
    pdr = np.mean([d.delivered for d in sent])
    assert pdr == pytest.approx(link.expected_pdr(jam, rng, n_states=500), abs=0.03)
    ok = [d for d in sent if d.delivered]
    assert all(d.delay_s == pytest.approx(link.base_latency_s + d.attempts * link.attempt_s)
               for d in ok)
    assert all(not np.isfinite(d.delay_s) for d in sent if not d.delivered)


def test_delay_line_orders_and_drops():
    dl = DelayLine()
    dl.push("b", 2.0)
    dl.push("a", 1.0)
    dl.push("lost", np.inf)
    assert len(dl) == 2
    assert dl.pop_ready(0.5) == []
    assert dl.pop_ready(2.0) == ["a", "b"]
    assert len(dl) == 0


# ----- Monte Carlo agrees with the analytic models ------------------------------------------------


def test_mc_bpsk_matches_theory():
    rng = np.random.default_rng(4)
    n = 400_000
    for ebn0 in (0.0, 4.0, 7.0):
        sim = bpsk_ber(ebn0, n, rng)
        p = float(ber_bpsk(db2lin(ebn0)))
        sigma = np.sqrt(p * (1 - p) / n)  # binomial standard error of the estimate
        assert abs(sim - p) < 4 * sigma, (ebn0, sim, p)


def test_mc_dsss_tone_matches_gaussian_approximation():
    rng = np.random.default_rng(5)
    gp, ebn0 = 64, 10.0
    for js in (10.0, 15.0):
        sim = dsss_tone_ber(ebn0, js, gp, 60_000, rng)
        theory = float(ber_bpsk(1.0 / (1 / db2lin(ebn0) + db2lin(js) / gp)))
        assert sim == pytest.approx(theory, rel=0.25), (js, sim, theory)


def test_mc_fh_partial_band_matches_analytic():
    rng = np.random.default_rng(6)
    ebn0, ebj0, rho, bits = 12.0, 10.0, 0.1, 256
    sim = fh_partial_band_per(ebn0, ebj0, rho, N, bits, 40_000, rng)
    k = round(rho * N)
    pkt = PacketSpec(bits, UNCODED)
    hit_sinr = 1 / (1 / db2lin(ebn0) + N / (k * db2lin(ebj0)))
    theory = k / N * pkt.per(hit_sinr) + (1 - k / N) * pkt.per(db2lin(ebn0))
    assert sim == pytest.approx(float(theory), abs=0.01)


def test_exact_partial_band_pdr_matches_sampled():
    rng = np.random.default_rng(7)
    jam = PartialBandJammer(np.array([1500.0, 500.0]), eirp_dbm=40.0, rho=0.25)
    for wf in (Plain(), FrequencyHopping(), DSSS()):
        link = make_link(wf, max_retries=2)
        exact = link.expected_pdr([jam], rng)
        # Sampled estimate via the generic path (two stochastic jammers' worth: add a silent one).
        silent = PartialBandJammer(np.array([1e7, 0.0]), eirp_dbm=-200.0, rho=0.5)
        sampled = link.expected_pdr([jam, silent], rng, n_states=3000)
        assert exact == pytest.approx(sampled, abs=0.02), wf
    assert len(jam.all_states(N)) == N


def test_bernoulli_link():
    from skyshield.comms.link import BernoulliLink

    rng = np.random.default_rng(8)
    link = BernoulliLink(0.7, latency_s=0.003)
    d = [link.send(0.0, [], rng) for _ in range(5000)]
    assert np.mean([x.delivered for x in d]) == pytest.approx(0.7, abs=0.02)
    assert all(x.delay_s == 0.003 for x in d if x.delivered)
    with pytest.raises(ValueError):
        BernoulliLink(1.5)
