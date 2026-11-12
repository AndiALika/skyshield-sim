# SkyShield-Sim

Software-only counter-UAS sensor network simulator: **detect, track, survive**.
Portfolio project (telecom engineering MSc). Simulation and defensive analysis only. No hardware, no offensive tooling.

## Concept

A protected site has distributed sensors (radar, RF direction finding, acoustic). Drones approach along simulated trajectories. The system must:

1. Detect them (RF CNN classifier, bearing sensors, radar model)
2. Track them (multi-sensor, multi-target fusion)
3. Survive interference (sensor-to-fusion links degrade under jamming, which degrades tracking)

The coupling in step 3 is the differentiator: link quality determines measurement delivery, so we can quantify how a comms-layer attack degrades situational awareness.

## Architecture

```
scenario -> ground truth -> sensors (RF/radar/acoustic) -> comms backhaul (jammer, FH/DSSS, loss, delay) -> tracker -> evaluation/dashboard
```

| Package | Responsibility |
|---|---|
| `scenario` | Trajectories (waypoint, loiter, swarm, evasive), sensor layouts, clutter |
| `sensors` | Measurement models with noise, Pd, false alarms |
| `rf_detection` | IQ/spectrogram generation, CNN classifier, DF bearing |
| `comms` | Backhaul link model, jammer models, FH/DSSS, optional RL agent |
| `tracking` | Association (GNN, JPDA), EKF/UKF/IMM, track management |
| `evaluation` | RMSE, NEES, OSPA, continuity, ROC, experiment runners |
| `dashboard` | Streamlit app (`streamlit_app.py`) + Plotly replay, matplotlib GIFs |

## Locked decisions (change only deliberately)

- 2D first, 3D as an extension
- Fixed-step time with a delay buffer for the comms layer
- RF data: public datasets (DroneRF, DroneDetect) plus synthetic IQ
- Filters implemented by hand, cross-validated against Stone Soup
- Python 3.11, NumPy/SciPy, PyTorch for ML, pytest + ruff

## Roadmap

| Phase | Weeks | Deliverable |
|---|---|---|
| 0. Setup | 1 | Repo, scenario generator, EKF + one radar (DONE in this scaffold) |
| 1. Tracking core | 2-4 | Multi-sensor models, UKF, GNN association, multi-target eval (GNN, track management, OSPA DONE) |
| 2. RF detection | 5-6 | Dataset pipeline, CNN, robustness study, detections feed tracker (synthetic pipeline, CNN, robustness study, DF sensor DONE) |
| 3. Comms layer | 7-8 | Link model, jammer models, FH/DSSS, BER and PDR curves (DONE) |
| 4. Integration | 9-10 | Closed loop: jamming -> packet loss -> tracking error (DONE, radar-only) |
| 5. Polish | 11-12 | Dashboard, GIFs, README, write-up (DONE) |

MVP = Phases 0, 1, 4 with a simplified link model. Stretch: IMM, JPDA, RL adaptive hopping, swarms, 3D terrain.

## Key experiments (headline results)

1. Detector accuracy vs SNR, with/without Wi-Fi/Bluetooth interference, held-out drone models
2. Tracker comparison: GNN vs JPDA, EKF vs UKF vs IMM
3. Sensor placement study
4. Tracking OSPA vs jammer power for plain vs FH vs DSSS links
5. Closed-loop resilience curve (flagship plot)

## Conventions

- State vector: `[x, y, vx, vy]` (metres, m/s). Measurements: `[range, bearing]` (m, rad), bearing from +x axis.
- Every experiment is a script in `experiments/` that writes figures/metrics to `results/`.
- Seed every RNG. Experiments must be reproducible.
- Each module gets tests in `tests/`. Filters are validated with NEES consistency checks.
- Run: `pip install -e .[dev]`, then `pytest` and `python experiments/phase0_single_target.py`
  (or `phase1_multi_target.py`, ~3 min; `phase2_rf_detection.py`, ~20 min on CPU, needs `.[ml]`;
  `phase5_replay_gifs.py`, ~1 min). Dashboard: `pip install -e .[dashboard]`, `streamlit run streamlit_app.py`.

## Tracking design (Phase 1)

- `tracking/gnn.py`: chi-square gate (default 0.99) + Hungarian assignment on cost d2 + ln|S|,
  with one dummy "miss" column per track (cost = gate + ln|S|).
- `tracking/manager.py`: `MultiTargetTracker`. Two-stage association (confirmed tracks first, then
  tentative). One-point initiation, M/N confirmation (default 4/5), deletion after `max_misses`
  consecutive misses or when position std exceeds `max_pos_std`. Only confirmed tracks are output.
- 4/5 was chosen over 3/4 by measurement: 3/4 gave ~0.17 false confirmed tracks per scan at
  20 clutter/scan with no targets; 4/5 gave 0, with the same OSPA on the crossing scenario.
- `evaluation/metrics.py`: `ospa(X, Y, c, p)` returns distance plus localisation/cardinality parts.
  Experiments use c = 100 m, p = 2.

Known limitation: the constant-velocity EKF loses a target in a sharp (90 degree, 18 m/s) turn and
re-initiates, so one target gets two track IDs. IMM is the planned fix.

## RF detection design (Phase 2)

- `rf_detection/signals.py`: synthetic complex baseband, 10 MS/s, 65536 samples (6.55 ms) per
  capture. Emitters: drone control link (GFSK or chirp frequency hopper), drone video (wideband
  OFDM-like), Wi-Fi bursts, Bluetooth hopper. Bandwidths are scaled to the 10 MHz slice.
  Drone models A-D are for training. Held out: E (same GFSK family, new parameters) and
  F (chirp modulation, never seen in training). All profiles are generic, not real products.
- **SNR is in-band**: signal power over noise power in the emitter's own bandwidth (noise is
  unit power over FS). This is what a link budget gives, so the detector's Pd(SNR) curve plugs
  straight into `sensors/rf_df.py`.
- `rf_detection/features.py`: 64 x 128 log spectrogram, dB relative to per-capture median
  (removes receiver gain). Also two classical baselines from the raw IQ: wideband energy and
  time-frequency peak.
- `rf_detection/cnn.py`: 4 conv blocks + global avg+max pooling, binary output. Uses GroupNorm,
  not BatchNorm: with short training runs BatchNorm's running statistics lagged the weights
  (eval-mode accuracy 0.69 vs 1.0 for the same weights).
- `sensors/rf_df.py`: passive DF sensor, bearing only. EIRP -> log-distance path loss ->
  in-band SNR -> Pd from the CNN's measured curve -> bearing noise ~ 1/sqrt(SNR), clipped.
- Detection thresholds are set once at Pfa = 1 % on the "mixed" environment and then held
  fixed for every test condition, so interference shows up as Pd loss and Pfa change.
- Public datasets (DroneRF, DroneDetect) are not wired in yet: they need a multi-GB download.
  `data/` is gitignored and holds cached synthetic datasets.

## Comms design (Phase 3)

- Band: `n_channels` (64) channels, each as wide as the 1 Mbit/s BPSK signal. Plain = one fixed
  channel; FH = slow hopping, one packet attempt per hop; DSSS = all channels, Gp = n_channels.
  This keeps bit rate and total bandwidth equal, so the comparison is fair.
- `comms/jammers.py`: a jammer is EIRP + position + `channel_fractions(n, t, rng)` (sums to 1).
  Barrage, partial-band (rho; `all_states` enables exact averaging), spot, sweep. Interference
  models only: received power shapes, no signal generation.
- `comms/waveforms.py`: `interference_options(j_ch)` = equally likely interference powers per
  attempt (FH: every channel; DSSS: sum / Gp; plain: its channel).
- `comms/modulation.py`: BPSK BER, (n, k, t) block code, packet PER. SINR is per info bit and
  coded bits get rate * SINR.
- `comms/link.py`: `BackhaulLink` (budget, `send` with stop-and-wait ARQ -> `Delivery`,
  `expected_pdr`) and `DelayLine`. The jammer state is held for all retries of one packet and FH
  redraws its hop per attempt, so retries help FH but not a plain link that is spot-jammed.
- `comms/simulate.py`: baseband Monte Carlo used to validate the analytic models.
- Backhaul links use path-loss exponent 2 (masts, line of sight); jammers default to 2.5.

## Integration design (Phase 4)

- `tracking/manager.py`: the tracker now runs as `begin_step(t)` -> `update(report)` per
  delivered sensor report -> `end_step()`. Track management happens once per step: a track is
  hit if any report updated it, so a target seen by one sensor is not "missed" by the others.
  Leftover measurements spawn tentative tracks immediately, so a second sensor's detection of
  the same new target updates that track. `step()` wraps the three calls; Phase 1 results are
  bit-identical after the refactor.
- `integration/closed_loop.py`: `ClosedLoopSim`. One report packet per radar scan
  (96-bit header + 48 bits per detection, BCH(63,45)) -> `BackhaulLink.send` -> `DelayLine` ->
  fusion at t + 0.2 s. Reports measured before the current step are stale and dropped (no OOSM
  retrodiction). Sensors and comms use separate RNGs (common random numbers across link configs).
- `comms/link.py`: `BernoulliLink(pdr)` is the simplified i.i.d.-loss link used for OSPA vs PDR.
- The network geometry is deliberately asymmetric (links of 1.1 / 2.3 / 3.2 km). With equal
  links, all of them failed at the same jammer power and the resilience curve was a single step.
- Finding: tracking quality depends on the loss structure, not only on PDR. Whole-sensor loss
  hurts far less than i.i.d. loss at the same PDR.

## Polish design (Phase 5)

- `integration/network.py`: the reference network (radar, fusion and jammer positions, waveform
  and jammer factories, `build_sim`). Shared by the Phase 4 experiment, the GIFs and the
  dashboard; Phase 4 results are unchanged.
- `ClosedLoopSim.run(..., record=True)` keeps a `Frame` per step (truth, detections per sensor
  with clutter flags, delivered flags, tracks with position covariance, jammer state). It draws
  no extra random numbers, so the series are identical with or without recording (tested).
- `dashboard/replay.py`: matplotlib GIFs (`experiments/phase5_replay_gifs.py`).
  `dashboard/figures.py`: simulation helpers and the Plotly replay figure, no Streamlit import,
  so it is unit-tested.
- Dashboard is Streamlit + Plotly, not FastAPI + Leaflet as first planned: the map is local metres
  (no geo tiles needed), and Streamlit gives an interactive what-if app in one file that can be
  hosted on Streamlit Community Cloud. Plotly is used for the map only because it has a built-in
  play button and slider for frame animation; the time series use native Streamlit charts.
- The app runs the four simulations sequentially: a process pool spawned from a Streamlit script
  re-imports the script in every worker (it re-ran the app).
- `docs/writeup.md`: the technical write-up (models, validation table, results, limitations).

## Next step

All roadmap phases are done. Optional: RF DF bearings into the closed loop (bearing-only EKF
updates), IMM for the turn re-initiation, UKF, JPDA, ID-switch metric, Stone Soup cross-check,
DroneRF real-data check, Streamlit Community Cloud deployment.
