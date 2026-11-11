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
| `dashboard` | FastAPI + Plotly/Leaflet live view |

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
| 1. Tracking core | 2-4 | Multi-sensor models, UKF, GNN association, multi-target eval |
| 2. RF detection | 5-6 | Dataset pipeline, CNN, robustness study, detections feed tracker |
| 3. Comms layer | 7-8 | Link model, jammer models, FH/DSSS, BER and PDR curves |
| 4. Integration | 9-10 | Closed loop: jamming -> packet loss -> tracking error |
| 5. Polish | 11-12 | Dashboard, GIFs, README, write-up |

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
- Run: `pip install -e .[dev]`, then `pytest` and `python experiments/phase0_single_target.py`.

## Next step

Phase 1: multi-target scenarios with clutter, GNN data association, track initiation/deletion, OSPA metric.
