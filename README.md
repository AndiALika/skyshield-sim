# SkyShield-Sim

Software-only counter-UAS sensor network simulator: **detect, track, survive**.

Simulates drones approaching a protected site, a distributed sensor network observing them, and a fusion center tracking them. A later phase couples a jammable backhaul link to the tracker to quantify how communications attacks degrade situational awareness.

> Simulation and defensive analysis only.

## Status

Phase 0 complete: scenario generator, noisy radar model (range/bearing, Pd, clutter), and an Extended Kalman Filter tracking a single drone.

![Phase 0 tracking result](results/phase0_tracking.png)

## Quickstart

```bash
pip install -e .[dev]
pytest
python experiments/phase0_single_target.py
```

See `CLAUDE.md` for architecture, decisions and the roadmap.
