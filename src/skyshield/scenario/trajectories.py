"""Ground-truth drone trajectory generators (2D, fixed time step).

State convention: [x, y, vx, vy] in metres and m/s.
"""

from __future__ import annotations

import numpy as np


def waypoint_trajectory(
    waypoints: np.ndarray,
    speed: float,
    dt: float,
    accel_noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Fly through waypoints at constant speed, optionally with random manoeuvre noise.

    Returns an array of shape (T, 4) with rows [x, y, vx, vy].
    """
    waypoints = np.asarray(waypoints, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[1] != 2 or len(waypoints) < 2:
        raise ValueError("waypoints must have shape (N>=2, 2)")
    rng = rng or np.random.default_rng()

    pos = waypoints[0].copy()
    target_idx = 1
    heading = waypoints[1] - waypoints[0]
    heading /= np.linalg.norm(heading)
    vel = speed * heading
    states = [np.concatenate([pos, vel])]

    max_steps = int(1e6)
    for _ in range(max_steps):
        to_target = waypoints[target_idx] - pos
        dist = np.linalg.norm(to_target)
        if dist < speed * dt:
            target_idx += 1
            if target_idx >= len(waypoints):
                break
            continue
        desired = speed * to_target / dist
        # Steer toward the waypoint, then add random manoeuvre.
        vel = desired
        if accel_noise > 0:
            vel = vel + accel_noise * dt * rng.standard_normal(2)
            vel = speed * vel / np.linalg.norm(vel)
        pos = pos + vel * dt
        states.append(np.concatenate([pos, vel]))
    return np.array(states)


def loiter_trajectory(
    center: np.ndarray, radius: float, speed: float, dt: float, duration: float, phase: float = 0.0
) -> np.ndarray:
    """Circular loiter around a centre point."""
    center = np.asarray(center, dtype=float)
    omega = speed / radius
    t = np.arange(0.0, duration, dt)
    ang = phase + omega * t
    x = center[0] + radius * np.cos(ang)
    y = center[1] + radius * np.sin(ang)
    vx = -speed * np.sin(ang)
    vy = speed * np.cos(ang)
    return np.stack([x, y, vx, vy], axis=1)
