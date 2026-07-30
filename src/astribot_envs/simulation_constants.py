#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025, Astribot Co., Ltd.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------

"""
File: simulation_constants.py
Brief: Central constants for simulation configuration
"""

# ==============================================================================
# Physics / Control Timing Configuration
# ==============================================================================
# Single source of truth for the frequencies: physics.{physics_hz, control_hz} in
# config/simulation.yaml. This module only holds the code-level defaults (the
# fallback) plus the derivation helper resolve_timing.
#
#   physics_hz  physics rate = 1/dt (contact accuracy + sensor sampling ceiling)
#   control_hz  control / env-loop rate (SDK command + state publish cadence;
#               the real robot supports 0-250Hz)
#   frame_skip  = round(physics_hz / control_hz) (derived, not configured)
#   sim_dt (the advance per env.step) = frame_skip * dt = 1/control_hz

# Code-level fallbacks, used only when the config supplies neither the rates nor
# the legacy dt/frame_skip. The shipped config/simulation.yaml sets 200/50, so
# these values are not what a normal run uses.
DEFAULT_PHYSICS_HZ = 200.0
DEFAULT_CONTROL_HZ = 100.0

# Legacy constants (kept for existing references; prefer resolve_timing).
DT_GENESIS = 1.0 / DEFAULT_PHYSICS_HZ  # 0.005
DT_MUJOCO_DEFAULT = 1.0 / DEFAULT_PHYSICS_HZ
FRAME_SKIP_GENESIS = round(DEFAULT_PHYSICS_HZ / DEFAULT_CONTROL_HZ)  # 2
FRAME_SKIP_MUJOCO = round(DEFAULT_PHYSICS_HZ / DEFAULT_CONTROL_HZ)  # 2

# Legacy constant equal to the fallback control rate, NOT the configured one.
# The pace lock derives its period from sim_dt, so never hardcode against this.
ENV_STEP_RATE_HZ = DEFAULT_CONTROL_HZ


def resolve_timing(physics_cfg):
    """Derive (dt, frame_skip, physics_hz, control_hz) from the physics config plus
    environment variables.

    Shared by both backends (MuJoCo and Genesis both call it) so their timing stays
    consistent. physics_hz and control_hz are the semantically meaningful values;
    frame_skip is derived from them and is never configured directly.

    Priority (high to low):
      1. env vars ASTRIBOT_PHYSICS_HZ / ASTRIBOT_CONTROL_HZ
      2. config physics.{physics_hz, control_hz} (from the simulation.yaml single source)
      3. legacy config physics.{dt, frame_skip} (backward compatibility)
      4. code defaults DEFAULT_PHYSICS_HZ / DEFAULT_CONTROL_HZ

    Args:
        physics_cfg: dict, the merged yaml `physics` section (may be empty).

    Returns:
        dict(dt=float, frame_skip=int, physics_hz=float, control_hz=float)
    """
    import os

    cfg = physics_cfg or {}

    physics_hz = os.getenv("ASTRIBOT_PHYSICS_HZ")
    if physics_hz is not None:
        physics_hz = float(physics_hz)
    elif cfg.get("physics_hz") is not None:
        physics_hz = float(cfg["physics_hz"])
    elif cfg.get("dt"):
        physics_hz = 1.0 / float(cfg["dt"])
    else:
        physics_hz = DEFAULT_PHYSICS_HZ

    control_hz = os.getenv("ASTRIBOT_CONTROL_HZ")
    if control_hz is not None:
        control_hz = float(control_hz)
    elif cfg.get("control_hz") is not None:
        control_hz = float(cfg["control_hz"])
    elif cfg.get("frame_skip"):
        control_hz = physics_hz / float(cfg["frame_skip"])
    else:
        control_hz = DEFAULT_CONTROL_HZ

    frame_skip = max(1, round(physics_hz / control_hz))
    dt = 1.0 / physics_hz
    return {
        "dt": dt,
        "frame_skip": frame_skip,
        "physics_hz": physics_hz,
        "control_hz": control_hz,
    }


# ==============================================================================
# Sensor Frequencies (Hz)
# ==============================================================================

# IMU sampling rate (matches real Livox MID-360 hardware)
IMU_FREQUENCY_HZ = 200

# LiDAR scan rate (Livox MID-360 specification)
LIDAR_FREQUENCY_HZ = 10

# Joint states publish rate
JOINT_STATES_FREQUENCY_HZ = 50

# Force/Torque sensor rate
FT_SENSOR_FREQUENCY_HZ = 50

# ==============================================================================
# LiDAR Configuration
# ==============================================================================

# Livox MID-360 24000-ray scan pattern (non-repeating)
LIDAR_RAY_COUNT = 24000

# Genesis LiDAR default resolution (ring × azimuth)
# Note: Genesis returns (200, 120, 3) for mid360 - ring × azimuth × xyz
LIDAR_GENESIS_DEFAULT_N_POINTS = (200, 120)

# ==============================================================================
# Real-Time Performance Thresholds
# ==============================================================================

# Real-time factor warning threshold (rtf < this value triggers warning)
RTF_WARNING_THRESHOLD = 0.95

# Real-time factor for "near real-time" performance
RTF_NEAR_REALTIME = 1.0

# ==============================================================================
# Camera Configuration
# ==============================================================================

# Default camera resolution
CAMERA_DEFAULT_WIDTH = 640
CAMERA_DEFAULT_HEIGHT = 480

# Default camera FPS
CAMERA_DEFAULT_FPS = 30


# ==============================================================================
# Genesis Async Viewer Pacing
# (docs/develop/realtime_async_render_refactor.md §4)
# ==============================================================================

# Genesis offline viewer refresh rate (Hz) when not real-time paced.
GENESIS_FULL_REFRESH_RATE = 60


def genesis_viewer_pacing(sim_profile, human_render_every, control_hz=None):
    """Map sim_profile → Genesis ViewerOptions (refresh_rate, realtime_factor).

    The Genesis viewer runs off the physics thread via run_in_thread=True. However
    **realtime_factor must never be set to a non-None value**: the tail of
    viewer.update() calls realtime_pacer.sleep() (genesis/vis/viewer.py:234-236),
    which sleeps *on the physics main thread* to hit the viewer's cadence, and its
    `with self.lock` contends for render_lock against the render thread. Measured,
    this pinned the realtime profile's RTF at ~0.7 (main thread dragged down by the
    viewer pacer plus lock contention) — it was not drawing cost. So realtime_factor
    is always None (the viewer never paces the main thread); real-time pacing, when
    needed, is done by the physics loop itself.

    refresh_rate controls the *render thread's* refresh rate (on-screen FPS).
    Measured (2026-07-16): with run_in_thread the render thread keeps hitting the
    GPU at refresh_rate and contends with the physics kernels for GPU/render_lock —
    with a window open, scene.step goes from 11.5ms (headless) to 20ms. The higher
    refresh_rate, the worse the contention. So refresh_rate is set to the actual
    frame rate = control_hz / human_render_every (not the stale ENV_STEP_RATE_HZ, so
    the render thread does no excess refreshing). Falls back to DEFAULT_CONTROL_HZ
    when control_hz is not supplied.

    - realtime: refresh_rate = frame rate; realtime_factor=None (**never let the
      viewer sleep on the main thread**).
    - custom/other: refresh_rate=60, realtime_factor=None (run as fast as possible).

    Returns:
        (refresh_rate:int, realtime_factor:None)
    """
    if sim_profile == "realtime":
        every = max(1, int(human_render_every))
        hz = control_hz if control_hz else DEFAULT_CONTROL_HZ
        refresh_rate = max(1, round(hz / every))
        return refresh_rate, None
    return GENESIS_FULL_REFRESH_RATE, None
