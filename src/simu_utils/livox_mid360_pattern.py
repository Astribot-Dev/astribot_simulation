#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025, Astribot Co., Ltd.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------
# Author: simu dev
# -----------------------------------------------------------------------------

"""
LivoxMid360Pattern — a Genesis `RaycastPattern` that reproduces the real
Livox MID-360 non-repeating scan on the Genesis backend.

Why this exists:
    Genesis 1.x ships only a uniform `SphericalPattern` (angular grid), which
    is NOT the MID-360 pattern — it also collapsed the vertical FOV to a thin
    downward band, so tall structures were never hit. The MuJoCo backend
    already gets the true pattern from `mujoco_lidar.scan_gen.LivoxGenerator`
    (the Livox-provided `mid360.npy` table of 800000 (azimuth, elevation)
    angles). This class feeds the SAME table into Genesis so both backends
    scan identically.

Design (mirrors the ctu-mrs / fratopa Gazebo Mid360 plugin):
    The Gazebo plugin loads a CSV of fixed (time, azimuth, zenith) angles and
    walks a sliding window through it each update (`currStartIndex += step`,
    wrap at end) to emulate the non-repeating sweep. `LivoxGenerator` is the
    Python equivalent — same table, same wrapping cursor. We do not reinvent
    the pattern; we reuse `LivoxGenerator` and only convert its per-frame
    (theta, phi) into Genesis local ray direction vectors.

    - FOV is NOT configured here. It is an intrinsic property of the table
      (azimuth 0..360deg, elevation -7..+52deg) and matches the real sensor.
    - `advance()` steps the cursor one frame's worth of rays, recomputing
      `ray_dirs` in place so successive frames use distinct angles (the
      non-repeating accumulation the real sensor produces).
    - Angle -> direction uses the exact convention the MuJoCo path uses
      (local = (cos p cos t, cos p sin t, sin p)); Genesis'
      `spherical_to_cartesian(theta, phi)` is identical, so points from both
      backends land in the same lidar_site local frame.
"""

import numpy as np
import torch

import genesis as gs
from genesis.options.sensors.raycaster import RaycastPattern
from genesis.utils.geom import spherical_to_cartesian


class LivoxMid360Pattern(RaycastPattern):
    """Non-repeating Livox MID-360 raycast pattern backed by ``LivoxGenerator``.

    Args:
        samples: rays emitted per frame (per publish). Defaults to the table's
            native ``samples`` (24000), matching the MuJoCo backend and the
            real ~200k pts/s @ 10 Hz point rate.
        downsample: keep every Nth ray (>=1). Same knob as the Gazebo plugin's
            ``downsample``; use it to trade density for speed in big scenes.
    """

    def __init__(self, samples: int | None = None, downsample: int = 1):
        # Reuse the Livox angle table + wrapping cursor (no wheel reinvented).
        from mujoco_lidar.scan_gen import LivoxGenerator

        self._gen = LivoxGenerator("mid360")
        self._downsample = max(1, int(downsample))
        if samples is not None:
            self._gen.samples = int(samples)
        # Rays actually returned after downsampling (matches sample_ray_angles).
        self._n_rays = len(range(0, int(self._gen.samples), self._downsample))
        # Snapshot line ids for the current window (kept aligned with ray_dirs).
        self._line_ids = np.zeros(self._n_rays, dtype=np.uint8)
        super().__init__()

    def _get_return_shape(self) -> tuple[int, ...]:
        return (self._n_rays,)

    def compute_ray_dirs(self):
        """Pull the next window of (theta, phi) and write local ray dirs.

        Called once by ``RaycastPattern.__init__`` and again by ``advance()``.
        """
        from simu_utils.livox_line_classifier import assign_livox_line_ids

        # Snapshot line ids BEFORE sample_ray_angles mutates currStartIndex,
        # exactly as the MuJoCo path does (npy is line-interleaved, id = row%4).
        line_ids_full = assign_livox_line_ids(self._gen, int(self._gen.samples))
        theta, phi = self._gen.sample_ray_angles(downsample=self._downsample)
        # Keep line ids aligned with the (possibly downsampled) returned rays.
        self._line_ids = line_ids_full[:: self._downsample][: len(theta)].astype(np.uint8)

        theta_t = torch.as_tensor(np.asarray(theta), dtype=gs.tc_float, device=gs.device)
        phi_t = torch.as_tensor(np.asarray(phi), dtype=gs.tc_float, device=gs.device)
        self._ray_dirs[:] = spherical_to_cartesian(theta_t, phi_t)

    def advance(self):
        """Step the sweep by one frame and refresh ``ray_dirs`` in place.

        Genesis caches ray_dirs once at build time, so we recompute + copy so
        the cached device buffer the sensor reads sees the new window.
        """
        self.compute_ray_dirs()

    @property
    def line_ids(self) -> np.ndarray:
        """(n_rays,) uint8 line ids in {0,1,2,3} aligned with the last window."""
        return self._line_ids
