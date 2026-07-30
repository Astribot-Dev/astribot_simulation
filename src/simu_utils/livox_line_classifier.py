#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025, Astribot Co., Ltd.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------
# Author: simu dev
# -----------------------------------------------------------------------------

"""
Livox MID-360 line_id classification.

Background:
    The Livox MID-360 emits 4 scan lines (line 0..3) per sweep, but the
    `mujoco_lidar.scan_gen.LivoxGenerator.sample_ray_angles()` call does
    not expose which line each returned ray belongs to. The underlying
    `mid360.npy` table interleaves the 4 lines in groups of 4 consecutive
    rows (line 0, 1, 2, 3, line 0, 1, 2, 3, ...). Concretely:
        npy[0]   -> line 0
        npy[1]   -> line 1
        npy[2]   -> line 2
        npy[3]   -> line 3
        npy[4]   -> line 0 (next group)
        ...

    Since `LivoxGenerator.currStartIndex` advances by `samples` per call and
    wraps around at `n_rays`, we can recover the line_id for the rays returned
    in a given call by snapshotting the start index *before* the call and
    computing `(arange(samples) + start) % 4` afterwards.
"""

import numpy as np


def assign_livox_line_ids(
    livox_gen,
    n_samples: int,
) -> np.ndarray:
    """Return (n_samples,) uint8 line_id array for a `LivoxGenerator` call.

    Args:
        livox_gen: an instance of `mujoco_lidar.scan_gen.LivoxGenerator`
            (with `.currStartIndex` and `.samples` attributes).
        n_samples: number of rays the upcoming `sample_ray_angles()` call
            will return. Usually == `livox_gen.samples`.

    Returns:
        (n_samples,) uint8 array of line ids in {0, 1, 2, 3}.

    Algorithm:
        1. snapshot `livox_gen.currStartIndex` (this is the npy-row index
           of the first ray that will be returned by the next call).
        2. compute line_id = (arange(n_samples) + snapshot) % 4.

        This works because the npy table is line-interleaved (every 4 rows
        cycle through line 0..3). The `LivoxGenerator` only slices/wraps
        the npy — it does not reorder lines.

    Usage:
        >>> gen = LivoxGenerator('mid360')
        >>> line_ids = assign_livox_line_ids(gen, gen.samples)
        >>> theta, phi = gen.sample_ray_angles()
        >>> # now `theta, phi, line_ids` are all aligned
    """
    if n_samples <= 0:
        return np.zeros(0, dtype=np.uint8)
    snapshot_start = int(livox_gen.currStartIndex)
    return ((np.arange(n_samples, dtype=np.int64) + snapshot_start) % 4).astype(np.uint8)
