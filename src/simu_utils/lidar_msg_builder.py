#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025, Astribot Co., Ltd.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------
# Author: simu dev
# -----------------------------------------------------------------------------

"""
build_livox_custom_msg — Convert a (N, 3) lidar point cloud + per-point line
ids into a `livox_ros_driver2.msg.CustomMsg` for the SLAM pipeline.

The msg layout (see livox_ros_driver2/msg/CustomMsg.msg) is:
    std_msgs/Header header    # stamp = first-point time, frame_id = "lidar_site"
    uint64 timebase           # ns
    uint32 point_num
    uint8  lidar_id           # 1 = Livox MID360
    uint8[3] rsvd             # reserved
    CustomPoint[] points      # {offset_time, x, y, z, reflectivity, tag, line}

Reference: meta_astribot_localization/src/core/livox_ros_driver2_msgs/msg/
           (the source dir is named `_msgs` but the package.xml <name> is
           `livox_ros_driver2` — so Python import is `livox_ros_driver2.msg`).
"""

import time

import numpy as np

try:
    from livox_ros_driver2.msg import CustomMsg as _CustomMsg
    from livox_ros_driver2.msg import CustomPoint
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "lidar_msg_builder requires livox_ros_driver2 "
        "(source meta_astribot_localization/cpp/install/setup.bash). "
        f"Underlying error: {e}"
    )


def build_livox_custom_msg(
    points_xyz: np.ndarray,
    line_ids: np.ndarray,
    timebase_ns: int = None,
    frame_id: str = "lidar_site",
    lidar_id: int = 1,
    reflectivity: int = 100,
    offset_time_mode: str = "linear",
):
    """Build a CustomMsg from a per-point (x, y, z) array and line ids.

    Args:
        points_xyz: (N, 3) float32 array of point coordinates **already
            expressed in the `frame_id` frame** (lidar_site local).
        line_ids: (N,) uint8/int array; each value ∈ {0, 1, 2, 3}
            identifying which of the MID360's 4 scan lines the point belongs to.
        timebase_ns: optional scan-start time in nanoseconds since epoch. When
            None, the current wall-clock is sampled at call time.
        frame_id: header.frame_id (default "lidar_site").
        lidar_id: 1 for Livox MID360 (default).
        reflectivity: 0-255 constant; 100 matches the MID360 hardware default.
        offset_time_mode: per-point offset_time policy.
            - 'linear' (default): interpolate offset_time = i/n * 100 ms, the
              real-sensor behavior FAST-LIO uses for intra-frame de-skew. Keep
              this for hardware-faithful backends (MuJoCo).
            - 'zero': all points share offset_time = 0. Correct for a
              frame-synchronous simulator where every point in a frame is
              sampled at the same instant — feeding a fabricated linear time
              spread would make FAST-LIO de-skew "correct" motion that never
              happened. All-zero makes de-skew a safe no-op (verified against
              meta_astribot_localization: preprocess.cpp:174 → curvature 0,
              IMU_Processing.hpp:452 loop skipped, no divide-by-zero).

    Returns:
        livox_ros_driver2.msg.CustomMsg — assign `.header.stamp` (rclpy Time or rclpy.node.Time)
        on the caller side, or set the wall-clock at the caller. We populate
        `timebase` (ns) as the canonical scan-start; the per-point `offset_time`
        is left at 0 since the Livox format also permits "all points from the
        same scan with 0 offset" for non-repeating patterns.
    """
    if points_xyz.ndim != 2 or points_xyz.shape[1] < 3:
        raise ValueError(f"points_xyz must have shape (N, >=3), got {points_xyz.shape}")
    n = int(points_xyz.shape[0])
    if n == 0:
        # Edge case: empty scan; emit a well-formed zero-point msg.
        msg = _CustomMsg()
        msg.lidar_id = int(lidar_id)
        msg.rsvd = [0, 0, 0]
        msg.point_num = 0
        msg.points = []
        msg.timebase = int(timebase_ns if timebase_ns is not None else time.time_ns())
        return msg

    if line_ids.shape != (n,):
        raise ValueError(f"line_ids must have shape ({n},), got {line_ids.shape}")

    # Ensure dtypes match the .msg expectations: float32 xyz, uint8 line.
    pts = np.ascontiguousarray(points_xyz[:, :3], dtype=np.float32)
    lines = np.asarray(line_ids, dtype=np.uint8)

    # bypass CustomPoint setters by writing __slots__ directly. The
    # ROS2 IDL-generated CustomPoint has @setter methods that re-run type
    # checks (isinstance + range) on every assignment — for 24k points
    # that's 24k × 7 = 168k Python-level type assertions. Writing _x / _y
    # directly skips them entirely.
    #
    # Measured 24k points: 43.7 ms → ~10 ms (4× faster). The setter path
    # is the dominant cost; allocation + GC are minor by comparison.
    #
    # Safety: we just dtype-coerced pts to float32 / lines to uint8 above,
    # so the assertions wouldn't fire anyway. Skipping them is sound.
    #
    # offset_time linear interpolation (P0-1 fix from
    # SIM_BAG_QUALITY_REPORT.md). Livox MID-360 scans at 10 Hz = 100 ms
    # period; real driver writes per-point offset_time ∈ [0, 100ms] by
    # sampling timestamp. FAST-LIO2 uses this for IMU preintegration-based
    # motion undistortion (project each point from its sample time back to
    # frame start). When offset_time=0 for all points, the algorithm assumes
    # simultaneous capture → skips undistortion → map ghosting / ring artifacts.
    # Fix: linearly interpolate offset_time = i / n * 100ms for point i.
    SCAN_PERIOD_NS = 100_000_000  # 100 ms = 0.1 s
    refl_int = int(reflectivity)
    # 'zero' skips the per-point interpolation entirely (frame-synchronous sim).
    zero_offset = offset_time_mode == "zero"
    pts_list = pts.tolist()  # 24k × 3 floats — list of [x,y,z] is fast batch
    lines_list = lines.tolist()  # 24k uint8 → Python int list

    points_out = []
    points_out_append = points_out.append  # local-bind for tight-loop speed
    for i in range(n):
        cp = CustomPoint.__new__(CustomPoint)
        # Linear interpolation: point i at time (i / n) * scan_period.
        # 'zero' mode: all points captured at the same instant (see docstring).
        cp._offset_time = 0 if zero_offset else int(i * SCAN_PERIOD_NS // n)
        xyz = pts_list[i]
        cp._x = xyz[0]
        cp._y = xyz[1]
        cp._z = xyz[2]
        cp._reflectivity = refl_int
        cp._tag = 0
        cp._line = lines_list[i]
        points_out_append(cp)

    msg = _CustomMsg()
    msg.header.frame_id = frame_id
    msg.lidar_id = int(lidar_id)
    msg.rsvd = [0, 0, 0]
    msg.point_num = n
    msg.points = points_out
    msg.timebase = int(timebase_ns if timebase_ns is not None else time.time_ns())
    return msg
