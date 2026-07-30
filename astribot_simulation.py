#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024, Astribot Co., Ltd.
# All rights reserved.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------
# Author: Tony Wang, tonywang@astribot.com
# -----------------------------------------------------------------------------

import argparse
import os
import sys

from astribot_envs.astribot_envs_factory import AstribotEnvsFactory

# Public robot whitelist for the release build. Each entry has a matching
# config/astribot_<robot>/sim.yaml plus astribot_scenes/<robot>/. All backends
# share one sim.yaml; per-backend differences live under
# backend_overrides.<backend> and are merged by config_loader. A robot outside
# this whitelist is rejected by argparse choices (never silently defaulted).
#   - astribot_s1 : S1, dual gripper, omni-wheel chassis
#   - astribot_t1 : T1, dual gripper, omni-wheel chassis + torso lift
# To add a robot: register it here and create its config/ + scenes/ directories.
_SUPPORTED_ROBOTS = ("astribot_s1", "astribot_t1")

# Public name -> internal robot_type used to resolve config/astribot_<type>/sim.yaml
# and astribot_scenes/<type>/. Also accepts the bare short form that env.sh
# exports via $ROBOT_TYPE (S1 / T1), so `source env.sh --robot s1` works with no
# argv. Keys are matched case-insensitively after lowercasing.
_ROBOT_TYPE_ALIASES = {
    "astribot_s1": "s1",
    "astribot_t1": "t1",
    "s1": "s1",
    "t1": "t1",
}


def _normalize_robot_type(raw):
    """Map a user- or env-supplied robot name to its internal robot_type.

    Returns None when the name is not recognised, so the caller can report the
    supported set instead of failing deep inside config resolution.
    """
    return _ROBOT_TYPE_ALIASES.get(raw.strip().lower())


def main(robot_type, backend):
    from simu_utils.config_loader import load_config

    try:
        # config_loader merge order: SimConfig defaults -> simulation.yaml (shared)
        # -> sim.yaml (robot) -> backend_overrides.<backend> -> env vars -> CLI.
        # simulator_type is derived from the backend.
        astribot_data, _prov = load_config(robot_type, backend=backend, verbose=False)
    except FileNotFoundError:
        print(
            f"Unsupported robot_type {robot_type!r}. "
            f"Available: {', '.join(_SUPPORTED_ROBOTS)}."
        )
        sys.exit(1)
    astribot_envs_factory = AstribotEnvsFactory()
    # Sensors (cameras / LiDAR / LiDAR-IMU / F-T) are not supported in this release,
    # so the sim always runs the `realtime` profile: it publishes control topics only
    # (joint states + chassis IMU) and masks everything else. There is no CLI switch
    # for this on purpose -- the sensor pipeline is not mature enough to expose.
    astribot_data["_disabled_sensors"] = []
    astribot_data["sim_profile"] = "realtime"
    _ = astribot_envs_factory.create_simulation_env(astribot_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Astribot simulation entry point. Robots: astribot_s1 / astribot_t1.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 astribot_simulation.py astribot_s1                     "
            "# MuJoCo (default backend)\n"
            "  python3 astribot_simulation.py astribot_t1 --backend genesis   "
            "# Genesis backend\n"
            "\n"
            "The scene is selected by the `scene` field in "
            "config/astribot_<robot>/sim.yaml, not by a CLI flag."
        ),
    )
    parser.add_argument(
        "robot_type",
        nargs="?",
        choices=_SUPPORTED_ROBOTS,
        metavar="robot_type",
        help="Robot model, one of: %(choices)s. Falls back to $ROBOT_TYPE when omitted.",
    )
    parser.add_argument(
        "--backend",
        default="mujoco",
        choices=["mujoco", "genesis"],
        help="Physics backend: mujoco (default) or genesis.",
    )
    args = parser.parse_args()

    # Explicit argv wins over $ROBOT_TYPE, so the default robot can be overridden
    # without re-sourcing env.sh. argv is already validated by argparse choices;
    # the env fallback is normalized and validated here (env.sh exports the short
    # uppercase form, e.g. S1, which _normalize_robot_type accepts).
    if args.robot_type is not None:
        robot_type = _normalize_robot_type(args.robot_type)
    elif os.environ.get("ROBOT_TYPE"):
        raw = os.environ["ROBOT_TYPE"]
        robot_type = _normalize_robot_type(raw)
        if robot_type is None:
            print(
                f"ERROR: unsupported $ROBOT_TYPE {raw!r}. "
                f"Expected one of: {', '.join(_SUPPORTED_ROBOTS)}."
            )
            sys.exit(1)
    else:
        print(
            "ERROR: robot_type is required. Pass one of "
            f"{', '.join(_SUPPORTED_ROBOTS)} as an argument or via $ROBOT_TYPE."
        )
        sys.exit(1)

    main(robot_type, args.backend)
