#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from dataclasses import fields, is_dataclass
from typing import Any, Dict, Optional, Tuple

import yaml

SOURCE_DEFAULT = "default"
SOURCE_COMMON = "simulation.yaml"
SOURCE_ROBOT = "sim.yaml"
SOURCE_BACKEND = "backend_overrides"
SOURCE_ENV = "env"
SOURCE_CLI = "cli"


class ConfigProvenance:
    """Tracks where each resolved config value came from."""

    def __init__(self):
        self._records: Dict[str, Tuple[Any, str]] = {}
        self._overrides: Dict[str, list] = {}

    def record(self, path: str, value: Any, source: str):
        if path in self._records:
            old_value, old_source = self._records[path]
            if old_value != value:
                self._overrides.setdefault(path, []).append((old_value, old_source))
        self._records[path] = (value, source)

    def get_source(self, path: str) -> Optional[str]:
        if path in self._records:
            return self._records[path][1]
        return None

    def print_summary(self, log_fn=print):
        """Args:"""
        log_fn("=" * 60)
        log_fn("[config] Configuration provenance summary")
        log_fn("=" * 60)

        if self._overrides:
            log_fn("The following settings were overridden:")
            for path in sorted(self._overrides.keys()):
                final_value, final_source = self._records[path]
                history = self._overrides[path]

                chain_parts = []
                for old_val, old_src in history:
                    chain_parts.append(f"{old_val} ({old_src})")
                chain_parts.append(f"{final_value} ({final_source})")

                log_fn(f"  {path}:")
                log_fn(f"    {' → '.join(chain_parts)}")
        else:
            log_fn("No overrides (all defaults or a single source)")

        log_fn("=" * 60)

    def print_full(self, log_fn=print):
        log_fn("=" * 60)
        log_fn("[config] Full configuration (with provenance)")
        log_fn("=" * 60)
        for path in sorted(self._records.keys()):
            value, source = self._records[path]
            log_fn(f"  {path} = {value}")
            log_fn(f"    ← {source}")
        log_fn("=" * 60)


def _flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep))
        else:
            items[new_key] = v
    return items


def _deep_merge(
    base: Dict, override: Dict, provenance: ConfigProvenance, source: str, path_prefix: str = ""
) -> Dict:
    """Recursively merge `override` into `base`, recording provenance per key."""
    for key, value in override.items():
        current_path = f"{path_prefix}.{key}" if path_prefix else key

        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value, provenance, source, current_path)
        else:
            base[key] = value
            provenance.record(current_path, value, source)

    return base


def _extract_env_overrides() -> Dict:
    """Returns:"""
    overrides = {}

    if os.getenv("ASTRIBOT_PHYSICS_DT"):
        overrides.setdefault("physics", {})["dt"] = float(os.getenv("ASTRIBOT_PHYSICS_DT"))
    if os.getenv("ASTRIBOT_FRAME_SKIP"):
        overrides.setdefault("physics", {})["frame_skip"] = int(os.getenv("ASTRIBOT_FRAME_SKIP"))

    if os.getenv("ASTRIBOT_IMU_FREQ"):
        overrides.setdefault("sensor_frequencies", {})["imu"] = float(
            os.getenv("ASTRIBOT_IMU_FREQ")
        )
    if os.getenv("ASTRIBOT_LIDAR_FREQ"):
        overrides.setdefault("sensor_frequencies", {})["lidar"] = float(
            os.getenv("ASTRIBOT_LIDAR_FREQ")
        )
    if os.getenv("ASTRIBOT_CHASSIS_IMU_FREQ"):
        overrides.setdefault("sensor_frequencies", {})["chassis_imu"] = float(
            os.getenv("ASTRIBOT_CHASSIS_IMU_FREQ")
        )

    return overrides


def load_config(
    robot_type: str,
    backend: str = "mujoco",
    cli_overrides: Optional[Dict] = None,
    config_root: Optional[str] = None,
    verbose: bool = True,
):
    """Load and merge the configuration for one robot and backend.

    Merge order (low to high): SimConfig defaults, config/simulation.yaml (shared
    base), the robot's sim.yaml, backend_overrides.<backend>, env vars, then CLI.

    Returns:
        (config_dict, provenance)

    Raises:
        FileNotFoundError: the robot's sim.yaml does not exist.
    """
    provenance = ConfigProvenance()

    if config_root is None:
        sim_root = os.getenv("ASTRIBOT_SIMU_ROOT", ".")
        config_root = os.path.join(sim_root, "config")

    from simu_utils.sim_config import SimConfig

    config_dict = _dataclass_to_dict(SimConfig())
    for path, value in _flatten_dict(config_dict).items():
        provenance.record(path, value, SOURCE_DEFAULT)

    common_path = os.path.join(config_root, "simulation.yaml")
    if os.path.exists(common_path):
        with open(common_path, "r") as f:
            common_data = yaml.safe_load(f) or {}
        _deep_merge(config_dict, common_data, provenance, SOURCE_COMMON)

    robot_path = os.path.join(config_root, f"astribot_{robot_type}", "sim.yaml")
    if not os.path.exists(robot_path):
        raise FileNotFoundError(
            f"Robot config not found: {robot_path}\n"
            f"  robot_type='{robot_type}' may be invalid"
        )
    with open(robot_path, "r") as f:
        robot_data = yaml.safe_load(f) or {}

    backend_overrides = robot_data.pop("backend_overrides", {})

    _deep_merge(config_dict, robot_data, provenance, SOURCE_ROBOT)

    if backend in backend_overrides:
        _deep_merge(
            config_dict, backend_overrides[backend], provenance, f"{SOURCE_BACKEND}.{backend}"
        )

    config_dict["simulator_type"] = backend.capitalize()
    provenance.record("simulator_type", config_dict["simulator_type"], SOURCE_CLI)

    env_overrides = _extract_env_overrides()
    if env_overrides:
        _deep_merge(config_dict, env_overrides, provenance, SOURCE_ENV)

    if cli_overrides:
        _deep_merge(config_dict, cli_overrides, provenance, SOURCE_CLI)

    config_dict = _resolve_model_path(config_dict, robot_type, config_root)

    if verbose:
        try:
            from simu_utils.simu_common_tools import astribot_simu_log

            log_fn = lambda msg: astribot_simu_log(msg)  # noqa: E731
        except Exception:
            log_fn = print
        provenance.print_summary(log_fn)

    return config_dict, provenance


def _dataclass_to_dict(obj) -> Dict:
    if is_dataclass(obj):
        result = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            if value is None:
                continue
            result[f.name] = _dataclass_to_dict(value)
        return result
    elif isinstance(obj, list):
        return [_dataclass_to_dict(item) for item in obj]
    else:
        return obj


SCENE_TO_PATH = {
    "floor": "floor/scene.xml",
    "table": "table/scene.xml",
}


def resolve_scene_model_path(data: Dict, robot_type: str, sim_root: Optional[str] = None) -> Dict:
    """Resolve data['model_path'] from `scene` plus `chassis_model`.

    Priority:
      1. scene -> astribot_scenes/<robot>/<scene>/scene[_kinematic].xml
      2. an explicit model_path, joined with sim_root

    chassis_model picks the chassis variant: kinematic loads scene_kinematic.xml
    (virtual-joint chassis, no roller contacts), dynamic loads scene.xml (real roller
    dynamics). Falls back to scene.xml when the kinematic variant is absent.

    Raises:
        FileNotFoundError: the resolved scene path is missing and no model_path fallback.
        ValueError: neither scene nor model_path was declared.
    """
    if sim_root is None:
        sim_root = os.getenv("ASTRIBOT_SIMU_ROOT", ".")

    scene = data.get("scene")
    model_path = data.get("model_path")

    if scene:
        rel_path = SCENE_TO_PATH.get(scene, f"{scene}/scene.xml")
        chassis_model = data.get("chassis_model", "kinematic")
        if chassis_model == "kinematic":
            kin_rel = rel_path.replace("/scene.xml", "/scene_kinematic.xml")
            if os.path.exists(os.path.join(sim_root, "astribot_scenes", robot_type, kin_rel)):
                rel_path = kin_rel
        scene_full_path = os.path.join(sim_root, "astribot_scenes", robot_type, rel_path)
        if os.path.exists(scene_full_path):
            data["model_path"] = scene_full_path
        elif model_path:
            data["model_path"] = sim_root + model_path
        else:
            raise FileNotFoundError(
                f"Scene file not found: {scene_full_path}\n"
                "  and no model_path fallback"
            )
    elif model_path:
        data["model_path"] = sim_root + model_path
    else:
        raise ValueError("Config must declare either 'scene' or 'model_path'")

    return data


def _resolve_model_path(config_dict: Dict, robot_type: str, config_root: str) -> Dict:
    return resolve_scene_model_path(config_dict, robot_type)


def build_sim_config(config_dict: Dict):
    """Build a typed SimConfig from a merged config dict.

    Unknown top-level keys are dropped.

    Returns:
        SimConfig

    Raises:
        ValueError: the resulting config fails validation.
    """
    from simu_utils.sim_config import (
        DisplayConfig,
        LiDARConfig,
        PhysicsConfig,
        ROSConfig,
        SensorFrequencies,
        SimConfig,
    )

    def _build_sub(cls, data):
        if not isinstance(data, dict):
            return cls()
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    top_valid = {f.name for f in fields(SimConfig)}
    top_data = {k: v for k, v in config_dict.items() if k in top_valid}

    top_data["physics"] = _build_sub(PhysicsConfig, config_dict.get("physics", {}))
    top_data["sensor_frequencies"] = _build_sub(
        SensorFrequencies, config_dict.get("sensor_frequencies", {})
    )
    top_data["display"] = _build_sub(DisplayConfig, config_dict.get("display", {}))
    top_data["ros"] = _build_sub(ROSConfig, config_dict.get("ros", {}))
    top_data["lidar"] = _build_sub(LiDARConfig, config_dict.get("lidar", {}))

    config = SimConfig(**top_data)
    config.validate()

    return config
