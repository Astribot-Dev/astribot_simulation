#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025, Astribot Co., Ltd.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------
# Author: Tony Wang, tonywang@astribot.com
# -----------------------------------------------------------------------------

"""
File: astribot_envs_factory.py
Brief: Factory for simulation env
"""

import importlib
import os
import threading

import gymnasium
import yaml

from astribot_envs.simulation_constants import ENV_STEP_RATE_HZ

# astribot_simu_log is imported lazily inside load_yaml_file because it pulls
# in rclpy/rospy, which requires a ROS context not yet initialized at yaml
# load time. The compat shim log uses plain print instead.

ros_version = os.getenv("ROS_VERSION")

if ros_version == "1":
    import rospy
elif ros_version == "2":
    import rclpy


def _is_shutdown_race(exc):
    """True when exc is the fallout of a Ctrl+C rather than a real failure.

    Interrupting the sim tears down the rclpy context while the physics thread may
    still be publishing, so the thread sees ExternalShutdownException or an
    RCLError about an invalid publisher context. Both mean "we are shutting down".
    """
    if ros_version != "2":
        return False
    shutdown_types = tuple(
        t
        for t in (
            getattr(rclpy.executors, "ExternalShutdownException", None),
            getattr(rclpy, "_rclpy", None) and getattr(rclpy._rclpy, "RCLError", None),
            RuntimeError,
        )
        if isinstance(t, type)
    )
    if not isinstance(exc, shutdown_types):
        return False
    if isinstance(exc, getattr(rclpy.executors, "ExternalShutdownException", ())):
        return True
    return "context is invalid" in str(exc) or "shutdown" in str(exc).lower()


class AstribotEnvsFactory:
    def __init__(self):
        self.astribot_simu_env = None

    def create_simulation_env(self, data):
        # Install the Genesis MJCF loader patch BEFORE the backend module is
        # imported, otherwise the first `gs.Scene.add_entity` call inside
        # AstribotGenesisEnv goes through Genesis's broken from_xml_string()
        # path and fails. MuJoCo is unaffected — it never calls build_model.
        if data.get("simulator_type", "").lower() == "genesis":
            from simu_utils.monkeypatch_genesis_mjcf import install_build_model_patch

            mp = data.get("model_path", "")
            install_build_model_patch(working_dir=os.path.dirname(mp) or None)

        module_name = f"src.astribot_envs.astribot_{data['simulator_type'].lower()}_env"
        class_name = f"Astribot{data['simulator_type']}Env"

        try:
            module = importlib.import_module(module_name)
            _ = getattr(module, class_name)  # Trigger module registration
            gym_env_name = f"astribot_envs/{class_name}-v0"
            self.astribot_simu_env = gymnasium.make(gym_env_name, param=data)

            # Both supported backends (MuJoCo / Genesis) run the sim loop in a
            # background thread.
            self.simu_thread = threading.Thread(target=self.simu_env_loop, args=())
            self.simu_thread.start()
            return self.simu_thread
        except ModuleNotFoundError:
            raise ValueError(
                f"Unknown simulator type: {data['simulator_type']}. Module: {module_name} not found."  # noqa: E501
            )
        except AttributeError:
            raise ValueError(f"Class {class_name} not found in {module_name}.")

    # Scene name (in sim.yaml) → path under the sim root. Scenes live
    # in `<sim_root>/astribot_scenes/<robot>/<scene>/scene.xml` (peer
    # to astribot_descriptions/ and astribot_simulation/). Each scene
    # is per-robot: it knows its robot via a REAL relative path (no
    # placeholder, no inliner). mjcf's native <include file="..."/>
    # mechanism handles the cross-dir resolution.
    #
    # Conventions:
    #   scene: floor  → astribot_scenes/<robot>/floor/scene.xml
    #   scene: table  → astribot_scenes/<robot>/table/scene.xml
    #   anything else → astribot_scenes/<robot>/<scene>/scene.xml (convention)
    #
    # Naming: scene names describe WHAT'S IN THE SCENE, not chassis
    # behavior. Use the `chassis_fixed:` yaml field for chassis behavior.
    # NOTE: "chassis_fixed" is NOT a scene. To lock the chassis, use the
    # sim.yaml `chassis_fixed: true` flag (orthogonal to scene).
    # Back-compat alias — the mapping now lives in config_loader (single
    # source of truth). Kept as a class attr for any external reference.
    from simu_utils.config_loader import SCENE_TO_PATH

    @staticmethod
    def _deep_merge(base, override):
        """Deep-merge override onto base and return a new dict. Nested dicts recurse;
        every other value type is replaced outright."""
        import copy

        result = copy.deepcopy(base)
        for key, value in (override or {}).items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = AstribotEnvsFactory._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def load_yaml_file(yaml_file_path, merge_common=False):
        with open(yaml_file_path, "r") as yaml_file:
            data = yaml.safe_load(yaml_file)
        astribot_simu_root = os.getenv("ASTRIBOT_SIMU_ROOT")

        # When merge_common is set, the shared config/simulation.yaml is loaded first
        # as the base and the robot yaml is deep-merged over it. The shared config
        # (physics frequencies / gravity / ros / sensor_frequencies) is then the single
        # source of truth across backends, while robot-specific settings
        # (robot_list / scene / camera) live in the robot yaml. Defaults to False for
        # backward compatibility with single-file loading in tests and fixtures.
        if merge_common:
            common_path = os.path.join(astribot_simu_root or ".", "config", "simulation.yaml")
            if os.path.exists(common_path):
                with open(common_path, "r") as cf:
                    common = yaml.safe_load(cf) or {}
                data = AstribotEnvsFactory._deep_merge(common, data or {})

        # Resolve mjcf path from scene + chassis_model. Delegates to
        # config_loader.resolve_scene_model_path — the single source of truth
        # shared with the production load_config path (was duplicated here).
        from simu_utils.config_loader import resolve_scene_model_path

        data = resolve_scene_model_path(data, data.get("robot_type", ""))
        return data

    def running(self):
        if ros_version == "1":
            if not hasattr(self, "rate"):
                self.rate = rospy.Rate(ENV_STEP_RATE_HZ)
            return not rospy.is_shutdown()
        elif ros_version == "2":
            if not hasattr(self, "rate"):
                self.rate = self.astribot_simu_env.unwrapped.node.create_rate(ENV_STEP_RATE_HZ)
            if not hasattr(self, "spin_thread"):
                self.executor = rclpy.executors.SingleThreadedExecutor()
                self.executor.add_node(self.astribot_simu_env.unwrapped.node)
                self.spin_thread = threading.Thread(target=self._spin_quietly, daemon=True)
                self.spin_thread.start()
            return rclpy.ok()

    def _spin_quietly(self):
        """Run the executor, swallowing the shutdown race on Ctrl+C.

        executor.spin() raises ExternalShutdownException once the context is
        destroyed. That is the normal end of life for this thread, so it should
        not print a traceback.
        """
        try:
            self.executor.spin()
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            if not _is_shutdown_race(exc):
                raise

    def simu_env_loop(self):
        observation, info = self.astribot_simu_env.reset()

        # Sim loop for MuJoCo and Genesis.
        # When the env paces itself (realtime profile sets _pace_realtime), skip
        # the factory's rate.sleep to avoid two competing clocks: the env locks
        # to sim_dt while rate locks to ENV_STEP_RATE_HZ, and the periods may
        # differ. The env's precise_sleep already guarantees 1:1 realtime.
        env_paces = getattr(self.astribot_simu_env.unwrapped, "_pace_realtime", False)
        try:
            while self.running():
                action = self.astribot_simu_env.action_space.sample()
                observation, reward, terminated, truncated, info = self.astribot_simu_env.step(
                    action
                )
                if not env_paces:
                    self.rate.sleep()
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            # On Ctrl+C rclpy tears the context down while this thread may be
            # mid-publish, which surfaces as ExternalShutdownException or
            # "publisher's context is invalid". Those are shutdown races, not
            # failures, so exit quietly and let close() release the renderer --
            # otherwise the thread dies with a live GL context and the process
            # ends in SIGABRT/segfault instead of a clean exit.
            if _is_shutdown_race(exc):
                pass
            else:
                raise
        finally:
            try:
                self.astribot_simu_env.close()
            except Exception:
                # Never mask the original exit reason with a teardown error.
                pass
