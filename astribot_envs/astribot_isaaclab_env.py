#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025, Astribot Co., Ltd.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------
# Author: Tony Wang, tonywang@astribot.com
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import sys
import time
import copy
import numpy as np
from typing import Optional, Dict, List, TYPE_CHECKING

os.environ["CARB_LOGGING_ENABLED"] = "0"
os.environ["OMNI_LOG_LEVEL"] = "error"
os.environ["ISAAC_VERBOSE"] = "0"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.CRITICAL)
for logger_name in ["omni", "isaac", "carb", "omni.kit", "omni.isaac",
                    "isaacsim", "omni.physx", "omni.replicator", "omni.ext",
                    "omni.usd", "omni.hydra", "omni.graph"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    logging.getLogger(logger_name).propagate = False

from isaaclab.app import AppLauncher

from simu_utils.simu_common_tools import astribot_simu_log
from astribot_envs.astribot_base_env import AstribotBaseEnv

if TYPE_CHECKING:
    from isaaclab.assets import ArticulationCfg


class IsaacLabAppManager:
    _instance = None
    _app_launcher = None
    _simulation_app = None
    _ref_count = 0
    _modules_loaded = False

    @staticmethod
    def _suppress_output():
        class SuppressOutput:
            def __enter__(self):
                self.original_stdout_fd = sys.stdout.fileno()
                self.original_stderr_fd = sys.stderr.fileno()

                self.saved_stdout_fd = os.dup(self.original_stdout_fd)
                self.saved_stderr_fd = os.dup(self.original_stderr_fd)

                self.devnull_fd = os.open(os.devnull, os.O_WRONLY)

                os.dup2(self.devnull_fd, self.original_stdout_fd)
                os.dup2(self.devnull_fd, self.original_stderr_fd)

                self.original_stdout = sys.stdout
                self.original_stderr = sys.stderr
                sys.stdout = open(os.devnull, 'w')
                sys.stderr = open(os.devnull, 'w')

                return self

            def __exit__(self, *args):
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout = self.original_stdout
                sys.stderr = self.original_stderr

                os.dup2(self.saved_stdout_fd, self.original_stdout_fd)
                os.dup2(self.saved_stderr_fd, self.original_stderr_fd)

                os.close(self.saved_stdout_fd)
                os.close(self.saved_stderr_fd)
                os.close(self.devnull_fd)

        return SuppressOutput()

    @classmethod
    def get_or_create_app(cls, headless: bool = False, width: int = 1920, height: int = 1080):
        if cls._instance is None:
            cls._instance = cls()

        if cls._app_launcher is None:
            with cls._suppress_output():
                import argparse
                parser = argparse.ArgumentParser()
                AppLauncher.add_app_launcher_args(parser)

                args_list = []
                if headless:
                    args_list.append("--headless")

                args = parser.parse_args(args_list)
                cls._app_launcher = AppLauncher(args)
                cls._simulation_app = cls._app_launcher.app

            try:
                import carb
                settings = carb.settings.get_settings()

                if not headless:
                    settings.set("/app/window/width", width)
                    settings.set("/app/window/height", height)

                settings.set("/app/log/outputStreamLevel", 40)
                settings.set("/app/extensions/verboseLogging", False)
            except Exception:
                pass

            cls._load_isaaclab_modules()

        cls._ref_count += 1
        return cls._app_launcher, cls._simulation_app

    @classmethod
    def _load_isaaclab_modules(cls):
        if not cls._modules_loaded:
            global torch, sim_utils, Articulation, ArticulationCfg
            global InteractiveScene, InteractiveSceneCfg, SimulationCfg, SimulationContext

            import torch
            import isaaclab.sim as sim_utils
            from isaaclab.assets import Articulation, ArticulationCfg
            from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
            from isaaclab.sim import SimulationCfg, SimulationContext

            cls._modules_loaded = True

    @classmethod
    def release_app(cls):
        if cls._instance is not None:
            cls._ref_count -= 1
            if cls._ref_count <= 0:
                if cls._simulation_app is not None:
                    astribot_simu_log("Closing Isaac Lab simulation app...")
                    cls._simulation_app.close()
                cls._app_launcher = None
                cls._simulation_app = None
                cls._modules_loaded = False
                cls._ref_count = 0


class AstribotIsaacLabEnv(AstribotBaseEnv):

    def __init__(self, param):
        astribot_simu_log("Initializing Isaac Lab environment...")

        super().__init__(param)

        self.dt = 0.01
        self.decimation = 1

        headless = (self.render_mode != "human")
        self.app_launcher, self.simulation_app = IsaacLabAppManager.get_or_create_app(
            headless=headless,
            width=self.width,
            height=self.height
        )

        self.setup_simulation_and_scene()

        self.setup_joint_mapping()

        self.joint_names_all, self.controller_mode, self.joint_position_command_all, \
            self.joint_velocity_command_all, self.joint_torque_command_all = self.update_joint_states()

        astribot_simu_log("Isaac Lab environment initialized successfully")

    def setup_simulation_and_scene(self):
        import torch
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        sim_cfg = SimulationCfg(
            dt=self.dt,
            render_interval=self.decimation,
            device=self.device,
            gravity=(0.0, 0.0, -9.81),
            physics_prim_path="/physicsScene",
        )

        self.sim = SimulationContext(sim_cfg)

        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)

        light_cfg = sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75)
        )
        light_cfg.func("/World/light", light_cfg)

        scene_cfg = InteractiveSceneCfg(
            num_envs=1,
            env_spacing=3.0,
        )

        robot_cfg = self._create_robot_config()
        scene_cfg.robot = robot_cfg

        self.scene = InteractiveScene(scene_cfg)

        self.robot: Articulation = self.scene["robot"]

        self.sim.reset()

    def _create_robot_config(self) -> ArticulationCfg:
        is_usd = self.model_path.endswith(('.usd', '.usda'))

        if is_usd:
            spawn_cfg = sim_utils.UsdFileCfg(
                usd_path=self.model_path,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4,
                ),
            )
        else:
            spawn_cfg = sim_utils.UrdfFileCfg(
                asset_path=self.model_path,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4,
                ),
            )

        from isaaclab.actuators import ImplicitActuatorCfg

        actuators_cfg = {
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=800.0,
                damping=40.0,
                effort_limit=400.0,
                velocity_limit=100.0,
            ),
        }

        robot_cfg = ArticulationCfg(
            prim_path="/World/envs/env_0/Robot",
            spawn=spawn_cfg,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                joint_pos={".*": 0.0},
                joint_vel={".*": 0.0},
            ),
            actuators=actuators_cfg,
        )

        return robot_cfg

    def setup_joint_mapping(self):
        astribot_simu_log("Setting up joint mapping...")

        self.joint_name_to_index = {}
        self.joint_index_to_name = {}

        all_joint_names = self.robot.data.joint_names

        for idx, joint_name in enumerate(all_joint_names):
            self.joint_name_to_index[joint_name] = idx
            self.joint_index_to_name[idx] = joint_name

    def step(self, action: np.ndarray) -> tuple:
        self.update_reset_flag()

        if not self.reset_flag:
            step_begin_time = time.time()

            self.joint_names_all, self.controller_mode, self.joint_position_command_all, \
                self.joint_velocity_command_all, self.joint_torque_command_all = self.update_joint_states()

            self.update_object_states()
            self.update_trajectory_pose()
            self.update_com_pose()
            self.update_sensor_states()

            self._apply_control()

            for _ in range(self.frame_skip):
                self.scene.write_data_to_sim()
                self.sim.step(render=True)
                self.scene.update(dt=self.dt)

            self.render()

            step_end_time = time.time()
            self.real_time_fps = 1 / (step_end_time - step_begin_time) if (step_end_time - step_begin_time) > 0 else 50
        else:
            self.reset()

        observation = self._get_obs()
        reward = 0
        terminated = False
        info = self._get_info()

        return observation, reward, terminated, False, info

    def _apply_control(self):
        import torch

        joint_position_targets = torch.zeros(self.robot.num_joints, device=self.device)
        joint_velocity_targets = torch.zeros(self.robot.num_joints, device=self.device)
        joint_effort_targets = torch.zeros(self.robot.num_joints, device=self.device)

        for joint_id, mode in enumerate(self.controller_mode):
            if joint_id >= len(self.joint_names_all):
                continue

            joint_name = self.joint_names_all[joint_id]
            if joint_name not in self.joint_name_to_index:
                continue

            robot_joint_idx = self.joint_name_to_index[joint_name]

            if mode == 1:
                pos_cmd = self.vel_compensation_ctrl(
                    self.joint_position_command_all,
                    self.joint_velocity_command_all,
                    idx=joint_id
                )[0]
                joint_position_targets[robot_joint_idx] = pos_cmd

            elif mode == 2:
                current_pos = self.get_joint_positions([joint_name])[0]
                vel_cmd = self.vel_compensation_ctrl(
                    [current_pos],
                    self.joint_velocity_command_all,
                    idx=joint_id
                )[0]
                joint_position_targets[robot_joint_idx] = vel_cmd

            elif mode == 3:
                joint_effort_targets[robot_joint_idx] = self.joint_torque_command_all[joint_id]

        self._sync_gripper_joints(joint_position_targets)

        self.robot.set_joint_position_target(joint_position_targets.unsqueeze(0))

    def _sync_gripper_joints(self, joint_position_targets):
        left_l1_idx = self.joint_name_to_index.get('astribot_gripper_left_joint_L1')
        if left_l1_idx is not None:
            l1_cmd = joint_position_targets[left_l1_idx].item()

            l1_cmd = max(-0.5, min(0.0, l1_cmd))
            gripper_qpos = abs(l1_cmd)

            joint_position_targets[left_l1_idx] = -gripper_qpos

            for joint_name in ['astribot_gripper_left_joint_R1',
                              'astribot_gripper_left_joint_R2',
                              'astribot_gripper_left_joint_L11']:
                idx = self.joint_name_to_index.get(joint_name)
                if idx is not None:
                    joint_position_targets[idx] = gripper_qpos

            for joint_name in ['astribot_gripper_left_joint_L2',
                              'astribot_gripper_left_joint_R11']:
                idx = self.joint_name_to_index.get(joint_name)
                if idx is not None:
                    joint_position_targets[idx] = -gripper_qpos

        right_l1_idx = self.joint_name_to_index.get('astribot_gripper_right_joint_L1')
        if right_l1_idx is not None:
            l1_cmd = joint_position_targets[right_l1_idx].item()

            l1_cmd = max(-0.5, min(0.0, l1_cmd))
            gripper_qpos = abs(l1_cmd)

            joint_position_targets[right_l1_idx] = -gripper_qpos

            for joint_name in ['astribot_gripper_right_joint_R1',
                              'astribot_gripper_right_joint_R2',
                              'astribot_gripper_right_joint_L11']:
                idx = self.joint_name_to_index.get(joint_name)
                if idx is not None:
                    joint_position_targets[idx] = gripper_qpos

            for joint_name in ['astribot_gripper_right_joint_L2',
                              'astribot_gripper_right_joint_R11']:
                idx = self.joint_name_to_index.get(joint_name)
                if idx is not None:
                    joint_position_targets[idx] = -gripper_qpos

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.scene.reset()

        self.reset_flag = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _get_obs(self) -> np.ndarray:
        if self.object_names:
            return np.zeros(7)
        return np.zeros(7)

    def _get_info(self) -> dict:
        robot_info = {}

        for robot_name in self.robot_list:
            joint_names = self.robot_joint_map[robot_name]

            if 'gripper' in robot_name:
                joint_names = joint_names[:1]
                robot_info[robot_name] = self.get_joint_positions(joint_names)
            elif 'chassis' in robot_name:
                pose = self.get_chassis_pose()
                robot_info[robot_name] = pose
            else:
                robot_info[robot_name] = self.get_joint_positions(joint_names)

        return robot_info

    def render(self):
        if self.render_mode == "human":
            pass
        elif self.render_mode == "rgb_array":
            return self.update_camera_data()

    def close(self):
        astribot_simu_log("Closing Isaac Lab environment...")
        IsaacLabAppManager.release_app()

    def get_joint_position(self, joint_name: str) -> float:
        if joint_name not in self.joint_name_to_index:
            return 0.0

        idx = self.joint_name_to_index[joint_name]
        joint_pos = self.robot.data.joint_pos[0, idx].cpu().item()

        if 'gripper' in joint_name:
            return abs(joint_pos / 0.93 * 100)

        return joint_pos

    def get_joint_positions(self, names: List[str]) -> List[float]:
        return [self.get_joint_position(name) for name in names]

    def get_joint_velocity(self, joint_name: str) -> float:
        if joint_name not in self.joint_name_to_index:
            return 0.0

        idx = self.joint_name_to_index[joint_name]
        joint_vel = self.robot.data.joint_vel[0, idx].cpu().item()

        if 'gripper' in joint_name:
            return joint_vel / 0.93 * 100

        return joint_vel

    def get_joint_velocities(self, names: List[str]) -> List[float]:
        return [self.get_joint_velocity(name) for name in names]

    def get_joint_acceleration(self, joint_name: str) -> float:
        if joint_name not in self.joint_name_to_index:
            return 0.0

        idx = self.joint_name_to_index[joint_name]
        joint_acc = self.robot.data.joint_acc[0, idx].cpu().item()

        return joint_acc

    def get_joint_accelerations(self, names: List[str]) -> List[float]:
        return [self.get_joint_acceleration(name) for name in names]

    def get_joint_torque(self, joint_name: str) -> float:
        if joint_name not in self.joint_name_to_index:
            return 0.0

        idx = self.joint_name_to_index[joint_name]
        joint_torque = self.robot.data.applied_torque[0, idx].cpu().item()

        return joint_torque

    def get_joint_torques(self, names: List[str]) -> List[float]:
        return [self.get_joint_torque(name) for name in names]

    def get_site_pose(self, site_name: str) -> np.ndarray:
        try:
            link_idx = self.robot.find_bodies(site_name)[0][0]
            pos = self.robot.data.body_pos_w[0, link_idx].cpu().numpy()
            quat = self.robot.data.body_quat_w[0, link_idx].cpu().numpy()
            return np.concatenate([pos, quat])
        except:
            return np.array([0, 0, 0, 1, 0, 0, 0])

    def get_body_pose(self, body_name: str = None) -> np.ndarray:
        if body_name is None:
            return np.zeros(7)

        return self.get_site_pose(body_name)

    def get_chassis_pose(self) -> np.ndarray:
        pos = self.robot.data.root_pos_w[0].cpu().numpy()
        quat = self.robot.data.root_quat_w[0].cpu().numpy()
        return np.concatenate([pos, quat])

    def get_ft_sensor_data(self, robot_name: str) -> Optional[np.ndarray]:
        return None

    def get_reset_status(self) -> bool:
        if self.simulation_app is not None:
            return not self.simulation_app.is_running()
        return False

    def get_camera_image(self, camera_name: str = 'head') -> Dict:
        data = {
            'rgb_img': None,
            'depth_img': None,
            'point_cloud': None
        }
        return data
