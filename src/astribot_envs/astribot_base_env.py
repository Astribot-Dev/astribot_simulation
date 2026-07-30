#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025, Astribot Co., Ltd.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------
# Author: Tony Wang, tonywang@astribot.com
# -----------------------------------------------------------------------------

"""
File: astribot_base_env.py
Brief: Base env for simulation
"""

import copy
import os
import queue
import time
from abc import ABC, abstractmethod
from typing import List, Optional

import gymnasium as gym
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import Quaternion, WrenchStamped
from gymnasium import spaces
from sensor_msgs.msg import Imu, PointCloud2, PointField
from std_msgs.msg import Float64MultiArray, Header

from simu_utils.robot_ros_interface import MultiRobotRosInterface
from simu_utils.simu_common_tools import SimuCommonTools, astribot_simu_log

ros_version = os.getenv("ROS_VERSION")
if ros_version == "1":
    import rospy
    from tf.transformations import quaternion_from_matrix
elif ros_version == "2":
    import rclpy
    from rclpy.node import Node
    from tf_transformations import quaternion_from_matrix

# Sensor-name prefix for chassis-mounted sensors. The realtime profile keeps only
# these (the chassis IMU is control-related) and drops the LiDAR-integrated IMU
# and F/T sensors. Matches CHASSIS_IMU_TRIGGER ("astribot_chassis_base_imu_gyro")
# in simu_utils.robot_ros_interface.
_CHASSIS_SENSOR_PREFIX = "astribot_chassis"


class AstribotBaseEnv(gym.Env, ABC):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, param):
        astribot_simu_log("Setup param from yaml", level="DEBUG")
        self.param = param
        self.robot_name = param.get("robot_name", "")
        self.model_path = param.get("model_path", "")
        self.robot_list = param.get("robot_list", [])
        self.joint_names_list = param.get("joint_names_list", [])

        # Controller configuration (replaces gravity_compensation)
        self.controller_config = param.get("controller_config", {})
        if not isinstance(self.controller_config, dict):
            raise ValueError("controller_config must be a dict")

        _display = param.get("display", {}) or {}
        self.render_mode = param.get("mode", "") or _display.get("mode", "")
        self.width = param.get("width", 0) or _display.get("width", 0)
        self.height = param.get("height", 0) or _display.get("height", 0)
        self.object_names = param.get("object_names", [])
        self.camera_names = param.get("camera_names", [])
        self.sensor_names = param.get("sensor_names", [])
        self.vel_compensation_list = param.get("vel_compensation_list", [])
        self.update_trajectory_map = param.get("update_trajectory_map", {})
        # Force/torque sensor switch. The sim does not use F/T, so it is off by
        # default to save the per-step read and publish cost. Only enabled when the
        # config sets enable_ft_sensor: true (the realtime profile forces it off).
        self.enable_ft_sensor = bool(param.get("enable_ft_sensor", False))

        # sim_profile runtime feature mask. Applied BEFORE setup_ros_node (which
        # creates publishers from camera_names / sensor_names), so the realtime profile
        # can bypass sensors and cameras to approach real time (RTF>=0.95). It only
        # rewrites the consumed instance attributes and leaves the original param
        # values alone, so the change is reversible.
        self._apply_sim_profile()

        self.setup_ros_node()

        self.reset_flag = False
        self.frame_skip = 3
        self.vel_compensation_map = dict()
        self.robot_joint_map = dict()
        self.cv_bridge = CvBridge()
        self.robot_dict = dict()

        for i in range(len(self.robot_list)):
            self.robot_joint_map[self.robot_list[i]] = copy.deepcopy(self.joint_names_list[i])
            self.vel_compensation_map[self.robot_list[i]] = self.vel_compensation_list[i]

        # Initial home-position command targets (config initial_joint_positions is the
        # single source of truth). On reset (startup and Backspace) the PD position
        # command is set to these values rather than 0; otherwise the controller drags
        # the joints back to 0. This is especially visible on Genesis, where
        # set_dofs_position sets the state but a PD target of 0 immediately pulls it
        # back. Built per component, one command vector each (joints not listed -> 0).
        # See config/astribot_*/sim.yaml.
        self.initial_position_command_map = self._build_initial_position_command_map()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64)
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(7,), dtype=np.float64)

        self.setup_joint_interface()
        # Initial PD command target = the home position (otherwise it defaults to 0
        # and the controller drags the joints from initial_qpos back to 0).
        for robot_name in self.robot_list:
            if robot_name in self.robot_dict:
                init_cmd = self.initial_position_command_map.get(robot_name, [])
                if init_cmd:
                    self.robot_dict[robot_name].joint_position_command = list(init_cmd)

        self._use_sensor_scheduler = int(os.environ.get("ASTRIBOT_USE_SENSOR_SCHEDULER", "1")) == 1
        self._sensor_scheduler = None  # subclass calls setup_sensor_scheduler(dt)

        # Measured wall-clock step rate, overwritten every step by the backends.
        # Diagnostic only -- never use it inside a control law (see
        # vel_compensation_ctrl).
        self.real_time_fps = 50

        # Nominal control period in sim seconds (frame_skip / physics_hz). The
        # backends overwrite this from resolve_timing before calling super().
        # This fallback matches DEFAULT_PHYSICS_HZ/DEFAULT_CONTROL_HZ (200/50).
        if not hasattr(self, "control_dt"):
            self.control_dt = 0.02

        # Periodic real-time-factor report. RTF = sim time advanced / wall time
        # elapsed; 1.00 means the sim keeps pace with the wall clock. Reported
        # every _rtf_report_period wall seconds (ASTRIBOT_RTF_REPORT_SEC, 0 to
        # disable) so it is visible during a run without flooding the log.
        self._rtf_report_period = float(os.environ.get("ASTRIBOT_RTF_REPORT_SEC", "10") or 0)
        self._rtf_wall_anchor = None
        self._rtf_sim_anchor = None

        self.setup_async_camera_rendering()

    def report_realtime_factor(self, sim_time):
        """Log RTF once per report period. Call once per env step with sim time.

        Uses wall/sim deltas between reports rather than since-start totals, so
        the number reflects current performance instead of a lifetime average.
        """
        if self._rtf_report_period <= 0:
            return
        now = time.time()
        if self._rtf_wall_anchor is None:
            self._rtf_wall_anchor = now
            self._rtf_sim_anchor = sim_time
            return
        wall_elapsed = now - self._rtf_wall_anchor
        if wall_elapsed < self._rtf_report_period:
            return
        sim_elapsed = sim_time - self._rtf_sim_anchor
        rtf = sim_elapsed / wall_elapsed if wall_elapsed > 0 else 0.0
        astribot_simu_log(
            f"RTF={rtf:.2f} (sim {sim_elapsed:.1f}s / wall {wall_elapsed:.1f}s)"
        )
        self._rtf_wall_anchor = now
        self._rtf_sim_anchor = sim_time

    def _apply_sim_profile(self):
        """Runtime feature mask: bypass sensors/cameras per sim_profile.

        Profiles:
          - realtime (default): publish only control-related topics — joint
            states plus the chassis IMU — and bypass all cameras, all LiDAR and
            the LiDAR-integrated IMU, and F/T sensors. Targets RTF>=0.95 for VR
            teleop / real-robot sync. The original param values are untouched,
            so switching to custom is reversible.
          - custom: apply no mask at all; the individual config fields
            (camera_names / sensor_names / lidar / mode) decide what is enabled.
            Enable sensors selectively here to balance cost against real-time
            factor, since camera rendering and LiDAR ray casting are expensive.

        Priority: ASTRIBOT_SIM_PROFILE env var > config param['sim_profile']
        > 'realtime'.

        The mask only rewrites the consumed instance attributes
        (self.camera_names / sensor_names / _lidar_force_disabled); the original
        param values are left intact. LiDAR is initialized by subclasses from
        param['lidar']['enabled'], so _lidar_force_disabled is set here for them
        to honour.
        """
        profile = (
            (os.environ.get("ASTRIBOT_SIM_PROFILE") or self.param.get("sim_profile") or "realtime")
            .strip()
            .lower()
        )
        if profile not in ("realtime", "custom"):
            astribot_simu_log(
                f"Unknown sim_profile={profile!r}, falling back to 'realtime'", level="WARN"
            )
            profile = "realtime"
        self.sim_profile = profile
        # Force-disable LiDAR (read by the subclass LiDAR init). Default off, so
        # param is respected.
        self._lidar_force_disabled = False
        # human-window render throttle: render once every N env cycles (physics
        # still runs at full rate). Default 1 (every step).
        self._human_render_every = 1
        # Pace lock: when the sim runs faster than realtime (RTF>1), precise_sleep
        # down to 1:1 so sim time does not run ahead of wall clock and desync
        # VR / real-robot sync. Off for full (run as fast as possible), on for realtime.
        self._pace_realtime = False
        # passive viewer intent (MuJoCo only): realtime wants
        # mujoco.viewer.launch_passive so rendering runs on its own thread and the
        # physics loop is not blocked by GLFW swap vsync. base_env only expresses
        # the intent; mujoco_env decides based on backend / render_mode / mujoco
        # version (see astribot_mujoco_env._resolve_passive_viewer).
        # ASTRIBOT_MUJOCO_PASSIVE=0 disables it.
        self._passive_viewer_requested = False

        if profile == "realtime":
            # Keep control-related publishing only: joint states + chassis IMU.
            # Bypass cameras, LiDAR, the LiDAR-integrated IMU and F/T sensors.
            # The human window's GLFW render measures ~8-33ms/frame and is the
            # dominant bottleneck, so throttle it to reach RTF>=0.95.
            # Priority: ASTRIBOT_HUMAN_RENDER_HZ > config human_render_hz > 17Hz.
            render_hz = os.environ.get("ASTRIBOT_HUMAN_RENDER_HZ")
            if render_hz is None and self.param.get("human_render_hz") is not None:
                render_hz = self.param.get("human_render_hz")
            if render_hz is not None:
                hz = max(1.0, float(render_hz))
                self._human_render_every = max(1, round(50.0 / hz))
            else:
                self._human_render_every = 3
            self.camera_names = []
            # Filter rather than clear: the sensor scheduler and the ROS publisher
            # setup both key off sensor names (see setup_sensor_scheduler and
            # robot_ros_interface.setup_sensor_interface), so keeping only the
            # chassis sensors disables the LiDAR-integrated IMU and F/T while
            # preserving the chassis IMU the control stack needs.
            self.sensor_names = [
                s for s in self.sensor_names if s.startswith(_CHASSIS_SENSOR_PREFIX)
            ]
            self._lidar_force_disabled = True
            self.enable_ft_sensor = False
            astribot_simu_log(
                "sim_profile=realtime: publishing control topics only "
                f"(joint states + chassis IMU: {self.sensor_names}); cameras, LiDAR, "
                "LiDAR IMU and F/T bypassed; human window throttled to every "
                f"{self._human_render_every} step(s) "
                f"(~{50.0/self._human_render_every:.0f}Hz), targeting RTF>=0.95"
            )
            # Pace lock to realtime: sleep to 1:1 when RTF>1 so sim time does not
            # run ahead of the wall clock. ASTRIBOT_PACE_REALTIME=0 disables it.
            self._pace_realtime = os.environ.get("ASTRIBOT_PACE_REALTIME", "1") != "0"
            self._passive_viewer_requested = os.environ.get("ASTRIBOT_MUJOCO_PASSIVE", "1") != "0"
            # render_mode is preserved (the human window stays open).
        else:
            # custom: apply no mask; the config fields alone decide what is enabled.
            pass

    @abstractmethod
    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple:
        observation = self._get_obs()
        reward = 0
        terminated = False
        info = self._get_info()

        return observation, reward, terminated, False, info

    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        return np.zeros(7)

    @abstractmethod
    def _get_info(self) -> dict:
        return {}

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def get_joint_positions(self, names):
        pass

    @abstractmethod
    def get_joint_velocities(self, names):
        pass

    @abstractmethod
    def get_joint_accelerations(self, names):
        pass

    @abstractmethod
    def get_joint_torques(self, names):
        pass

    @abstractmethod
    def get_site_pose(self, site_name: str):
        pass

    @abstractmethod
    def get_body_pose(self):
        pass

    @abstractmethod
    def get_chassis_pose(self):
        pass

    @abstractmethod
    def compute_dynamics_compensation(
        self,
        joint_indices: Optional[List[int]] = None,
        include_gravity: bool = True,
        include_coriolis: bool = True,
    ) -> np.ndarray:
        """Compute dynamics feedforward compensation torque.

        Full dynamics equation: τ = M(q)*q̈ + C(q,q̇)*q̇ + g(q)

        Args:
            joint_indices: Joint indices (None=all)
            include_gravity: Include gravity term g(q)
            include_coriolis: Include Coriolis and centrifugal forces C(q,q̇)*q̇

        Returns:
            compensation_torque: (n_joints,) or subset

        Note:
            Inertia term M(q)*q̈ requires desired acceleration, not implemented yet.
        """
        pass

    def setup_ros_node(self):
        astribot_simu_log("Init ros node", self.robot_name, level="DEBUG")
        if ros_version == "1":
            rospy.init_node(self.robot_name)
            self.node = None
        elif ros_version == "2":
            # Only init if not already initialized (supports several envs per process)
            if not rclpy.ok():
                rclpy.init()
            self.node = Node(self.robot_name)

    def _build_initial_position_command_map(self):
        """Build the initial PD position command vector for each component.

        Source of truth is the config's `initial_joint_positions` (joint name -> angle;
        joints not listed default to 0). Returns {robot_name: [per-joint initial
        command]}, sized in ENV space (the real joint count), not SDK space:
          - regular components (arm / head / 2-joint torso): values are taken per joint
            in joint_names_list order. Under torso_lift the torso still has 2 real
            joints (the env command buffer size); no 1-DOF collapse happens here.
          - gripper: a single main joint, taking that joint's initial value.
        A missing config field yields all zeros (equivalent to the old behaviour).
        """
        init = self.param.get("initial_joint_positions") if hasattr(self.param, "get") else None
        init = init or {}
        cmd_map = {}
        for i, robot_name in enumerate(self.robot_list):
            joints = self.joint_names_list[i]
            if "gripper" in robot_name:
                # On the env side a gripper has only its main joint (joints[0]); the
                # rest follow via MJCF coupling.
                cmd_map[robot_name] = [float(init.get(joints[0], 0.0))]
            else:
                # The env-side command buffer is sized by the real joint count
                # (including torso_lift's 2 real joints). The 1-DOF collapse for
                # torso_lift happens only in the ROS callback, on *incoming SDK
                # commands*; the initial/reset buffers use real joint values directly so
                # they stay aligned with joint_names_all.
                cmd_map[robot_name] = [float(init.get(j, 0.0)) for j in joints]
        return cmd_map

    def setup_joint_interface(self):
        astribot_simu_log("Setup joint interface")
        self.multi_robot_ros_interface = MultiRobotRosInterface(self.node, param=self.param)
        self.robot_dict = self.multi_robot_ros_interface.setup_robot_joint_interface(
            self.robot_joint_map
        )
        self.multi_robot_ros_interface.setup_camera_interface(self.camera_names)
        self.multi_robot_ros_interface.setup_sensor_interface(
            self.sensor_names,
            disabled_sensors=self.param.get("_disabled_sensors", []),
        )
        self.multi_robot_ros_interface.setup_trajectory_and_com_psoe(self.update_trajectory_map)

    def update_reset_flag(self):
        press_status = self.get_reset_status()

        self.reset_flag = self.multi_robot_ros_interface.get_reset_flag() or press_status
        if self.reset_flag:
            self.multi_robot_ros_interface.reset_flag = False
            press_status = False
            for robot_name in self.robot_list:
                if self.robot_dict[robot_name].simu_running:
                    self.reset_time = time.time()
                    self.robot_dict[robot_name].joint_position_command.clear()
                    # Reset the position command to the home position (not 0) so the PD
                    # controller settles at initial_joint_positions.
                    self.robot_dict[robot_name].joint_position_command = list(
                        self.initial_position_command_map.get(
                            robot_name, [0] * self.robot_dict[robot_name].dof
                        )
                    )
                    self.robot_dict[robot_name].joint_velocity_command.clear()
                    self.robot_dict[robot_name].joint_velocity_command = [0] * self.robot_dict[
                        robot_name
                    ].dof
                    self.robot_dict[robot_name].joint_torque_command.clear()
                    self.robot_dict[robot_name].joint_torque_command = [0] * self.robot_dict[
                        robot_name
                    ].dof
                else:
                    self.reset_flag = False
                    astribot_simu_log("Resetting is not supported while following the real robot")

    def update_object_states(self):
        for object_name in self.object_names:
            pose = self.get_body_pose(object_name)
            if pose is None:
                continue

            pose = self.pose_add(self.get_chassis_pose(), pose, inv_1_flag=True)

            self.multi_robot_ros_interface.publish_object_pose(pose, object_name)

        pose_tuple_list = self.multi_robot_ros_interface.get_object_pose_list()

        for pose_tuple in pose_tuple_list:
            if len(pose_tuple) > 0 and pose_tuple[2] in self.object_names:
                _ = self.pose_add(self.get_chassis_pose(), pose_tuple[0])
                if pose_tuple[3]:
                    self.set_body_pose(pose_tuple[2], pose=pose_tuple[0], twist=pose_tuple[1])
                    pose_list = list(pose_tuple)
                    pose_list[3] = False
                    pose_tuple = tuple(pose_list)
        self.multi_robot_ros_interface.pose_tuple_list.clear()

        return pose_tuple_list

    def setup_async_camera_rendering(self):
        """Camera rendering setup (refactored to be synchronous).

        History: cameras used to be rendered asynchronously on a background thread to
        raise the real-time factor.
        Now: rendering is synchronous (called directly from render()), matching how the
        mainstream simulators do it.
        Rationale:
          - Gazebo, Isaac Sim and MuJoCo all expose synchronous interfaces
          - accurate timestamps, simpler tests, better ROS compatibility
          - fast enough (at a 50Hz env loop, a 5-10ms camera render is not the bottleneck)

        This method is kept for backward compatibility but no longer starts a thread.
        """
        # No background thread any more; camera rendering happens synchronously in render()
        self.camera_rendering_active = False
        astribot_simu_log(
            "Camera rendering: synchronous mode (aligned with Gazebo/Isaac/MuJoCo)",
            level="DEBUG",
        )

        # Kept for compatibility; unused
        self.camera_render_queue = None
        self.camera_render_thread = None

    def _camera_render_worker(self):
        while self.camera_rendering_active:
            try:
                camera_name = self.camera_render_queue.get(timeout=0.1)
                camera_data = self.render_single_camera(camera_name)
                if camera_data is not None:
                    self._publish_camera_data(camera_name, camera_data)
                self.camera_render_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                astribot_simu_log(f"Camera render error [{camera_name}]: {e}", level="ERROR")

    def trigger_async_camera_update(self):
        if not self.camera_rendering_active:
            return
        for camera_name in self.camera_names:
            try:
                self.camera_render_queue.put_nowait(camera_name)
            except queue.Full:
                pass

    @abstractmethod
    def render_single_camera(self, camera_name):
        pass

    def _publish_camera_data(self, camera_name, camera_data):
        rgb = camera_data.get("rgb_img")
        if isinstance(rgb, np.ndarray) and rgb.size > 0:
            rgb_msg = self.cv_bridge.cv2_to_imgmsg(rgb, "rgb8")
            rgb_msg.header.stamp = self.multi_robot_ros_interface.get_timestamp()
            rgb_msg.header.frame_id = "simulation"
            self.multi_robot_ros_interface.camera_raw_ros_pub[camera_name].publish(rgb_msg)

        depth = camera_data.get("depth_img")
        if isinstance(depth, np.ndarray) and depth.size > 0:
            depth_msg = self.cv_bridge.cv2_to_imgmsg(depth, "32FC1")
            depth_msg.header.stamp = self.multi_robot_ros_interface.get_timestamp()
            depth_msg.header.frame_id = "simulation"
            self.multi_robot_ros_interface.camera_depth_ros_pub[camera_name].publish(depth_msg)

        point_cloud = camera_data.get("point_cloud")
        if point_cloud is not None:
            if ros_version == "1":
                point_cloud_msg = PointCloud2()
                point_cloud_msg.header.stamp = self.multi_robot_ros_interface.get_timestamp()
                point_cloud_msg.header.frame_id = "simulation"
                point_cloud_msg.height = 1
                point_cloud_msg.width = len(point_cloud.points)
                point_cloud_msg.fields = [
                    PointField("x", 0, PointField.FLOAT32, 1),
                    PointField("y", 4, PointField.FLOAT32, 1),
                    PointField("z", 8, PointField.FLOAT32, 1),
                ]
                point_cloud_msg.is_bigendian = False
                point_cloud_msg.point_step = 12
                point_cloud_msg.row_step = point_cloud_msg.point_step * point_cloud_msg.width
                point_cloud_msg.is_dense = False
                point_cloud_msg.data = np.asarray(point_cloud.points).astype(np.float32).tostring()
            elif ros_version == "2":
                point_cloud_msg = PointCloud2()
                point_cloud_msg.header = Header()
                point_cloud_msg.header.stamp = self.multi_robot_ros_interface.get_timestamp()
                point_cloud_msg.header.frame_id = "simulation"
                point_cloud_msg.height = 1
                point_cloud_msg.width = len(point_cloud.points)
                point_cloud_msg.fields = [
                    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                ]
                point_cloud_msg.is_bigendian = False
                point_cloud_msg.point_step = 12
                point_cloud_msg.row_step = point_cloud_msg.point_step * point_cloud_msg.width
                point_cloud_msg.is_dense = False
                point_cloud_msg.data = np.asarray(point_cloud.points, dtype=np.float32).tobytes()
            self.multi_robot_ros_interface.camera_point_cloud_ros_pub[camera_name].publish(
                point_cloud_msg
            )

    def shutdown_async_camera_rendering(self):
        """Stop camera rendering (a no-op in synchronous mode)."""
        self.camera_rendering_active = False
        # Synchronous mode: there is no background thread to stop
        if hasattr(self, "camera_render_thread") and self.camera_render_thread is not None:
            self.camera_render_thread.join(timeout=1.0)

    def update_joint_states(self):
        joint_position_command_all = list()
        joint_velocity_command_all = list()
        joint_torque_command_all = list()

        controller_mode_list = []
        joint_names_all = []

        # Per-stage timing (ASTRIBOT_PROFILE_STEP=1): splits read (state access) from
        # pub (ROS publish) to locate the bottleneck inside update_joint_states.
        # Zero overhead when disabled.
        _prof = os.environ.get("ASTRIBOT_PROFILE_STEP") == "1"
        if _prof:
            import time as _t

            _read_ms = [0.0]
            _pub_ms = [0.0]

        for robot_name in self.robot_list:
            joint_names = self.robot_joint_map[robot_name]

            if "gripper" in robot_name and len(joint_names) > 1:
                joint_names_all += joint_names[0]
                del joint_names[1:]
            else:
                joint_names_all += joint_names

                if self.enable_ft_sensor:
                    pub_sensor_data = self.get_ft_sensor_data(robot_name)
                    if pub_sensor_data is not None:
                        self.robot_dict[robot_name].publish_force_torque_sensor(pub_sensor_data)

            if _prof:
                _t0 = _t.perf_counter()
                _p = self.get_joint_positions(joint_names)
                _v = self.get_joint_velocities(joint_names)
                _a = self.get_joint_accelerations(joint_names)
                _tq = self.get_joint_torques(joint_names)
                _read_ms[0] += (_t.perf_counter() - _t0) * 1000.0
                _t1 = _t.perf_counter()
                self.robot_dict[robot_name].publish_joint_states(_p, _v, _a, _tq)
                _pub_ms[0] += (_t.perf_counter() - _t1) * 1000.0
            else:
                self.robot_dict[robot_name].publish_joint_states(
                    self.get_joint_positions(joint_names),
                    self.get_joint_velocities(joint_names),
                    self.get_joint_accelerations(joint_names),
                    self.get_joint_torques(joint_names),
                )
            if "chassis" in robot_name:
                pose = self.get_site_pose("chassis")
                self.robot_dict[robot_name].publish_chassis_pose(pose)

            joint_position_command = self.robot_dict[robot_name].get_joint_position_command()
            joint_position_command_all += joint_position_command
            # mode==2 (pure velocity control): the velocity command is the *primary*
            # command and must pass through untouched, ungated by vel_compensation_map —
            # that map only decides whether "velocity feedforward compensation" applies
            # under mode=1 position control. Setting vel_compensation=False on the
            # chassis was meant to disable feedforward in position control, but it also
            # zeroed the mode=2 primary velocity command as a side effect, which made
            # set_joints_velocity completely ineffective for the chassis (measured:
            # vel_cmd arrived at the sim as 0). So mode==2 always uses the velocity
            # command.
            mode = self.robot_dict[robot_name].get_controller_mode()
            if mode == 2 or self.vel_compensation_map[robot_name]:
                joint_velocity_command = self.robot_dict[robot_name].get_joint_velocity_command()
                joint_velocity_command_all += joint_velocity_command
            else:
                joint_velocity_command = [0.0 for i in range(len(joint_position_command))]
                joint_velocity_command_all += joint_velocity_command

            joint_torque_command = self.robot_dict[robot_name].get_joint_torque_command()
            joint_torque_command_all += joint_torque_command

            controller_mode_list += [self.robot_dict[robot_name].get_controller_mode()] * len(
                joint_position_command
            )

        if _prof:
            # Accumulate into the env-level probe (printed together with
            # mujoco_env._prof_mark's 100-step window).
            self._prof_js_read = getattr(self, "_prof_js_read", 0.0) + _read_ms[0]
            self._prof_js_pub = getattr(self, "_prof_js_pub", 0.0) + _pub_ms[0]

        return (
            joint_names_all,
            controller_mode_list,
            joint_position_command_all,
            joint_velocity_command_all,
            joint_torque_command_all,
        )

    def update_trajectory_pose(self, trajectory=False):
        if self.update_trajectory_map is not None:
            for robot_name, trajectory_pose_names in self.update_trajectory_map.items():
                if trajectory is False:
                    pose = self.robot_dict[robot_name].get_endpoint_desired_pose()
                    last_pose = self.multi_robot_ros_interface.trajectory_pose_dict[robot_name][-1]

                    err = np.linalg.norm(np.array(pose[:3]) - np.array(last_pose[:3]))
                    last_update_time = self.get_time() - self.last_pose_time
                    if err > 0.02 or last_update_time > 0.5:
                        self.multi_robot_ros_interface.trajectory_pose_dict[robot_name].pop(0)
                        self.multi_robot_ros_interface.trajectory_pose_dict[robot_name].append(pose)

                        self.last_pose_time = self.get_time()
                        for i in range(len(trajectory_pose_names)):
                            self.set_mocap_pose_with_id(
                                i,
                                self.multi_robot_ros_interface.trajectory_pose_dict[robot_name][i],
                            )
                else:
                    for i in range(len(trajectory_pose_names)):
                        self.set_mocap_pose_with_id(
                            i, self.multi_robot_ros_interface.trajectory_pose_dict[robot_name][i]
                        )

    def update_com_pose(self):
        self.com_pos = copy.deepcopy(self.multi_robot_ros_interface.com_pos)
        self.com_pos = self.pose_add(self.get_chassis_pose(), self.com_pos + [1, 0, 0, 0])[0:3]

    def setup_sensor_scheduler(self, dt):
        """PR-2: build the sim-time sensor scheduler. Called by each backend
        after its model/scene (and thus dt) is known. dt = physics timestep.

        No-op unless ASTRIBOT_USE_SENSOR_SCHEDULER=1 (migration stage 1 default
        off). When on, IMU/chassis-IMU/F-T are driven off sim-time here instead
        of the per-step update_sensor_states() path. LiDAR is registered too but
        its sample fn (_sample_publish_lidar) is backend-specific (abstract).
        See docs/sensor_scheduler_refactor.md §5.2.
        """
        if not self._use_sensor_scheduler:
            return
        from simu_utils.sensor_scheduler import SensorScheduler

        self._sensor_scheduler = SensorScheduler(max_rate_hz=1.0 / dt)
        freq = self.param.get("sensor_frequencies", {}) or {}
        _disabled = set(self.param.get("_disabled_sensors", []))

        # Environment variable override (for pipeline scripts). Takes precedence
        # over sim.yaml. Example: ASTRIBOT_IMU_FREQ=200 overrides freq['imu'].
        if os.getenv("ASTRIBOT_IMU_FREQ"):
            freq["imu"] = float(os.getenv("ASTRIBOT_IMU_FREQ"))
        if os.getenv("ASTRIBOT_LIDAR_FREQ"):
            freq["lidar"] = float(os.getenv("ASTRIBOT_LIDAR_FREQ"))
        if os.getenv("ASTRIBOT_CHASSIS_IMU_FREQ"):
            freq["chassis_imu"] = float(os.getenv("ASTRIBOT_CHASSIS_IMU_FREQ"))

        # M360S LiDAR-integrated IMU + point cloud. Defaults: IMU 100Hz, LiDAR
        # 10Hz. Override per-run via sim.yaml `sensor_frequencies:` or env vars
        # above. IMU cannot exceed physics rate 1/dt (scheduler clamps + warns).
        if any(s in self.sensor_names for s in ("lidar_imu_gyro", "lidar_imu_acc", "lidar_site")):
            self._sensor_scheduler.register(
                "lidar_imu", freq.get("imu", 100.0), self._sample_publish_lidar_imu
            )
            self._sensor_scheduler.register(
                "lidar", freq.get("lidar", 10.0), self._sample_publish_lidar
            )

        # Chassis IMU (9-D Float64MultiArray)
        if (
            "astribot_chassis_base_imu_gyro" in self.sensor_names
            and "astribot_chassis_base_imu_gyro" not in _disabled
        ):
            self._sensor_scheduler.register(
                "chassis_imu", freq.get("chassis_imu", 100.0), self._sample_publish_chassis_imu
            )

        # F/T sensors (configurable, default 50Hz). Only registered when
        # enable_ft_sensor=true.
        if self.enable_ft_sensor:
            for sn in self.sensor_names:
                if "force" in sn and sn not in _disabled:
                    self._sensor_scheduler.register(
                        f"force_{sn}",
                        freq.get("force_torque", 50.0),
                        lambda s=sn: self._sample_publish_force(s),
                    )

        # Start the scheduler anchored to sim_time=0. MuJoCo data.time and Genesis
        # scene.cur_t both start from 0. Without this, tick() is a no-op (_started=False).
        self._sensor_scheduler.start(0.0)

        astribot_simu_log(
            f"[sensor-sched] enabled: {list(self._sensor_scheduler.stats().keys())} "
            f"(max_rate={1.0/dt:.0f}Hz)"
        )

    # ---- sample+publish wrappers (called by scheduler.tick in physics thread) ----

    def _sample_publish_lidar_imu(self):
        imu_msg = self._build_full_imu_msg(stamp=self.multi_robot_ros_interface.get_timestamp())
        if imu_msg is not None:
            pub = self.multi_robot_ros_interface.sensor_ros_pub.get("lidar_imu_publisher")
            if pub is not None:
                pub.publish(imu_msg)

    def _sample_publish_chassis_imu(self):
        msg = self._build_chassis_imu_msg()
        if msg is not None:
            pub = self.multi_robot_ros_interface.sensor_ros_pub.get("chassis_imu_publisher")
            if pub is not None:
                pub.publish(msg)

    def _sample_publish_force(self, sensor_name):
        d = self.get_sensor_data(sensor_name)
        if d is not None and len(d) >= 3:
            m = WrenchStamped()
            m.header.stamp = self.multi_robot_ros_interface.get_timestamp()
            m.wrench.force.x = d[0]
            m.wrench.force.y = d[1]
            m.wrench.force.z = d[2]
            pub = self.multi_robot_ros_interface.sensor_ros_pub.get(sensor_name)
            if pub is not None:
                pub.publish(m)

    def _sample_publish_lidar(self):
        """Backend-specific LiDAR point-cloud sample+publish. Default no-op so
        backends without LiDAR (or before PR-2 wiring) don't crash; MuJoCo /
        Genesis override this. Not @abstractmethod to keep back-compat with any
        env that doesn't publish LiDAR."""
        pass

    def update_sensor_states(self):
        # when the sim-time scheduler is active, sensors are sampled via
        # scheduler.tick() in the physics loop, not here. Skip the legacy path
        # to avoid double-publishing. See docs/sensor_scheduler_refactor.md §6.2.
        if self._use_sensor_scheduler:
            return
        if len(self.sensor_names) > 0:
            # skip sensors disabled by CLI --disable-sensors. They were
            # filtered out of sensor_ros_pub by setup_sensor_interface, so
            # publishing here would KeyError; we just don't call them.
            _disabled = set(self.param.get("_disabled_sensors", []))
            for sensor_name in self.sensor_names:
                if sensor_name in _disabled:
                    continue
                # F/T sensors: published only when enable_ft_sensor=true (off by
                # default in sim to save cost).
                if "force" in sensor_name and not self.enable_ft_sensor:
                    continue
                sensor_data = self.get_sensor_data(sensor_name)
                if "force" in sensor_name and sensor_data is not None and len(sensor_data) >= 3:
                    force_sensor_msg = WrenchStamped()
                    force_sensor_msg.header.stamp = self.multi_robot_ros_interface.get_timestamp()
                    force_sensor_msg.wrench.force.x = sensor_data[0]
                    force_sensor_msg.wrench.force.y = sensor_data[1]
                    force_sensor_msg.wrench.force.z = sensor_data[2]
                    self.multi_robot_ros_interface.sensor_ros_pub[sensor_name].publish(
                        force_sensor_msg
                    )
                # Chassis IMU (real-robot-aligned Float64MultiArray, 9-D).
                # Trigger: sensor_name == 'astribot_chassis_base_imu_gyro' (only).
                # Acc is read inline; the trigger fires once per tick.
                # MUST come BEFORE the generic 'imu' branch — gyro/accel sensors
                # return 3-D arrays, not the 9-D rotation matrix the generic
                # branch expects, so falling through there would KeyError.
                elif sensor_name == "astribot_chassis_base_imu_gyro":
                    msg = self._build_chassis_imu_msg()
                    if msg is not None:
                        self.multi_robot_ros_interface.sensor_ros_pub[
                            "chassis_imu_publisher"
                        ].publish(msg)
                elif sensor_name == "astribot_chassis_base_imu_acc":
                    pass  # Read inline by the gyro branch; no separate publish.
                elif "imu" in sensor_name and sensor_data is not None and len(sensor_data) >= 4:
                    mat_4x4 = np.eye(4)
                    mat_4x4[:3, :3] = sensor_data.reshape(3, 3)
                    imu_quat = quaternion_from_matrix(mat_4x4)
                    imu_msg = Imu()
                    imu_msg.header.stamp = self.multi_robot_ros_interface.get_timestamp()
                    quaternion_msg = Quaternion()
                    quaternion_msg.x = imu_quat[0]
                    quaternion_msg.y = imu_quat[1]
                    quaternion_msg.z = imu_quat[2]
                    quaternion_msg.w = imu_quat[3]
                    imu_msg.orientation = quaternion_msg
                    self.multi_robot_ros_interface.sensor_ros_pub[sensor_name].publish(imu_msg)
                # M360S integrated LiDAR+IMU: assemble a complete Imu msg from gyro +
                # accel + lidar_site orientation (angular_velocity +
                # linear_acceleration + orientation). Triggered when sensor_name is one
                # of lidar_imu_gyro / lidar_imu_acc / lidar_site.
                elif sensor_name in ("lidar_imu_gyro", "lidar_imu_acc", "lidar_site"):
                    imu_msg = self._build_full_imu_msg(
                        stamp=self.multi_robot_ros_interface.get_timestamp()
                    )
                    if imu_msg is not None:
                        self.multi_robot_ros_interface.sensor_ros_pub[
                            "lidar_imu_publisher"
                        ].publish(imu_msg)

    def _build_full_imu_msg(self, stamp):
        """Combine lidar_imu_gyro + lidar_imu_acc + lidar_site orientation into
        one full sensor_msgs/Imu message. frame_id = 'lidar_site'.
        Returns None if any of the three sources is missing.
        """
        gyro = self.get_sensor_data("lidar_imu_gyro")  # (3,) angular velocity
        accel = self.get_sensor_data("lidar_imu_acc")  # (3,) linear acceleration
        site_xmat = self.get_sensor_data("lidar_site")  # (3,3) rotation matrix

        if gyro is None or accel is None or site_xmat is None:
            return None
        if len(gyro) < 3 or len(accel) < 3:
            return None

        # site_xmat in this mujoco-python binding is a (9,) flat row-major rotation
        # matrix. Convert to quaternion manually using trace-based algorithm
        # (avoids the tf_transformations / transforms3d shape assumption bugs).
        # Reference: Shepperd "Quaternion from Rotation Matrix" #244.
        if site_xmat.shape == (4,):
            imu_quat = site_xmat.astype(np.float64)
        elif site_xmat.shape == (9,):
            R = site_xmat.reshape(3, 3)
            trace = R[0, 0] + R[1, 1] + R[2, 2]
            if trace > 0:
                s = 0.5 / np.sqrt(trace + 1.0)
                imu_quat = np.array(
                    [
                        0.25 / s,  # w
                        (R[2, 1] - R[1, 2]) * s,  # x
                        (R[0, 2] - R[2, 0]) * s,  # y
                        (R[1, 0] - R[0, 1]) * s,  # z
                    ]
                )
            else:
                # Find which diagonal is largest to avoid numerical issues
                if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                    s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                    imu_quat = np.array(
                        [
                            (R[2, 1] - R[1, 2]) / s,  # w
                            0.25 * s,  # x
                            (R[0, 1] + R[1, 0]) / s,  # y
                            (R[0, 2] + R[2, 0]) / s,  # z
                        ]
                    )
                elif R[1, 1] > R[2, 2]:
                    s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                    imu_quat = np.array(
                        [
                            (R[0, 2] - R[2, 0]) / s,  # w
                            (R[0, 1] + R[1, 0]) / s,  # x
                            0.25 * s,  # y
                            (R[1, 2] + R[2, 1]) / s,  # z
                        ]
                    )
                else:
                    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                    imu_quat = np.array(
                        [
                            (R[1, 0] - R[0, 1]) / s,  # w
                            (R[0, 2] + R[2, 0]) / s,  # x
                            (R[1, 2] + R[2, 1]) / s,  # y
                            0.25 * s,  # z
                        ]
                    )
        elif site_xmat.shape == (3, 3):
            R = site_xmat
            # (use same logic as above, omitting for brevity)
            trace = R[0, 0] + R[1, 1] + R[2, 2]
            if trace > 0:
                s = 0.5 / np.sqrt(trace + 1.0)
                imu_quat = np.array(
                    [
                        0.25 / s,
                        (R[2, 1] - R[1, 2]) * s,
                        (R[0, 2] - R[2, 0]) * s,
                        (R[1, 0] - R[0, 1]) * s,
                    ]
                )
            else:
                imu_quat = np.array([1.0, 0.0, 0.0, 0.0])  # fallback
        else:
            astribot_simu_log(f"IMU: unexpected site_xmat shape {site_xmat.shape}", level="WARN")
            return None
        imu_msg = Imu()
        imu_msg.header.stamp = stamp
        imu_msg.header.frame_id = "lidar_site"
        # imu_quat is in [w, x, y, z] order (Shepperd convention used by the
        # rotation-matrix-to-quat conversion above). sensor_msgs/Imu uses
        # [x, y, z, w]. PR-3 followup 2026-06-17: the old code did
        # imu_msg.orientation.x = imu_quat[0] etc., which put w into the
        # x slot — SLAM/Fast-LIO then saw a rotated orientation, the IMU
        # pre-integration disagreed with the lidar scan-matching pose, and
        # the SLAM PCD came out as a 'closed triangle' instead of the
        # expected U-shape (visible in s1_v2_rot_20260617_182221 PCD).
        # Fix: explicit [w, x, y, z] -> [x, y, z, w] swizzle.
        imu_msg.orientation.x = imu_quat[1]
        imu_msg.orientation.y = imu_quat[2]
        imu_msg.orientation.z = imu_quat[3]
        imu_msg.orientation.w = imu_quat[0]
        imu_msg.angular_velocity.x = float(gyro[0])
        imu_msg.angular_velocity.y = float(gyro[1])
        imu_msg.angular_velocity.z = float(gyro[2])
        # linear_acceleration in SENSOR BODY frame (local), per SLAM
        # developer's spec 2026-06-18. MuJoCo's accelerometer sensor
        # returns body-frame accel (which is -gravity_local + motion_local);
        # SLAM uses the orientation quaternion to handle gravity
        # internally, so we just publish MuJoCo's raw body-frame output.
        # The earlier "transform to world frame" change (commit 9192f54)
        # was based on a wrong assumption.
        imu_msg.linear_acceleration.x = float(accel[0])
        imu_msg.linear_acceleration.y = float(accel[1])
        imu_msg.linear_acceleration.z = float(accel[2])
        return imu_msg

    def _build_chassis_imu_msg(self):
        """Build std_msgs/Float64MultiArray for chassis IMU (real-robot format).

        9-D layout: [roll, pitch, yaw, wx, wy, wz, ax, ay, az]
            - roll/pitch/yaw: ZYX-intrinsic Euler from site orientation (rad)
            - wx, wy, wz: angular velocity from gyro sensor (rad/s)
            - ax, ay, az: linear acceleration from accel sensor (m/s²), body frame

        Returns None if any source is missing. Frame_id implicitly 'chassis_base'
        (the site is mounted there with no rotation offset).
        """
        gyro = self.get_sensor_data("astribot_chassis_base_imu_gyro")
        accel = self.get_sensor_data("astribot_chassis_base_imu_acc")
        site_xmat = self.get_sensor_data("astribot_chassis_imu_site")

        if gyro is None or accel is None or site_xmat is None:
            return None
        if len(gyro) < 3 or len(accel) < 3:
            return None

        # site_xmat is (9,) flat row-major rotation matrix. Convert to RPY
        # using ZYX-intrinsic (ROS standard). The math: given R = Rz(yaw) * Ry(pitch) * Rx(roll),
        # extract [roll, pitch, yaw] via atan2 on specific matrix elements.
        # Reference: https://www.geometrictools.com/Documentation/EulerAngles.pdf §2.6.
        if site_xmat.shape == (9,):
            R = site_xmat.reshape(3, 3)
        elif site_xmat.shape == (3, 3):
            R = site_xmat
        else:
            return None

        # ZYX-intrinsic (ROS): roll = atan2(R[2,1], R[2,2])
        #                     pitch = atan2(-R[2,0], sqrt(R[2,1]² + R[2,2]²))
        #                       yaw = atan2(R[1,0], R[0,0])
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
        yaw = np.arctan2(R[1, 0], R[0, 0])

        msg = Float64MultiArray()
        msg.data = [
            float(roll),
            float(pitch),
            float(yaw),
            float(gyro[0]),
            float(gyro[1]),
            float(gyro[2]),
            float(accel[0]),
            float(accel[1]),
            float(accel[2]),
        ]
        return msg

    def reindex_states_data(self):
        joint_names = self.joint_names
        controller_modes = self.reindex_string_data(self.controller_mode)
        joint_position_commands = self.reindex_command_data(self.joint_position_command_all)
        joint_velocity_commands = self.reindex_command_data(self.joint_velocity_command_all)
        joint_torque_commands = self.reindex_command_data(self.joint_torque_command_all)

        return (
            joint_names,
            controller_modes,
            joint_position_commands,
            joint_velocity_commands,
            joint_torque_commands,
        )

    def vel_compensation_ctrl(self, ctrl_position, ctrl_velocity=None, idx=None):
        """Compute position target with velocity feedforward compensation.

        Returns position command with velocity feedforward:
        target_pos = ctrl_position + ctrl_velocity * dt

        dt = 1/control_hz, typically 0.02s for control_hz=50

        The lead time must be the *nominal* control period, i.e. the sim time one
        env.step() advances (frame_skip / physics_hz, from resolve_timing). Do NOT
        use self.real_time_fps: both backends overwrite it every step with the
        measured wall-clock step rate (mujoco_env.py / genesis_env.py), so a
        faster-than-realtime run shrinks the lead (~0.003s at 300fps observed) and
        a stall inflates it -- the control law would be modulated by CPU load and
        would not be reproducible.
        """
        dt = self.control_dt

        return [ctrl_position[idx] + ctrl_velocity[idx] * dt]

    def from_matrix(self, matrix):
        return SimuCommonTools.from_matrix(matrix)

    def pose_to_matrix(self, pose):
        return SimuCommonTools.pose_to_matrix(pose)

    def insert_values(self, original_list, index, values):
        return SimuCommonTools.insert_values(original_list, index, values)

    def pose_add(self, pose1, pose2, inv_1_flag=False, inv_2_flag=False):
        return SimuCommonTools.pose_add(pose1, pose2, inv_1_flag, inv_2_flag)

    def trans_depth_image_to_point_cloud(self, depth_img, height, width, camera_name):
        near, far = self.get_near_and_far()
        fovy = self.get_camera_fovy(camera_name)
        transform = self.get_camera_transform(camera_name)

        return SimuCommonTools.trans_depth_image_to_point_cloud(
            depth_img, height, width, fovy, near, far, transform
        )
