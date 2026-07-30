#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025, Astribot Co., Ltd.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------
# Author: Tony Wang, tonywang@astribot.com
# -----------------------------------------------------------------------------

"""
File: astribot_genesis_env.py
Brief: genesis simulation env
"""

import os
import time

import cv2
import genesis as gs
import numpy as np

from astribot_envs.astribot_base_env import AstribotBaseEnv
from astribot_envs.simulation_constants import genesis_viewer_pacing
from simu_utils.simu_common_tools import astribot_simu_log

# MJCF backend renders the floor through MuJoCo's `<texture builtin="checker"
# rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="500" height="500">` paired with
# `<material texrepeat="5 5">`. Genesis 1.0's MJCF parser keeps the texture
# bytes (in mj.tex_data) but renders mjGEOM_PLANE with single-tile UV [0,1]
# and drops texrepeat, so the textured floor falls back to flat grey. We
# generate the same blue-grey checker as a numpy image and attach it to our
# own Plane (added at z=0.001 to win the depth fight against the MJCF floor
# at z=0 without changing physics). astribot_scenes/*/floor/scene.xml stays
# unchanged so MuJoCo backend is unaffected; LiDAR ray-casts still hit the
# MJCF floor geom.
_MJCF_CHECKER_RGB1 = (0.2, 0.3, 0.4)  # = MJCF scene.xml `rgb1`
_MJCF_CHECKER_RGB2 = (0.1, 0.2, 0.3)  # = MJCF scene.xml `rgb2`
_MJCF_CHECKER_TILES = 10  # texrepeat 5 (per side, 2 colors) → 10 cells per axis
_MJCF_CHECKER_PIXELS_PER_TILE = 50  # 10 * 50 = 500 px image, matches MJCF tex width/height


def _build_mjcf_checker_image():
    """Build a numpy uint8 RGB image matching the MJCF scene.xml floor texture.

    Output is a 500x500 RGB image with a 10x10 grid of alternating
    `rgb1`/`rgb2` cells. Returned as `uint8` so genesis.surfaces.ImageTexture
    treats it as already-encoded sRGB (no double-gamma).
    """
    rgb1 = np.array([int(c * 255) for c in _MJCF_CHECKER_RGB1], dtype=np.uint8)
    rgb2 = np.array([int(c * 255) for c in _MJCF_CHECKER_RGB2], dtype=np.uint8)
    n_tiles = _MJCF_CHECKER_TILES
    px = _MJCF_CHECKER_PIXELS_PER_TILE
    img = np.empty((n_tiles * px, n_tiles * px, 3), dtype=np.uint8)
    for ti in range(n_tiles):
        for tj in range(n_tiles):
            color = rgb1 if (ti + tj) % 2 == 0 else rgb2
            img[ti * px : (ti + 1) * px, tj * px : (tj + 1) * px] = color
    return img


class AstribotGenesisEnv(AstribotBaseEnv):
    def __init__(self, param):
        super().__init__(param)

        self.dof_index = []
        self.joint_name_to_index = {}

        # stash sensor configs (yaml-driven) so setup_genesis_lidar/imu
        # can read them. Defaults mirror the MuJoCo PR-2 yaml.
        self.lidar_cfg = param.get("lidar", {}) if hasattr(param, "get") else {}
        # IMU enable derived from sensor_names (same single source of
        # truth as the MuJoCo path — base_env.setup_sensor_interface inspects
        # sensor_names by substring). Removed the legacy `imu.enabled` yaml
        # flag so sim.yaml stays backend-agnostic.
        self._lidar_imu_sensor_names = ("lidar_imu_gyro", "lidar_imu_acc", "lidar_site")

        # opt-in via ASTRIBOT_VIDEO_OUT env var (set by
        # scripts/pipeline/run_slam_pipeline.sh). When set, an off-screen
        # overview camera renders one frame every N sim steps and pipes RGB
        # to ffmpeg. Path is the output mp4 — kept out of yaml because it's
        # per-run, not per-config.
        import os as _os

        self.video_out_path = _os.environ.get("ASTRIBOT_VIDEO_OUT", "")
        self.video_fps = int(_os.environ.get("ASTRIBOT_VIDEO_FPS", "25"))
        self._video_camera = None
        self._video_ffmpeg = None
        self._video_step_counter = 0

        # Chassis freeze: when sim.yaml has `chassis_fixed: true`, step()
        # will pin chassis qpos/qvel to 0 every frame (welded at origin).
        # Matches MuJoCo mujoco_env.py L56-68 chassis_fixed logic.
        self.chassis_fixed = (
            bool(param.get("chassis_fixed", False)) if hasattr(param, "get") else False
        )
        self.chassis_joint_names = []  # will be filled in setup_joint_index_mapping
        self.chassis_dof_indices = []  # dof_idx_local for chassis_x/y/zrot
        self._chassis_vel_kv = float(
            self.param.get("chassis_vel_kv", 2000.0) if hasattr(self.param, "get") else 2000.0
        )
        self._chassis_int_pose = [0.0, 0.0, 0.0]
        self._chassis_int_inited = False
        self._vel_kv_set = set()

        astribot_simu_log("Init genesis viewer mode")
        self.show_viewer = False
        if self.render_mode == "human":
            self.show_viewer = True
        elif self.render_mode == "rgb_array":
            self.show_viewer = False
        # env var override to force headless without re-editing yaml.
        # Pipeline / batch / cluster runs set ASTRIBOT_GENESIS_HEADLESS=1
        # to skip the viewer entirely (~10 ms/step savings — visualizer.update
        # is sync even with no GLFW window). Human interactive runs leave it
        # unset and pick up `mode: human` from yaml as before.
        if os.environ.get("ASTRIBOT_GENESIS_HEADLESS", "0") == "1":
            self.show_viewer = False

        # frame_skip=2 with dt=0.01 → each env.step() advances 2×0.01=0.02s sim
        # time, keeping the 50 Hz env-loop / control rate. Physics ticks at
        # 100 Hz (1/dt), scheduler samples IMU up to 100 Hz across the 2 ticks.
        # frame_skip=4 with dt=0.005 → 200 Hz IMU (4 samples per env.step).
        from astribot_envs.simulation_constants import resolve_timing

        physics_cfg = param.get("physics", {}) if hasattr(param, "get") else {}
        _timing = resolve_timing(physics_cfg)
        self.frame_skip = _timing["frame_skip"]
        # Nominal control period: sim time advanced by one env.step().
        self.control_dt = self.frame_skip * _timing["dt"]

        # SDK-independent chassis yaw rotation for SLAM
        # verification. When ASTRIBOT_GENESIS_ROTATE=<rad/s> is set, step()
        # kinematically advances the chassis_zrot DOF each frame, bypassing the
        # SDK command path entirely. This isolates sensor-quality verification
        # (offset_time undistortion, 200Hz IMU) from the flaky SDK↔sim command
        # channel (QoS mismatch caused zero rotation in 2026-07-06 runs). Value
        # is the yaw rate in rad/s (pipeline default 0.1745 = 36s/rev). Empty
        # or "0" disables injection (normal SDK-driven behavior).
        self.rotate_yaw_rate = float(_os.environ.get("ASTRIBOT_GENESIS_ROTATE", "0") or "0")
        self._injected_zrot = 0.0  # accumulated yaw (rad), advanced per env.step
        # extension: SDK-independent chassis TRANSLATION injection for
        # big-scene SLAM patrol. ASTRIBOT_GENESIS_VX / _VY = body-frame linear
        # rate (m/s). Same in-loop ramp mechanism as ROTATE (write position cmd
        # + velocity feed-forward each frame) — this is what keeps the IMU clean
        # (ROS-topic 50Hz injection jitters the PD target → IMU explosion). x/y
        # are integrated in the chassis body-integration space (chassis_x/y DOFs
        # are the same _int_pose space the OmniChassisController tracks).
        self.inject_vx = float(_os.environ.get("ASTRIBOT_GENESIS_VX", "0") or "0")
        self.inject_vy = float(_os.environ.get("ASTRIBOT_GENESIS_VY", "0") or "0")
        self._injected_x = 0.0
        self._injected_y = 0.0
        # optional delay (s) before translation starts, e.g. spin-in-place first
        self.inject_move_delay = float(_os.environ.get("ASTRIBOT_GENESIS_MOVE_DELAY", "0") or "0")
        self._inject_elapsed = 0.0

        # Teleop (keyboard) via /cmd_vel Twist -> live in-loop injection. Sending
        # only VELOCITY intent (not integrated absolute position) and letting the
        # step loop integrate with the real sim-dt is the correct, jitter-free
        # design: the publisher's wall-clock and the sim clock no longer race,
        # and the PD target advances smoothly in lock-step with the controller
        # (same clean path as the ASTRIBOT_GENESIS_VX/ROTATE injection). Guarded
        # by ASTRIBOT_GENESIS_TELEOP=1 so normal SDK runs are unaffected.
        self.teleop_enabled = _os.environ.get("ASTRIBOT_GENESIS_TELEOP", "0") == "1"
        if self.teleop_enabled and getattr(self, "node", None) is not None:
            from geometry_msgs.msg import Twist
            from rclpy.qos import QoSProfile, ReliabilityPolicy

            # move_delay makes no sense for live teleop — drive immediately.
            self.inject_move_delay = 0.0
            _q = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
            self._teleop_sub = self.node.create_subscription(
                Twist, "/cmd_vel", self._on_cmd_vel, _q
            )
            astribot_simu_log(
                "Genesis teleop: subscribing /cmd_vel (velocity-intent, in-loop integrate)",
                level="DEBUG",
            )

        astribot_simu_log("Init genesis scene", level="DEBUG")

        gs.init(backend=gs.gpu, logging_level="error")
        physics_dt = _timing["dt"]
        self.physics_dt = physics_dt

        # Startup banner, mirroring the MuJoCo backend so both emit one summary
        # line naming the backend, robot, profile and resolved timing.
        astribot_simu_log(
            f"backend=genesis robot={self.param.get('robot_type', '?')} "
            f"profile={getattr(self, 'sim_profile', '?')} "
            f"physics_hz={_timing['physics_hz']:.0f} "
            f"control_hz={_timing['control_hz']:.0f} "
            f"frame_skip={_timing['frame_skip']} dt={physics_dt:.4f}s "
            f"scene={self.param.get('scene', '?')}"
        )

        # Async render (docs/develop/realtime_async_render_refactor.md §4):
        # Genesis viewer natively runs off the physics thread via
        # run_in_thread=True (non-macOS default) and paces itself with
        # realtime_factor — this is the Genesis analogue of MuJoCo's passive
        # viewer + _pace_realtime, but native and lock-decoupled. We map the
        # sim_profile here:
        #   - realtime: refresh_rate = human_render_hz (render decoupled from
        #     physics, low FPS to save GPU), realtime_factor = 1.0 (viewer
        #     paces sim to 1:1 wall time for VR/real-robot sync).
        #   - full/other: refresh_rate = 60, realtime_factor = None (run as
        #     fast as possible, no real-time cap — for training/offline).
        # Genesis does NOT use base_env's _pace_realtime (Genesis step has no
        # precise_sleep); pacing is the viewer's realtime_factor. See §4.2.3.
        if getattr(self, "sim_profile", "custom") == "realtime":
            _refresh_rate, _realtime_factor = genesis_viewer_pacing(
                "realtime",
                getattr(self, "_human_render_every", 5),
                control_hz=_timing.get("control_hz"),
            )
        else:
            _refresh_rate, _realtime_factor = genesis_viewer_pacing("custom", 1)
        self._pace_realtime = (
            getattr(self, "sim_profile", "custom") == "realtime"
            and os.environ.get("ASTRIBOT_PACE_REALTIME", "1") != "0"
        )

        _rt_view = getattr(self, "sim_profile", "custom") == "realtime"
        if _rt_view:
            _vw = int(os.environ.get("ASTRIBOT_VIEWER_W") or 1280)
            _vh = int(os.environ.get("ASTRIBOT_VIEWER_H") or 720)
        else:
            _vw, _vh = (self.width or 1280), (self.height or 720)

        _cam_pos, _cam_lookat, _cam_fov = self._resolve_human_view()

        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                res=(_vw, _vh),
                camera_pos=_cam_pos,
                camera_lookat=_cam_lookat,
                camera_fov=_cam_fov,
                run_in_thread=True,
                refresh_rate=_refresh_rate,
                realtime_factor=_realtime_factor,
            ),
            sim_options=gs.options.SimOptions(
                dt=physics_dt,
                gravity=(0, 0, -10.0),
                substeps=1,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=physics_dt,  # Match sim_options.dt
                constraint_solver=gs.constraint_solver.Newton,
                iterations=15,  # PR-G6: 50 → 15 (s1_v2 mobile mapping is simple)
                ls_iterations=10,  # PR-G6: 50 → 10 (line search)
                max_collision_pairs=80,  # PR-G6: 150 → 80 (s1_v2 ~23 links)
                enable_self_collision=False,  # PR-G6: MJCF already defines contact pairs
                enable_adjacent_collision=False,
            ),
            show_viewer=self.show_viewer,
        )
        # MuJoCo backend renders the floor through MJCF's
        # `<texture builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3">`,
        # which gives a blue-grey checker pattern. Genesis 1.0's MJCF parser
        # reads the texture bytes from mj.tex_data correctly but then renders
        # the `mjGEOM_PLANE` with a single-tile UV `[0, 1]` and ignores
        # `texrepeat="5 5"`, so the textured floor falls back to flat grey
        # (R=G=B=118). We compensate by adding our own Plane WITH a checker
        # surface; this Plane wins the depth fight against the MJCF floor.
        #
        # Z anchor: setup_genesis_model() passes pos=(0,0,0) to gs.morphs.MJCF
        # so the MJCF floor sits at world z=0 (same as the MuJoCo backend).
        # We anchor the visual Plane at z=0.002 (2mm above MJCF floor) — an
        # exact z=0 match works in static-camera renders but causes z-fighting
        # (grey/blue-grey alternating) when the viewer rotates; 2mm clearance
        # makes the depth winner stable across all camera angles.
        # `collision=False` ensures this visual layer doesn't interfere with
        # robot physics (wheels, chassis ground contact); physics still acts
        # against the MJCF floor geom.
        self.scene.add_entity(
            gs.morphs.Plane(pos=(0.0, 0.0, 0.002), collision=False),
            surface=gs.surfaces.Default(
                diffuse_texture=gs.surfaces.ImageTexture(
                    image_array=_build_mjcf_checker_image(),
                ),
            ),
        )
        self.setup_genesis_model()
        self.setup_genesis_camera()
        # off-screen overview camera, opt-in via ASTRIBOT_VIDEO_OUT.
        # Must be added BEFORE scene.build() (add_camera is @gs.assert_unbuilt).
        # Wide-angle 3rd-person view at (3, 0, 2.2) lookat origin — frames the
        # 3-wall room + table without clipping. Off-screen (GUI=False) so we
        # don't open a second viewer window when the human-mode viewer is up.
        if self.video_out_path:
            self._video_camera = self.scene.add_camera(
                res=(640, 480),
                pos=(3.0, 0.0, 2.2),
                lookat=(0.5, 0.0, 0.4),
                fov=55,
                GUI=False,
            )
            astribot_simu_log(
                f"Video capture: off-screen overview cam added, will render to "
                f"{self.video_out_path} at {self.video_fps}fps"
            )
        # sensor setup must happen BEFORE `scene.build()` —
        # `scene.add_sensor` is `@gs.assert_unbuilt`.
        self.setup_genesis_lidar()
        self.setup_genesis_imu()
        self.setup_genesis_chassis_imu()
        self.scene.build()

        # register a sim_time provider so all ROS publishes
        # (IMU, LiDAR, joint_states, cameras, F/T) stamp messages with
        # PHYSICS time rather than wall time. Genesis runs at ~21% real-time
        # on s1_v2 chassis_fixed today (sim_view.mp4 only 18.8s for a 90s
        # wall pipeline). With wall stamps, Fast-LIO's IMU preintegration
        # sees dt ≈ 700ms while the actual physics dt is ≈ 50ms → it
        # integrates yaw_rate 14× too long → screw-shaped SLAM drift (the
        # ±1km bbox we measured post PR-G3). Switching to sim-time stamps
        # collapses that error. MuJoCo is unaffected (no provider → wall
        # fallback). See docs/PR-G2_slam_pipeline_verification.md.
        if hasattr(self, "multi_robot_ros_interface"):
            self.multi_robot_ros_interface.set_sim_time_provider(lambda: float(self.scene.cur_t))

        # set up the sim-time sensor scheduler.
        # dt matches sim_options.dt above (from ASTRIBOT_PHYSICS_DT or default 0.01).
        # step() ticks it with scene.cur_t inside the frame_skip loop. When enabled,
        # this replaces the legacy update_sensor_states() + `_lidar_pub_counter % 5`.
        self.setup_sensor_scheduler(physics_dt)

        # open ffmpeg pipe after build so the camera is ready.
        if self._video_camera is not None:
            import subprocess as _sp

            self._video_ffmpeg = _sp.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-vcodec",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    "640x480",
                    "-r",
                    str(self.video_fps),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-vcodec",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-preset",
                    "ultrafast",
                    self.video_out_path,
                ],
                stdin=_sp.PIPE,
                stderr=_sp.DEVNULL,
            )

        astribot_simu_log("Init robot joint map")
        self.joint_names = [joint for sublist in self.joint_names_list for joint in sublist]
        self._setup_mecanum_adapter()
        self._free_omni_wheel_rollers()
        self.setup_joint_index_mapping()
        self._apply_initial_joint_positions()
        self._apply_controller_config_to_dofs()
        (
            self.joint_names_all,
            self.controller_mode,
            self.joint_position_command_all,
            self.joint_velocity_command_all,
            self.joint_torque_command_all,
        ) = self.update_joint_states()

    def step(self, action: np.ndarray) -> tuple:
        self.update_reset_flag()
        if not self.reset_flag:
            step_begin_time = time.time()
            if os.environ.get("ASTRIBOT_PROFILE_STEP") == "1":
                if hasattr(self, "_prof_prev_begin"):
                    self._prof_paced_wall = getattr(self, "_prof_paced_wall", 0.0) + (
                        step_begin_time - self._prof_prev_begin
                    )
                    self._prof_paced_n = getattr(self, "_prof_paced_n", 0) + 1
                self._prof_prev_begin = step_begin_time
            self._state_cache_tick = getattr(self, "_state_cache_tick", 0) + 1
            _prof_steps = int(os.environ.get("ASTRIBOT_GENESIS_PROFILE_STEPS", "0"))
            _prof_stepwise = _prof_steps > 0 and not hasattr(self, "_prof_done")
            _prof_avg = os.environ.get("ASTRIBOT_PROFILE_STEP") == "1"
            _prof_active = _prof_stepwise or _prof_avg
            if _prof_active:
                _t = time.perf_counter()
                _phase = {}

                def _mark(name):
                    nonlocal _t
                    now = time.perf_counter()
                    _phase[name] = (now - _t) * 1000.0
                    _t = now

            else:

                def _mark(name):
                    pass

            (
                self.joint_names_all,
                self.controller_mode,
                self.joint_position_command_all,
                self.joint_velocity_command_all,
                self.joint_torque_command_all,
            ) = self.update_joint_states()
            _mark("update_joint_states")

            # SDK-independent chassis yaw injection. When
            # ASTRIBOT_GENESIS_ROTATE=<rad/s> is set, override the chassis_zrot
            # POSITION COMMAND (and velocity feed-forward) with a smooth ramp,
            # so the existing PD control path (mode 1) tracks a steadily
            # rotating target. This works WITH the controller instead of
            # teleporting the joint (set_dofs_position mid-loop fights the PD
            # target → violent oscillation, IMU garbage). Bypasses the flaky
            # SDK↔sim command channel (QoS mismatch left the chassis static in
            # 2026-07-06 runs) to isolate sensor verification (offset_time
            # undistortion, 200 Hz IMU). dt_env = frame_skip × dt = 0.02s.
            if self.rotate_yaw_rate and "astribot_chassis_zrot" in self.joint_name_to_index:
                dt_env = self.frame_skip * self.physics_dt
                self._injected_zrot += self.rotate_yaw_rate * dt_env
                # joint_position_command_all is aligned position-for-position
                # with joint_names_all (both built in the same robot/joint
                # iteration order in base_env.update_joint_states), so the flat
                # command index is just zrot's slot in joint_names_all.
                if "astribot_chassis_zrot" in self.joint_names_all:
                    zrot_flat = self.joint_names_all.index("astribot_chassis_zrot")
                    if zrot_flat < len(self.joint_position_command_all):
                        self.joint_position_command_all[zrot_flat] = self._injected_zrot
                    if zrot_flat < len(self.joint_velocity_command_all):
                        self.joint_velocity_command_all[zrot_flat] = self.rotate_yaw_rate

            # extension: chassis body-frame TRANSLATION injection (VX/VY).
            # Same ramp-the-position-cmd + velocity-feed-forward approach as the
            # yaw block above. Integrates in body space (chassis_x/y DOFs). An
            # optional MOVE_DELAY lets the robot spin in place first, then drive.
            if (self.inject_vx or self.inject_vy):
                dt_env = self.frame_skip * self.physics_dt
                self._inject_elapsed += dt_env
                if self._inject_elapsed >= self.inject_move_delay:
                    self._injected_x += self.inject_vx * dt_env
                    self._injected_y += self.inject_vy * dt_env
                    for dof_name, acc_val, vel_val in (
                        ("astribot_chassis_x", self._injected_x, self.inject_vx),
                        ("astribot_chassis_y", self._injected_y, self.inject_vy),
                    ):
                        if dof_name in self.joint_names_all:
                            flat = self.joint_names_all.index(dof_name)
                            if flat < len(self.joint_position_command_all):
                                self.joint_position_command_all[flat] = acc_val
                            if flat < len(self.joint_velocity_command_all):
                                self.joint_velocity_command_all[flat] = vel_val

            # NOTE: do NOT call self.reindex_states_data() here. It is a
            # legacy path designed for a stale 5-finger gripper MJCF layout
            # (gripper_left_L1/L2/R1/R2/L11 — 5 joints per gripper). The
            # current yaml uses a single joint per gripper, so command/state
            # arrays are 26 entries; reindex_command_data inserts 3 zero
            # placeholders at index 14 and 27 to pad to 32, which shifts
            # arm_left_6/arm_left_7/gripper_left commands forward by 3 slots,
            # but joint_names_all stays at 26 entries — so step()'s loop
            # ends up sending arm_left_6's command to arm_right_1 (and so
            # on for all joints after index 14). Result: left/right arms
            # moved asymmetrically. MuJoCo env never calls reindex_states_data
            # either, so this also brings Genesis behavior in line with
            # MuJoCo.

            self.update_object_states()
            _mark("update_object_states")
            self.update_trajectory_pose()
            _mark("update_trajectory_pose")
            self.update_com_pose()
            _mark("update_com_pose")
            # genesis overrides update_sensor_states() to publish via
            # gs.sensors (lidar + imu) instead of MuJoCo's sensor API.
            self.update_sensor_states()
            _mark("update_sensor_states (incl IMU+LiDAR pub)")
            # capture frame every 2 sim steps (50Hz/2 = 25fps)
            # if ASTRIBOT_VIDEO_OUT was set at startup.
            self._capture_video_frame()
            _mark("_capture_video_frame")

            pos_target = []  # PD position target (mode 1)
            pos_vel_target = []  # PD position-mode velocity feed-forward (mode 1)
            pos_dof_index = []
            vel_target = []  # VELOCITY-mode target (mode 2)
            vel_dof_index = []
            force_ctrl_data = []
            force_dof_index = []
            # B-7 follow-up (2026-06-23): base_env.update_joint_states
            # returns joint_names_all / controller_mode indexed by
            # FLAT joint (one entry per joint, after the gripper
            # special-case that drops duplicate gripper entries).
            # s1_v2's joint_names_all is therefore 26 entries long
            # (3 chassis + 4 torso + 2 head + 7 arm_left + 1 gripper_left +
            # 7 arm_right + 1 gripper_right + the duplicate-gripper
            # merges), and controller_mode has the same length
            # (one mode per joint). Both index self.joint_names_list
            # at the same offsets if you walk the robot components
            # and their joints in order. joint_name_to_index keys on
            # the same flat joint names (set up in
            # setup_joint_index_mapping from self.joint_names). We
            # therefore iterate joint_id from 0..len(joint_names_all),
            # which is safe and matches base_env's contract.
            if not hasattr(self, "joint_names_all") or self.joint_names_all is None:
                self.joint_names_all = []
            mec_chassis = getattr(self, "_mecanum_chassis_joints", set())
            mec_pose = {}

            self._rotate_chassis_vel_body_to_world()

            for joint_id in range(min(len(self.controller_mode), len(self.joint_names_all))):
                mode = self.controller_mode[joint_id]
                joint_name = self.joint_names_all[joint_id]
                if joint_name in mec_chassis:
                    mec_pose[joint_name] = self.joint_position_command_all[joint_id]
                    continue
                if mode == 1:
                    # Position control with velocity feed-forward. Genesis's
                    # control_dofs_position_velocity is a true PD controller —
                    # the target is the SDK's raw position command, NOT the
                    # MuJoCo-style "position + velocity * dt" mixture (which
                    # only made sense for MuJoCo's actuator gainprm/biasprm).
                    pos_target += [self.joint_position_command_all[joint_id]]
                    pos_vel_target += [self.joint_velocity_command_all[joint_id]]
                    pos_dof_index += [self.joint_name_to_index[joint_name]]
                elif mode == 2:
                    # Velocity control. Use Genesis's dedicated
                    # control_dofs_velocity (CTRL_MODE.VELOCITY) which
                    # applies `force = -act_bias[2] * (ctrl_vel - vel)`
                    # — a clean velocity-PI without any position term.
                    #
                    # Earlier attempt used control_dofs_position_velocity
                    # with `pos_target = current_pos` to "hold position
                    # while feeding velocity", but Genesis's POSITION-mode
                    # force formula is
                    #   force = act_gain*(ctrl_pos - pos)
                    #         + act_bias[0]
                    #         + (act_gain + act_bias[1]) * pos
                    #         + act_bias[2] * (vel - ctrl_vel)
                    # — the `(act_gain + act_bias[1]) * pos` term (i.e.
                    # the spring pulling the joint toward zero, which
                    # cancels in pure-PD only when gain == -bias[1])
                    # dominates for actuators whose gainprm doesn't
                    # equal -biasprm[1]. The gripper actuator declares
                    # `gainprm="4.65 0 0" biasprm="0 -500 -10"`, so
                    # `act_gain + act_bias[1] = -495.35`, which slams
                    # the gripper toward pos=0 no matter what velocity
                    # we feed forward. Result: SDK velocity command
                    # ignored; gripper doesn't move. Switching to
                    # control_dofs_velocity removes the position term
                    # entirely.
                    vel_target += [self.joint_velocity_command_all[joint_id]]
                    vel_dof_index += [self.joint_name_to_index[joint_name]]
                    if joint_name not in getattr(self, "_vel_kv_set", set()):
                        _idx = self.joint_name_to_index[joint_name]
                        try:
                            import numpy as _np
                            self.robot.set_dofs_kv(
                                _np.array([self._chassis_vel_kv], dtype=_np.float32), [_idx])
                            if not hasattr(self, "_vel_kv_set"):
                                self._vel_kv_set = set()
                            self._vel_kv_set.add(joint_name)
                            astribot_simu_log(
                                f"velocity control: {joint_name} dof={_idx} set "
                                f"kv={self._chassis_vel_kv} (kv=0 would leave velocity "
                                f"commands with no authority)", level="DEBUG")
                        except Exception as _e:
                            astribot_simu_log(
                                f"Failed to set chassis kv: {_e}", level="WARN")
                elif mode == 3:
                    force_ctrl_data += [self.joint_torque_command_all[joint_id]]
                    force_dof_index += [self.joint_name_to_index[joint_name]]

            # Chassis freeze (pre-step): when chassis_fixed=true, override
            # chassis position targets to 0 so the PD controller doesn't
            # try to follow the SDK's chassis commands. Mirrors MuJoCo's
            # ctrl[chassis_actuator_ids] = 0.0 in mujoco_env.py:138-140.
            # (chassis is normally position-controlled — handle both
            # buckets to be safe.)
            if self.chassis_fixed and self.chassis_dof_indices:
                for i, idx in enumerate(pos_dof_index):
                    if idx in self.chassis_dof_indices:
                        pos_target[i] = 0.0
                        pos_vel_target[i] = 0.0
                for i, idx in enumerate(vel_dof_index):
                    if idx in self.chassis_dof_indices:
                        vel_target[i] = 0.0

            if self.mecanum_adapter is not None and mec_pose and not self.chassis_fixed:
                self._apply_mecanum(mec_pose)

            # Dynamics feed-forward is computed ONCE per control period, outside the
            # frame_skip loop, to match the MuJoCo backend (which evaluates it once
            # and holds it across its physics substeps). Keeping the two backends on
            # the same feed-forward update rate is what makes their tracking numbers
            # comparable; recomputing per substep here would silently give Genesis a
            # frame_skip-times higher-rate feed-forward than MuJoCo.
            #
            # The control *commands* still have to be re-issued every substep (see
            # the ctrl_mode note below), only the feed-forward is hoisted.
            _comp_offsets = (
                self.joint_space_compensation_offsets(pos_dof_index)
                if pos_dof_index
                else {}
            )
            _have_joint_space_comp = bool(_comp_offsets)
            _pos_target = (
                [p + _comp_offsets.get(d, 0.0) for p, d in zip(pos_target, pos_dof_index)]
                if _have_joint_space_comp
                else pos_target
            )
            _torque_ff = (
                self._torque_mode_feedforward(force_dof_index) if force_ctrl_data else {}
            )
            _force_ctrl = [
                f + _torque_ff.get(d, 0.0)
                for f, d in zip(force_ctrl_data, force_dof_index)
            ]

            for _ in range(self.frame_skip):
                # Genesis API distinction (CRITICAL):
                #   set_dofs_position(...)            — directly sets joint state
                #                                       (kinematic reset, NOT control)
                #   control_dofs_position(...)        — PD position-mode target
                #   control_dofs_position_velocity(.) — PD position-mode + vel ff
                #   control_dofs_velocity(...)        — pure velocity-mode target
                #   control_dofs_force(...)           — direct force/torque
                # Each call sets ctrl_mode for the touched dofs; downstream
                # dofs that aren't touched keep their previous ctrl_mode,
                # so re-issuing per-step (inside frame_skip) is required
                # for mode switching to take effect immediately.
                if pos_dof_index:
                    # _pos_target already carries the feed-forward folded in (see
                    # joint_space_compensation_offsets for why it cannot go through
                    # qf_applied); it was computed once above, per control period.
                    self.robot.control_dofs_position_velocity(
                        _pos_target, pos_vel_target, pos_dof_index
                    )
                if vel_dof_index:
                    self.robot.control_dofs_velocity(vel_target, vel_dof_index)
                if force_ctrl_data:
                    # mode=3 (Zero-G) aligned with the real robot's chain:
                    #   force = tau_cmd + (C+g compensation) - kd * qd
                    # control_dofs_force is a straight pass-through (no parasitic
                    # spring, unlike MuJoCo's affine actuator), so the feed-forward
                    # and the small damping term are added explicitly. _force_ctrl
                    # already carries both, computed once above per control period.
                    self.robot.control_dofs_force(_force_ctrl, force_dof_index)

                if self.mecanum_adapter is not None and mec_pose and not self.chassis_fixed:
                    self.mecanum_adapter.reissue_wheel_speeds()

                # Link-COM anti-gravity path. Skipped when the joint-space path
                # already compensated every position-controlled dof, otherwise the
                # two would stack and the robot would be pushed upward by ~m*g.
                if not _have_joint_space_comp:
                    self._apply_dynamics_compensation_step()

                self.scene.step()

                if self._chassis_int_pose is not None:
                    try:
                        vx_b, vy_b, wz = self._read_chassis_body_twist()
                        self._chassis_int_pose[0] += vx_b * self.physics_dt
                        self._chassis_int_pose[1] += vy_b * self.physics_dt
                        self._chassis_int_pose[2] += wz * self.physics_dt
                    except Exception:
                        pass

                # tick the sim-time scheduler after every physics step
                # (inside frame_skip loop). Sensors that are due (IMU/LiDAR/F-T)
                # sample+publish here on sim-time alignment.
                if self._sensor_scheduler is not None:
                    self._sensor_scheduler.tick(float(self.scene.cur_t))

                # old 200Hz IMU publish path (PR-G6 hardcoded after each
                # scene.step). During migration stage 1 (scheduler off by default),
                # keep this for back-compat. Stage 3: delete.
                if not self._use_sensor_scheduler:
                    # Publish IMU after each physics step to achieve
                    # 200 Hz IMU sampling. With dt=0.005 + frame_skip=4, we execute
                    # 4 physics steps per env.step. Publishing IMU after each step
                    # gives 4 samples per env.step × 50 Hz env loop = 200 Hz IMU.
                    # This matches real hardware (Livox MID-360 + IMU at 200 Hz).
                    # LiDAR stays in update_sensor_states() at 50 Hz / 5 = 10 Hz.
                    self._publish_imu_genesis()

                # Chassis freeze (post-step): force chassis qpos/qvel to 0.
                # Even with PD target=0, numerical drift / contact forces can
                # nudge the chassis. Explicitly zero qpos+qvel to defeat drift,
                # matching mujoco_env.py:150-153. set_dofs_position(zero_velocity=True)
                # does both in one call.
                if self.chassis_fixed and self.chassis_dof_indices:
                    self.robot.set_dofs_position(
                        [0.0] * len(self.chassis_dof_indices),
                        self.chassis_dof_indices,
                        zero_velocity=True,
                    )
            _mark(f"scene.step × {self.frame_skip} (control + physics)")

            if self.mecanum_adapter is not None and not self.chassis_fixed:
                self.mecanum_adapter.integrate_state(self.frame_skip * self.physics_dt)

            self.render()
            _mark("render (visualizer.update + camera trigger)")

            step_end_time = time.time()
            self.real_time_fps = 1 / (step_end_time - step_begin_time)
            self.report_realtime_factor(float(self.scene.cur_t))

            from astribot_envs.simulation_constants import RTF_WARNING_THRESHOLD

            sim_time_delta = self.frame_skip * self.physics_dt
            wall_time_delta = step_end_time - step_begin_time
            rtf = sim_time_delta / wall_time_delta if wall_time_delta > 0 else 0.0
            if not hasattr(self, "_rtf_window"):
                from collections import deque

                self._rtf_window = deque(maxlen=50)
                self._rtf_warn_counter = 0
            self._rtf_window.append(rtf)
            self.rtf_avg = sum(self._rtf_window) / len(self._rtf_window)
            self._rtf_warn_counter += 1
            # Throttled to DEBUG: report_realtime_factor() already prints the RTF
            # every ASTRIBOT_RTF_REPORT_SEC seconds, so this per-window line is
            # redundant on INFO. The step-count gate fires once per 50 steps, which
            # at control_hz=50 was once per second -- pure log spam on Genesis, and
            # it never showed on MuJoCo only because RTF stays above the threshold.
            if (
                len(self._rtf_window) == self._rtf_window.maxlen
                and self.rtf_avg < RTF_WARNING_THRESHOLD
                and self._rtf_warn_counter >= 50
            ):
                self._rtf_warn_counter = 0
                astribot_simu_log(
                    f"Sliding-mean RTF {self.rtf_avg:.3f} < {RTF_WARNING_THRESHOLD} "
                    f"(mean over last {self._rtf_window.maxlen} steps): physics is "
                    f"slower than real time. In the realtime profile, lower "
                    f"ASTRIBOT_HUMAN_RENDER_HZ or close the viewer window.",
                    level="DEBUG",
                )

            if getattr(self, "_pace_realtime", False):
                now = time.time()
                if not hasattr(self, "_pace_next"):
                    self._pace_next = now
                self._pace_next += sim_time_delta
                remaining = self._pace_next - now
                if remaining > 0.0015:
                    time.sleep(remaining - 0.001)
                    while time.time() < self._pace_next:
                        pass
                elif remaining > 0:
                    while time.time() < self._pace_next:
                        pass
                else:
                    self._pace_next = time.time()

            if _prof_active:
                tot = sum(_phase.values())
                if _prof_stepwise:
                    if not hasattr(self, "_prof_count"):
                        self._prof_count = 0
                        self._prof_max = _prof_steps
                    self._prof_count += 1
                    phases = " | ".join(f"{k}={v:.1f}ms" for k, v in _phase.items())
                    print(f"[PROF #{self._prof_count}] total={tot:.1f}ms  {phases}", flush=True)
                    if self._prof_count >= self._prof_max:
                        self._prof_done = True
                        print(f"[PROF] done after {self._prof_max} steps", flush=True)
                if _prof_avg:
                    if not hasattr(self, "_prof_acc"):
                        from collections import OrderedDict

                        self._prof_acc = OrderedDict()
                        self._prof_avg_n = 0
                    for _k, _v in _phase.items():
                        self._prof_acc[_k] = self._prof_acc.get(_k, 0.0) + _v
                    self._prof_avg_n += 1
                    if self._prof_avg_n >= 100:
                        _n = self._prof_avg_n
                        _tot = sum(self._prof_acc.values())
                        parts = "  ".join(f"{k}={v/_n:.1f}ms" for k, v in self._prof_acc.items())
                        _sim_dt_ms = self.frame_skip * self.physics_dt * 1000.0
                        _raw_rtf = _sim_dt_ms / (_tot / _n) if _tot > 0 else 0.0
                        _pace_note = ""
                        if getattr(self, "_prof_paced_n", 0) > 0:
                            _paced_ms = self._prof_paced_wall / self._prof_paced_n * 1000.0
                            _paced_rtf = _sim_dt_ms / _paced_ms if _paced_ms > 0 else 0.0
                            _pace_note = (
                                f"  |  paced={_paced_ms:.1f}ms/step "
                                f"(paced RTF~{_paced_rtf:.2f}, sim_dt={_sim_dt_ms:.0f}ms)"
                            )
                            self._prof_paced_wall = 0.0
                            self._prof_paced_n = 0
                        astribot_simu_log(
                            f"[prof-step] mean over last {_n} steps: {parts}  "
                            f"total={_tot/_n:.1f}ms/step "
                            f"(raw RTF~{_raw_rtf:.2f}){_pace_note}"
                        )
                        self._prof_acc.clear()
                        self._prof_avg_n = 0

        else:
            self.reset()

            self.reset_flag = False

        observation = self._get_obs()
        reward = 0
        terminated = False
        info = self._get_info()

        return observation, reward, terminated, False, info

    def _on_cmd_vel(self, msg):
        """Teleop: map Twist -> in-loop injection rates. Only stores intent;
        the step loop integrates position with the true sim-dt (jitter-free)."""
        self.inject_vx = float(msg.linear.x)
        self.inject_vy = float(msg.linear.y)
        self.rotate_yaw_rate = float(msg.angular.z)

    def reset(self, seed=None, options=None):
        astribot_simu_log("Reset scene")
        super().reset(seed=seed)
        self.scene.reset()
        self._apply_initial_joint_positions()

        self.reset_flag = False
        if getattr(self, "mecanum_adapter", None) is not None:
            self.mecanum_adapter.reset_feedforward()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _get_obs(self) -> np.ndarray:
        pose = list()
        if self.object_names:
            for object_name in self.object_names:
                pose = self.get_body_pose(object_name)
        else:
            pose = np.zeros(7)
        return pose

    def _get_info(self) -> dict:
        robot_info = dict()

        for robot_name in self.robot_list:
            joint_names = self.robot_joint_map[robot_name]

            if "gripper" in robot_name:
                del joint_names[1:]
                robot_info[robot_name] = self.get_joint_positions(joint_names)
            elif "chassis" in robot_name:
                pose = self.get_site_pose("chassis")
                robot_info[robot_name] = pose
            else:
                robot_info[robot_name] = self.get_joint_positions(joint_names)

        return robot_info

    def render(self):
        # skip visualizer.update when show_viewer=False. In Genesis 1.2,
        # this call still does ~10 ms of off-screen pipeline work (camera
        # framebuffer transfers, even when no GLFW window exists) — see
        # PROF data in docs/PR-G2 §Genesis Backend §PR-G5. The visualizer
        # is only needed when the human viewer is up.
        #
        if self.show_viewer:
            n = getattr(self, "_human_render_every", 1)
            self._genesis_render_counter = getattr(self, "_genesis_render_counter", 0) + 1
            if n <= 1 or self._genesis_render_counter % n == 0:
                self.scene.visualizer.update(force=False)
        self.update_camera_pose()
        for camera_name in self.camera_names:
            camera_data = self.render_single_camera(camera_name)
            if camera_data is not None:
                self._publish_camera_data(camera_name, camera_data)

    def close(self):
        pass

    def setup_genesis_model(self):
        astribot_simu_log("Setup genesis model with mjcf")
        # W1-PR4: Genesis 1.0's MJCF loader (genesis/utils/mjcf.py) re-parses the
        # inlined root via mujoco.MjModel.from_xml_string(data) which loses the
        # source XML's directory context — relative <include> and mesh paths then
        # resolve against cwd, not the entry MJCF. The factory installs a
        # build_model monkey-patch (in monkeypatch_genesis_mjcf.py) that fixes
        # this. Once that patch is in place, no chdir gymnastics are needed here.
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file=self.model_path,
                # drop the historical entity-level
                # +5cm cushion. Reason: Genesis applies `gs.morphs.MJCF(pos=...)`
                # as an entity-wide world offset, so a non-zero value lifts
                # EVERYTHING in the MJCF — worldbody floor + chassis_base body +
                # the whole robot — together. The MJCF authors chassis_base at
                # local z=0.097 (with wheels at z=0.082, on a floor geom at z=0),
                # i.e. the model is already designed to sit on a z=0 ground.
                # MuJoCo backend uses the same MJCF unaltered (chassis_base
                # world z = 0.097 ✓); Genesis was inconsistently lifting it
                # to 0.147 because of the legacy 0.05 here (introduced in
                # commit 7438414 with no comment, no reason). Setting pos=0
                # brings Genesis world frame in line with MuJoCo and with the
                # MJCF authoring intent — SDK trajectories targeting absolute
                # heights now mean the same thing across both backends.
                pos=(0, 0, 0),
                euler=(0, 0, 0),
            ),
            material=gs.materials.Rigid(gravity_compensation=self._resolve_gravity_compensation()),
        )

    def _resolve_gravity_compensation(self):
        is_floating = (
            getattr(self, "param", None) is not None
            and self.param.get("chassis_type") == "omni"
            and not self.chassis_fixed
        )
        if is_floating:
            # Expected for every omni-wheel robot: Genesis applies material
            # gravity_compensation entity-wide, which would make the whole robot
            # (and loose scene objects) float. Per-joint compensation is applied
            # at the control layer instead, so behaviour matches MuJoCo.
            # DEBUG, not WARN -- this is the designed path, not a problem.
            astribot_simu_log(
                "Floating-base chassis (chassis_type=omni): material "
                "gravity_compensation forced to 0 so the robot keeps normal "
                "gravity; per-joint compensation is applied by the controller.",
                level="DEBUG",
            )
            return 0.0

        # Fixed-base robots: also keep entity gravity at full strength and let the
        # control layer compensate, so both chassis configurations and both backends
        # share one compensation path (controller_config.dynamics_compensation).
        #
        # The old code read self.gravity_compensation here, an attribute that was
        # removed when controller_config replaced it (see astribot_base_env.py's
        # "Controller configuration (replaces gravity_compensation)"). Nothing ever
        # assigned it again, so every non-floating config raised AttributeError and
        # the Genesis backend could not start at all.
        return 0.0

    def setup_genesis_camera(self):
        astribot_simu_log("Setup genesis camera")

        _display = self.param.get("display", {}) or {}
        _cam_res = _display.get("camera_resolutions") or {}
        _cam_ext_raw = _display.get("camera_extrinsics") or {}
        default_width = self.width or 640
        default_height = self.height or 480

        self.camera_extrinsics = {}
        for name, config in _cam_ext_raw.items():
            if isinstance(config, dict):
                from simu_utils.sim_config import CameraExtrinsic

                self.camera_extrinsics[name] = CameraExtrinsic(
                    parent_link=config["parent_link"],
                    local_pos=tuple(config["local_pos"]),
                    local_quat=tuple(config["local_quat"]),
                    fovy=float(config["fovy"]),
                )
            else:
                self.camera_extrinsics[name] = config

        for camera_name in self.camera_names:
            normalized_name = camera_name
            if camera_name == "head_rgbd":
                normalized_name = "astribot_head"
            elif camera_name in ("left_wrist_rgbd", "astribot_arm_left_effector"):
                normalized_name = "astribot_arm_left_effector"
            elif camera_name in ("right_wrist_rgbd", "astribot_arm_right_effector"):
                normalized_name = "astribot_arm_right_effector"

            if camera_name in _cam_res:
                width, height = _cam_res[camera_name]
            else:
                width, height = default_width, default_height
            res = (int(width), int(height))

            has_extrinsics = camera_name in self.camera_extrinsics
            if has_extrinsics:
                extrinsic = self.camera_extrinsics[camera_name]
                fov = extrinsic.fovy

                init_pos = (0.0, 0.0, 2.0)
                init_lookat = (1.0, 0.0, 2.0)
            else:
                fov = 51.4175
                init_pos = None
                init_lookat = None

            if normalized_name == "astribot_head":
                if not has_extrinsics:
                    init_pos = (2.0, 0.0, 1.5)
                    init_lookat = (0.0, 0.0, 1.0)

                self.head_camera = self.scene.add_camera(
                    res=res,
                    pos=init_pos,
                    lookat=init_lookat,
                    fov=fov,
                    GUI=False,
                )
            elif normalized_name == "astribot_arm_left_effector":
                self.left_camera = self.scene.add_camera(
                    res=res,
                    pos=(3.8, 0.0, 2.5),
                    lookat=(0, 0, 0.5),
                    fov=30,
                    GUI=False,
                )
            elif normalized_name == "astribot_arm_right_effector":
                self.right_camera = self.scene.add_camera(
                    res=res,
                    pos=(4.2, 0.0, 2.5),
                    lookat=(0, 0, 0.5),
                    fov=30,
                    GUI=False,
                )
            elif normalized_name == "astribot_global_camera":
                self.global_camera = self.scene.add_camera(
                    res=res,
                    pos=(0.5, 0.011, 2.0),
                    lookat=(0.5, 0.011, 0.0),
                    up=(-1, 0, 0),
                    fov=70,
                    GUI=False,
                )

    def setup_genesis_lidar(self):
        """PR-5: M360S LiDAR via Genesis Raycaster sensor (= Lidar in GS 1.0 API).

        PR-G2 (G2-B): Attached to lidar_front link (not chassis_base) so the
        MJCF body pose fully determines the sensor world pose. pos_offset and
        euler_offset are both zero — the MJCF lidar_mid360_front.xml already
        specifies the calibrated mount (s1_v2: pos="0.17393 0.16893 0.073397",
        quat yaw=-45°). This eliminates the 2cm z-offset + 0.86° yaw error the
        old hardcoded offsets had vs the MJCF ground truth.

        `Raycaster` is aliased to `Lidar` in `genesis.options.sensors.__init__`.

        Fidelity gap (R5.0): Genesis 1.0's SphericalPattern produces a uniform
        angular grid, NOT the Livox MID-360 non-repeating scan pattern. For
        the MuJoCo backend we get the non-repeating 24000-ray pattern from
        `mujoco_lidar.scan_gen.LivoxGenerator`. Replicating that on Genesis
        needs a custom `RaycastPattern` subclass (compute_ray_dirs override)
        feeding in the same mid360.npy angles — tracked as PR-6.

        `return_world_frame=False` keeps points in the sensor local frame,
        matching the `frame_id='lidar_site'` convention the livox CustomMsg
        publisher already uses (PR-3).
        """
        cfg = getattr(self, "lidar_cfg", None)
        # Honour the profile mask, exactly as the MuJoCo backend does. Without this
        # the realtime profile still built a 24000-ray Raycaster that nothing ever
        # read: pure setup cost, plus a misleading "Raycaster on ..." log line
        # suggesting LiDAR was active.
        if getattr(self, "_lidar_force_disabled", False) or not (cfg or {}).get("enabled", False):
            self.lidar_sensor = None
            return
        if not cfg:
            self.lidar_sensor = None
            return
        # when scan_pattern is 'mid360' use the real Livox non-repeating
        # scan table (LivoxMid360Pattern) instead of the uniform SphericalPattern
        # grid. Any other pattern name keeps the legacy grid, so other robots /
        # configs are unaffected. self.lidar_pattern is None for the grid path.
        self.lidar_pattern = None
        scan_pattern = str(cfg.get('scan_pattern', '')).lower()
        if scan_pattern == 'mid360':
            try:
                from simu_utils.livox_mid360_pattern import LivoxMid360Pattern
                self.lidar_pattern = LivoxMid360Pattern(
                    samples=int(cfg.get('samples', 24000)),
                    downsample=int(cfg.get('downsample', 1)),
                )
                pattern = self.lidar_pattern
                pattern_desc = f"mid360 non-repeating rays={self.lidar_pattern.return_shape[0]}"
            except Exception as e:
                astribot_simu_log(
                    f"LivoxMid360Pattern unavailable, falling back to grid: {e}", level="WARN"
                )
        if self.lidar_pattern is None:
            n_points = tuple(cfg.get('n_points', (200, 120)))
            pattern = gs.options.sensors.SphericalPattern(fov=(360.0, 30.0), n_points=n_points)
            pattern_desc = f"grid n_points={n_points}"
        try:
            self.lidar_sensor = self.scene.add_sensor(
                gs.options.sensors.Raycaster(
                    entity_idx=self._robot_entity_idx(),
                    link_idx_local=self._lidar_front_link_idx(),
                    pos_offset=(0.0, 0.0, 0.0),
                    euler_offset=(0.0, 0.0, 0.0),
                    pattern=pattern,
                    min_range=float(cfg.get('min_dist', 0.1)),
                    max_range=float(cfg.get('cutoff_dist', 40.0)),
                    return_world_frame=False,
                )
            )
            astribot_simu_log(
                f"LiDAR Raycaster on lidar_front link, {pattern_desc}, "
                f"max_range={cfg.get('cutoff_dist', 40.0)}m"
            )
        except Exception as e:
            astribot_simu_log(f"LiDAR Raycaster setup failed: {e}", level="WARN")
            self.lidar_sensor = None
            self.lidar_pattern = None

    def setup_genesis_imu(self):
        """PR-5: IMU for /livox/imu_front.

        PR-G2 (G2-A): Enable condition derived from sensor_names — matches
        the MuJoCo path's setup_sensor_interface() substring check, so the
        yaml stays a single source of truth (no separate `imu.enabled` flag).

        PR-G2 (G2-B): Mounted on lidar_front link (not chassis_base), so the
        MJCF body pose (yaw=-45° for s1_v2) determines the IMU sensor frame.
        Genesis IMU sensor returns (lin_acc, ang_vel, mag) tuple — gravity
        already subtracted by the sensor kernel (see `_update_raw_data` in
        `genesis/engine/sensors/imu.py`), so the values map directly to
        `sensor_msgs/Imu.linear_acceleration` and `.angular_velocity`.
        """
        has_lidar_imu = any(s in self.sensor_names for s in self._lidar_imu_sensor_names)
        if not has_lidar_imu:
            # Expected in this release: the LiDAR-integrated IMU is unsupported, so
            # sensor_names never lists it. DEBUG rather than WARN -- nothing is wrong.
            astribot_simu_log(
                "LiDAR-integrated IMU not declared in sensor_names; not created",
                level="DEBUG",
            )
            self.imu_sensor = None
            return
        try:
            self.imu_sensor = self.scene.add_sensor(
                gs.options.sensors.IMU(
                    entity_idx=self._robot_entity_idx(),
                    link_idx_local=self._lidar_front_link_idx(),
                )
            )
            astribot_simu_log(
                f"IMU on lidar_front link "
                f"(triggered by sensor_names: {sorted(set(self.sensor_names) & set(self._lidar_imu_sensor_names))})"  # noqa: E501
            )
        except Exception as e:
            astribot_simu_log(f"IMU setup failed: {e}", level="WARN")
            self.imu_sensor = None

    def setup_genesis_chassis_imu(self):
        """Chassis IMU (9-D Float64MultiArray) for the Genesis backend.

        Mirrors the lidar IMU setup but mounts on chassis_base, so the chassis
        IMU pipeline works on Genesis instead of warn-once skipping (the old
        behavior which also crashed the scheduler path — base_env's
        _build_chassis_imu_msg calls get_sensor_data, MuJoCo-only). Enabled
        only when 'astribot_chassis_base_imu_gyro' is in sensor_names and not
        CLI-disabled, matching setup_sensor_scheduler's registration gate.

        The Genesis IMU sensor gives lin_acc (gravity subtracted) + ang_vel in
        the mount-link local frame; link.get_quat() gives world orientation for
        the RPY part. See _sample_publish_chassis_imu for the 9-D assembly.
        """
        _disabled = set(self.param.get("_disabled_sensors", []))
        if (
            "astribot_chassis_base_imu_gyro" not in self.sensor_names
            or "astribot_chassis_base_imu_gyro" in _disabled
        ):
            self.chassis_imu_sensor = None
            return
        try:
            self.chassis_imu_sensor = self.scene.add_sensor(
                gs.options.sensors.IMU(
                    entity_idx=self._robot_entity_idx(),
                    link_idx_local=self._chassis_link_idx_local(),
                )
            )
            astribot_simu_log("Genesis: chassis IMU on chassis_base link")
        except Exception as e:
            astribot_simu_log(f"Genesis: chassis IMU setup failed: {e}", level="WARN")
            self.chassis_imu_sensor = None

    def _sample_publish_chassis_imu(self):
        """Genesis override of base_env's chassis IMU sampler (base_env reads
        MuJoCo-only get_sensor_data, which doesn't exist here). Publishes the
        same 9-D Float64MultiArray layout base_env._build_chassis_imu_msg does:
        [roll, pitch, yaw, wx, wy, wz, ax, ay, az].
        """
        if getattr(self, "chassis_imu_sensor", None) is None:
            return
        ros = self.multi_robot_ros_interface
        pub = getattr(ros, "sensor_ros_pub", {}).get("chassis_imu_publisher")
        if pub is None:
            return
        try:
            data = self.chassis_imu_sensor.read()
            lin_acc = data.lin_acc.cpu().numpy().flatten()
            ang_vel = data.ang_vel.cpu().numpy().flatten()

            # World orientation of the chassis_base link → ZYX-intrinsic RPY
            # (ROS convention), matching base_env._build_chassis_imu_msg.
            from std_msgs.msg import Float64MultiArray

            from simu_utils.transform_utils import quaternion_to_rotation_matrix

            quat = self._to_numpy(self.robot.get_link("chassis_base").get_quat())
            R = quaternion_to_rotation_matrix(quat)  # (w, x, y, z)
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
            yaw = np.arctan2(R[1, 0], R[0, 0])

            msg = Float64MultiArray()
            msg.data = [
                float(roll),
                float(pitch),
                float(yaw),
                float(ang_vel[0]),
                float(ang_vel[1]),
                float(ang_vel[2]),
                float(lin_acc[0]),
                float(lin_acc[1]),
                float(lin_acc[2]),
            ]
            pub.publish(msg)
        except Exception as e:
            astribot_simu_log(f"Genesis chassis IMU publish failed: {e}", level="WARN")

    def _robot_entity_idx(self):
        # The robot entity is the first one we add (Plane is second). Lookup
        # by reference so this survives reordering.
        return self.scene.entities.index(self.robot)

    def _chassis_link_idx_local(self):
        # chassis_base is the MJCF root body — first link of the entity.
        link = self.robot.get_link("chassis_base")
        return getattr(link, "idx", 0)

    def _lidar_front_link_idx(self):
        """PR-G2 (G2-B): Find the lidar_front body's local index in the robot
        entity. The MJCF declares lidar_front as a child of chassis_base with
        the calibrated mount pose (e.g. s1_v2: pos="0.17393 0.16893 0.073397"
        quat="0.9238796 0 0 -0.3826834" = yaw -45°). Mounting Genesis sensors
        on this link (instead of chassis_base + manual pos_offset/euler_offset)
        means the sensor world pose is whatever MJCF says — no risk of XYZ vs
        ZYX euler convention bugs, and auto-synced when the MJCF model changes.

        Falls back to chassis_base if the model doesn't declare lidar_front
        (e.g. a future variant without front lidar, or an older MJCF layout).
        Genesis 1.0/1.2 link objects have .idx_local (preferred) or .idx.
        """
        try:
            link = self.robot.get_link("lidar_front")
            idx = getattr(link, "idx_local", None)
            if idx is None:
                idx = getattr(link, "idx", 0)
            return int(idx)
        except Exception:
            astribot_simu_log(
                "Genesis: lidar_front body not found in MJCF; "
                "falling back to chassis_base (sensor pose may be off by "
                "~17cm + yaw -45°)",
                level="WARN",
            )
            return self._chassis_link_idx_local()

    def _publish_imu_genesis(self):
        """PR-5: read Genesis IMU sensor, publish sensor_msgs/Imu to
        /livox/imu_front (the same topic the MuJoCo PR-2 path uses).

        Replaces the MuJoCo-only `_build_full_imu_msg` path (which needs
        `lidar_imu_gyro`/`lidar_imu_acc`/`lidar_site` MuJoCo sensor objects).
        Genesis IMUData already subtracts gravity and returns values in the
        link local frame, so we just plug into sensor_msgs/Imu directly.
        """
        if self.imu_sensor is None:
            return
        ros = self.multi_robot_ros_interface
        pub = getattr(ros, "sensor_ros_pub", {}).get("lidar_imu_publisher")
        if pub is None:
            return
        try:
            data = self.imu_sensor.read()
            lin_acc = data.lin_acc.cpu().numpy().flatten()
            ang_vel = data.ang_vel.cpu().numpy().flatten()
            from sensor_msgs.msg import Imu

            msg = Imu()
            msg.header.stamp = ros.get_timestamp()
            msg.header.frame_id = "lidar_site"
            msg.linear_acceleration.x = float(lin_acc[0])
            msg.linear_acceleration.y = float(lin_acc[1])
            msg.linear_acceleration.z = float(lin_acc[2])
            msg.angular_velocity.x = float(ang_vel[0])
            msg.angular_velocity.y = float(ang_vel[1])
            msg.angular_velocity.z = float(ang_vel[2])
            # Orientation: not in IMUData (no fusion); leave unset (zeros).
            pub.publish(msg)
        except Exception as e:
            astribot_simu_log(f"IMU publish failed: {e}", level="WARN")

    def _sample_publish_lidar_imu(self):
        """PR-2: LiDAR-IMU sample+publish for the sim-time scheduler (Genesis).
        Overrides base_env's MuJoCo-specific wrapper (which calls
        _build_full_imu_msg). Genesis uses _publish_imu_genesis instead."""
        self._publish_imu_genesis()

    def _sample_publish_lidar(self):
        """PR-2: LiDAR sample+publish for the sim-time scheduler (Genesis).
        Reads Raycaster then publishes CustomMsg. Replaces the old `% 5`
        throttle in update_sensor_states() — scheduler owns the 10Hz cadence."""
        self._publish_lidar_genesis()

    def _publish_lidar_genesis(self):
        """PR-5: read Genesis Raycaster hits, publish livox CustomMsg to
        /livox/lidar_front. Reuses the same build_livox_custom_msg helper
        the MuJoCo PR-2 path uses (PR-3 msg namespace switch applies).

        P1 fidelity gap: SphericalPattern gives uniform grid, so we assign
        line_id=0 to every point (Livox MID-360 has 4 lines 0..3, PR-6
        fixes this with a custom RaycastPattern subclass).
        """
        if self.lidar_sensor is None:
            return
        ros = self.multi_robot_ros_interface
        pub = getattr(ros, "lidar_custommsg_publisher", None)
        if pub is None:
            return
        try:
            from simu_utils.lidar_msg_builder import build_livox_custom_msg

            # advance the non-repeating sweep one frame before reading, so
            # successive frames use distinct angles (real MID-360 behavior). Grid
            # path leaves lidar_pattern None and skips this.
            self._advance_lidar_sweep()
            data = self.lidar_sensor.read()
            pts = data.points.cpu().numpy()
            # Genesis Raycaster returns points shaped like the
            # pattern return_shape + (3,): the grid SphericalPattern gives
            # (ring, azim, 3); LivoxMid360Pattern gives (n_rays, 3). Flatten any
            # leading dims to (N, 3); downstream filters drop near-zero/NaN hits.
            if pts.ndim > 2:
                pts = pts.reshape(-1, pts.shape[-1])
            if pts.size == 0 or pts.shape[0] == 0:
                return
            # real per-point line ids from the mid360 table (grid path has
            # no line structure, so all-zero as before).
            if self.lidar_pattern is not None:
                line_ids = self.lidar_pattern.line_ids
                if line_ids.shape[0] != pts.shape[0]:  # defensive: keep aligned
                    line_ids = np.zeros(pts.shape[0], dtype=np.uint8)
            else:
                line_ids = np.zeros(pts.shape[0], dtype=np.uint8)
            # timebase from the same sim-time source as header.stamp (not
            # wall clock) so the two are consistent within a bag. offset_time is
            # zeroed — a frame-synchronous sim samples every point at one instant.
            stamp = ros.get_timestamp()
            timebase_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            msg = build_livox_custom_msg(
                points_xyz=pts.astype(np.float32),
                line_ids=line_ids,
                timebase_ns=timebase_ns,
                frame_id='lidar_site',
                lidar_id=1,
                reflectivity=100,
                offset_time_mode='zero' if self.lidar_pattern is not None else 'linear',
            )
            msg.header.stamp = stamp
            pub.publish(msg)
        except Exception as e:
            astribot_simu_log(f"LiDAR publish failed: {e}", level="WARN")

    def _advance_lidar_sweep(self):
        """PR-6: step the MID-360 non-repeating sweep and push the new ray
        directions into the sensor's shared ray-dir buffer.

        Genesis copies pattern.ray_dirs into RaycasterSharedMetadata.ray_dirs
        once at build time; the cast kernel reads that buffer, not the pattern.
        So each frame we recompute the pattern's next angle window and write it
        into this sensor's slice of the shared buffer. mount euler_offset is
        (0,0,0) → identity rotation, so no re-transform is needed. Fully
        guarded: any failure degrades to a static (still valid) pattern.
        """
        pattern = getattr(self, 'lidar_pattern', None)
        if pattern is None:
            return
        try:
            pattern.advance()
            sensor = self.lidar_sensor
            meta = sensor._shared_metadata
            start = meta.sensors_ray_start_idx[sensor._idx]
            n = pattern.return_shape[0]
            meta.ray_dirs[start:start + n] = pattern.ray_dirs.reshape(-1, 3)
        except Exception as e:
            astribot_simu_log(f"LiDAR sweep advance skipped (static frame): {e}", level="WARN")

    def update_sensor_states(self):
        """PR-5/PR-G2: Genesis sensor publish path with sensor_names dispatch.

        Replaces the MuJoCo-specific base-class implementation (which reads
        `lidar_imu_gyro`/`lidar_imu_acc`/`lidar_site` MuJoCo sensor objects —
        don't exist on Genesis backend). The base class still creates the
        publishers via `setup_sensor_interface(['lidar_site'])` so we publish
        into the same ROS topics as the MuJoCo backend.

        PR-G2 (G2-C): sensor_names is the single source of truth (yaml).
        Genesis-supported sensors (lidar_imu_* + lidar_site) trigger their
        publish path; unsupported ones (chassis_imu / lidar_back / camera /
        force) warn-once and skip, so the yaml can stay backend-agnostic.

        PR-G6 (P0-2): IMU is now published in step()'s frame_skip loop (after
        each scene.step()) to achieve 200 Hz. This method only handles LiDAR
        at 10 Hz (50 Hz env loop / 5 = 10 Hz). The throttle counter is still
        tracked here for LiDAR; IMU counter is removed since IMU publishes 4×
        per env.step inside the frame_skip loop.

        PR-2: when the sim-time scheduler is active, this method is skipped
        (sensors sampled via scheduler.tick() in the physics loop). Back-compat
        with stage 1 (scheduler off by default). Stage 3: delete body below guard.
        """
        if self._use_sensor_scheduler:
            return
        if not hasattr(self, "_lidar_pub_counter"):
            self._lidar_pub_counter = 0
            self._sensor_warned = set()
        self._lidar_pub_counter += 1

        # 1. Supported: lidar_imu_* + lidar_site (share one publisher).
        #    The base class setup_sensor_interface creates lidar_imu_publisher
        #    and lidar_custommsg_publisher when it sees 'lidar_site' in
        #    sensor_names. We reuse those publishers here.
        has_lidar_imu = any(s in self.sensor_names for s in self._lidar_imu_sensor_names)
        if has_lidar_imu:
            # IMU now published in step() frame_skip loop (4× per env.step = 200 Hz).
            # Only LiDAR is published here.
            # LiDAR stays throttled — Raycaster.read() is the GPU bottleneck.
            if self._lidar_pub_counter % 5 == 0:
                self._publish_lidar_genesis()

        # Chassis IMU is now implemented on Genesis (setup_genesis_chassis_imu /
        # _sample_publish_chassis_imu). Legacy path publishes at the 50 Hz
        # env-loop rate; the scheduler path (default) drives it at the
        # configured sensor_frequencies instead.
        if "astribot_chassis_base_imu_gyro" in self.sensor_names:
            self._sample_publish_chassis_imu()

        # 2. Unsupported: warn-once and skip. This lets yaml stay a single
        #    source of truth — sim.yaml and sim_genesis.yaml share the same
        #    sensor_names list, and each backend picks what it can support.
        for name in self.sensor_names:
            if name in self._sensor_warned:
                continue
            if name in self._lidar_imu_sensor_names:
                continue  # already handled above
            if name in ("astribot_chassis_base_imu_gyro", "astribot_chassis_base_imu_acc"):
                continue  # handled above (implemented, not a warn-once skip)
            # New sensor — warn once and remember we've warned
            self._sensor_warned.add(name)
            if name.startswith("lidar_back"):
                astribot_simu_log(
                    f"Genesis: rear LiDAR not implemented; sensor '{name}' skipped",
                    level="WARN",
                )
            elif "force" in name:
                astribot_simu_log(
                    f"Genesis 1.x: force/torque sensor not supported; '{name}' skipped",
                    level="WARN",
                )
            elif any(
                cam in name for cam in ("head_rgbd", "wrist_rgbd", "head_stereo", "torso_rgbd")
            ):
                # Camera names from sim_genesis.yaml camera_names list — they
                # shouldn't appear in sensor_names, but if they do, skip silently
                # (setup_genesis_camera already handles camera setup separately).
                pass
            else:
                astribot_simu_log(
                    f"Genesis: unrecognized sensor '{name}' skipped",
                    level="WARN",
                )

    def _capture_video_frame(self):
        """PR-G2 video: render off-screen overview cam → ffmpeg pipe at 25fps.
        Called once per step(). Throttles to self.video_fps by skipping frames
        (sim runs at 50Hz; video_fps=25 → every 2nd step emits a frame).
        """
        if self._video_ffmpeg is None:
            return
        frame_interval = int(50 / self.video_fps)  # 50Hz sim / 25fps = 2
        self._video_step_counter += 1
        if self._video_step_counter % frame_interval != 0:
            return
        try:
            rgb = self._video_camera.render(rgb=True, depth=False)
            # Genesis 1.2.0 camera.render returns a tuple
            # (rgb, depth, ...) even when only rgb=True; 1.0 returned a bare
            # ndarray. Defensively unwrap whichever variant we get.
            if isinstance(rgb, tuple):
                rgb = rgb[0]
            import numpy as np

            if rgb is None:
                return
            # Genesis 1.2 returns uint8 (0..255) directly; 1.0 returned float32
            # in [0,1]. Both convertible to ffmpeg rgb24 stream — handle either.
            if rgb.dtype == np.uint8:
                rgb_u8 = rgb
            else:
                rgb_u8 = (rgb * 255).clip(0, 255).astype(np.uint8)
            self._video_ffmpeg.stdin.write(rgb_u8.tobytes())
        except Exception as e:
            astribot_simu_log(
                f"PR-G2 video frame capture failed: {e}",
                level="WARN",
            )

    def compute_dynamics_compensation(
        self,
        joint_indices=None,
        include_gravity=True,
        include_coriolis=True,
    ):
        """Compute dynamics feedforward compensation using Genesis qf_bias.

        Genesis func_bias_force computes:
            qf_bias = cdof · cfrc
        where cfrc includes inertia acceleration, Coriolis, and gravity.

        Equivalent to MuJoCo: qfrc_bias = C(q,qd)*qd + g(q)

        Note: Genesis does not support separating gravity from Coriolis.
              Only include_gravity=True, include_coriolis=True is supported.
        """
        if not (include_gravity and include_coriolis):
            raise NotImplementedError(
                "Genesis does not support separating gravity from Coriolis. "
                "Use include_gravity=True, include_coriolis=True."
            )

        n_dofs = self.robot.n_dofs
        compensation = np.zeros(n_dofs)

        try:
            # qf_bias is a quadrants Tensor of shape (n_dofs, n_envs); .to_numpy()
            # is the supported accessor. Indexing it as qf_bias[:, 0] and calling
            # .cpu() (the old code) raises TypeError on genesis 1.2.x.
            qf_bias = self.robot._solver.dyn_state.dofs.qf_bias.to_numpy()
            compensation = np.asarray(qf_bias).reshape(-1, 1)[:, 0][:n_dofs].copy()
        except Exception as e:
            astribot_simu_log(
                f"Genesis dynamics compensation failed (API change?): {e}", level="WARN"
            )
            compensation = np.zeros(n_dofs)

        return compensation[joint_indices] if joint_indices else compensation

    def _get_component_name_from_joint(self, joint_name):
        """Extract component name from joint name (same logic as MuJoCo)."""
        for component in self.robot_list:
            if joint_name.startswith(component):
                return component
        return "unknown"

    def _position_gains_for(self, joint_name):
        """mode=1 (kp, kd), priority per_joint > per_component > default_gains.

        Same convention as MuJoCo's _apply_controller_config_to_actuators:
        kp -> act_gain, kd -> |act_bias[2]| (the real velocity damping).
        Returns None when the config does not cover this joint, in which case the
        MJCF values are kept.
        """
        cfg = self.controller_config
        if not isinstance(cfg, dict):
            return None

        per_joint = cfg.get("per_joint", {}) or {}
        if joint_name in per_joint and "position" in per_joint[joint_name]:
            c = per_joint[joint_name]["position"]
            return float(c["kp"]), float(c["kd"])

        component = self._get_component_name_from_joint(joint_name)
        comp_cfg = (cfg.get("per_component", {}) or {}).get(component, {}) or {}
        gains = comp_cfg.get("default_gains", {}) or {}
        if "position" in gains:
            c = gains["position"]
            return float(c["kp"]), float(c["kd"])

        default_gains = cfg.get("default_gains", {}) or {}
        if "position" in default_gains:
            c = default_gains["position"]
            return float(c["kp"]), float(c["kd"])
        return None

    def _apply_controller_config_to_dofs(self):
        """Push controller_config position gains into Genesis dof PD parameters.

        Genesis was never wired up: the only set_dofs_kp/kv calls in the repo were
        the roller zeroing and mode=2's chassis_vel_kv, so mode-1 gains always came
        from the MJCF and tuning the yaml had no effect on this backend.

        For body joints the yaml values were copied from the MJCF to begin with, so
        wiring them does not change their behaviour. What this actually fixes is the
        **gripper**. Genesis's POSITION force law carries a residual term
        `(act_gain + act_bias[1]) * pos` that only vanishes when
        act_gain == -act_bias[1]. The gripper is tendon-driven and its MJCF
        gainprm=4.65 does not pair with biasprm[1]=-500, leaving a residual of
        -495.4 -- a stiff spring pinning the joint near zero. Measured: a mode-1
        target of 0.5 rad only reached 0.0048 rad, and 0.2973 rad after this fix
        (converging to 0.5011 by step 1200; kp=4.65 is simply slow).
        set_dofs_kp sets act_gain=kp and act_bias[1]=-kp together
        (abd/accessor.py:610-627), cancelling the residual.

        Only position gains are written here. mode=2/3 set their own parameters
        inside step(). This mirrors a bug just fixed on the MuJoCo side: writing all
        three modes' gains into the same parameter slots lets the last one win, and
        adding torque:{kp:0.0} zeroed act_gain everywhere, stripping mode=1 of any
        position authority.
        """
        applied = 0
        try:
            import numpy as _np

            chassis = set(getattr(self, "chassis_joint_names", []) or [])
            chassis |= getattr(self, "_mecanum_chassis_joints", set()) or set()

            for joint_name, dof in self.joint_name_to_index.items():
                # Virtual chassis joints have no PD dof; the mecanum adapter
                # expands them into wheel speeds instead.
                if joint_name in chassis:
                    continue
                gains = self._position_gains_for(joint_name)
                if gains is None:
                    continue
                kp, kd = gains
                self.robot.set_dofs_kp(_np.array([kp], dtype=_np.float32), [dof])
                self.robot.set_dofs_kv(_np.array([kd], dtype=_np.float32), [dof])
                applied += 1

            astribot_simu_log(
                f"controller_config: applied position gains to {applied} Genesis dof(s)",
                level="DEBUG",
            )
        except Exception as e:
            # A wiring failure should fall back to the MJCF gains rather than
            # preventing the simulation from starting.
            astribot_simu_log(
                f"Failed to apply controller_config gains to Genesis dofs "
                f"(after {applied}); falling back to MJCF values: {e}",
                level="WARN",
            )

    def _torque_mode_kd(self, joint_name):
        """mode=3 (Zero-G) 的阻尼系数，语义与 MuJoCo 侧一致。

        优先级 per_joint > per_component > default_gains，与
        ControllerConfig.get_gains 相同。真机 kp=0，故只有 kd 可配。
        """
        cfg = self.controller_config
        if not isinstance(cfg, dict):
            return 0.0

        per_joint = cfg.get("per_joint", {}) or {}
        if joint_name in per_joint and "torque" in per_joint[joint_name]:
            return float(per_joint[joint_name]["torque"].get("kd", 0.0))

        component = self._get_component_name_from_joint(joint_name)
        comp_cfg = (cfg.get("per_component", {}) or {}).get(component, {}) or {}
        gains = comp_cfg.get("default_gains", {}) or {}
        if "torque" in gains:
            return float(gains["torque"].get("kd", 0.0))

        default_gains = cfg.get("default_gains", {}) or {}
        if "torque" in default_gains:
            return float(default_gains["torque"].get("kd", 0.0))
        return 0.0

    def _torque_mode_feedforward(self, dof_indices):
        """mode=3 前馈：补偿(C+g) 与小阻尼 -kd*qd 之和，按 dof 返回。

        control_dofs_force 是纯力矩直通（不像 MuJoCo 的 affine actuator 会自带
        位置/速度项），所以这两项要显式加进去，才能对齐真机链路
        force = tau_cmd + 补偿 - kd*qd（kp=0，期望速度为零）。
        """
        comp_cfg = self.controller_config.get("dynamics_compensation", {})
        comp_on = comp_cfg.get("enabled", True) and 3 in comp_cfg.get("modes", [1, 2, 3])

        try:
            qf_bias = self.compute_dynamics_compensation() if comp_on else None
            vel = np.asarray(
                self.robot.get_dofs_velocity(dof_indices).cpu().numpy()
            ).reshape(-1)
        except Exception as e:
            if not getattr(self, "_torque_ff_warned", False):
                self._torque_ff_warned = True
                astribot_simu_log(
                    f"Genesis torque-mode feed-forward unavailable: {e}", level="WARN"
                )
            return {}

        # dof -> 关节名，用于查 per-joint kd
        dof_to_name = {v: k for k, v in self.joint_name_to_index.items()}

        out = {}
        for i, dof in enumerate(dof_indices):
            ff = 0.0
            if qf_bias is not None and dof < len(qf_bias):
                ff += float(qf_bias[dof])
            kd = self._torque_mode_kd(dof_to_name.get(dof, ""))
            if kd and i < len(vel):
                ff -= kd * float(vel[i])
            out[dof] = ff
        return out

    def joint_space_compensation_offsets(self, dof_indices):
        """Feed-forward compensation expressed as POSITION-target offsets.

        Genesis's POSITION ctrl mode overwrites dofs.qf_applied with the PD result
        every substep (abd/forward_dynamics.py), so a feed-forward torque cannot be
        added there -- it would be discarded. Instead fold it into the target:

            force = kp * ((pos_cmd + qf_bias/kp) - pos) + ...
                  = kp * (pos_cmd - pos) + qf_bias

        i.e. offsetting the target by qf_bias/kp injects exactly the desired
        feed-forward torque through the existing PD path.

        Compared with the link-COM external-force path this covers the full
        C(q,qd)*qd + g(q) (not just -m*g) and needs no link-index bookkeeping.
        Measured on a low-kp hold test of astribot_arm_right_joint_2 over 120
        steps: no compensation 23.86 deg drift, link-COM forces 12.79 deg,
        this path 0.00 deg.

        Returns {dof_index: offset_rad}; empty when disabled or unavailable.
        """
        comp_cfg = self.controller_config.get("dynamics_compensation", {})
        if not comp_cfg.get("enabled", True):
            return {}
        if 1 not in comp_cfg.get("modes", [1, 2, 3]):
            return {}

        try:
            qf_bias = self.compute_dynamics_compensation()
            act_gain = np.asarray(
                self.robot._solver.dyn_info.dofs.act_gain.to_numpy()
            ).reshape(-1)
        except Exception as e:
            if not getattr(self, "_dyn_comp_warned", False):
                self._dyn_comp_warned = True
                astribot_simu_log(
                    f"Genesis joint-space compensation unavailable: {e}", level="WARN"
                )
            return {}

        offsets = {}
        for dof in dof_indices:
            if dof >= len(qf_bias) or dof >= len(act_gain):
                continue
            kp = float(act_gain[dof])
            # kp==0 means no position authority -- an offset would do nothing.
            if abs(kp) < 1e-6:
                continue
            offsets[dof] = float(qf_bias[dof]) / kp
        return offsets

    def _apply_dynamics_compensation_step(self):
        """Apply dynamics compensation during step() based on controller_config.

        Genesis applies external forces at link COM (not joint torques like MuJoCo).
        This method filters links based on controller_config settings.

        NOTE: for mode-1 joints the joint-space path
        (joint_space_compensation_offsets) is used instead; this link-COM path
        remains for dofs that path cannot serve (kp==0, or non-position modes).
        """
        comp_cfg = self.controller_config.get("dynamics_compensation", {})
        if not comp_cfg.get("enabled", True):
            return

        comp_modes = comp_cfg.get("modes", [1, 2, 3])

        try:
            import numpy as _np

            # Get gravity vector
            g = _np.asarray(self.scene.sim.rigid_solver.get_gravity().cpu()).reshape(-1)[:3]

            # Collect links to compensate
            links_to_comp = []  # [(link_idx_local, mass), ...]
            for link in self.robot.links:
                name = (link.name or "").lower()
                if link.parent_idx == -1:
                    continue

                # Skip chassis base and wheels
                if "wheel" in name or name in ("world", "chassis_base"):
                    continue

                # Check if this link's component should be compensated
                component_name = self._get_component_name_from_joint(name)

                # Skip grippers based on component name
                if "gripper" in component_name or "gripper" in name:
                    continue

                # Check if current control mode enables compensation
                # Note: Genesis doesn't have per-joint modes, so we use a heuristic
                # Assume all body joints are in the configured modes
                mode = 1  # Default to position control
                if mode not in comp_modes:
                    continue

                m = float(
                    _np.asarray(self.robot.get_links_inertial_mass([link.idx_local]).cpu()).reshape(
                        -1
                    )[0]
                )
                if m <= 0:
                    continue
                links_to_comp.append((int(link.idx_local), m))

            # Apply anti-gravity forces.
            #
            # INDEX SPACE (was a bug): apply_links_external_force here is the
            # *solver*-level API, which expects scene-global link indices. The
            # link objects carry idx_local (entity-relative). Because the scene
            # adds a visual Plane entity before the robot, the robot's link_start
            # is 1, so passing idx_local shifted every force by one link and the
            # last one landed outside the robot entirely. Entity-level helpers
            # like get_links_inertial_mass do the +link_start conversion
            # internally, which is why the masses were right and the forces were
            # not. Convert explicitly here.
            if links_to_comp:
                link_start = int(self.robot.link_start)
                idxs = [i + link_start for i, _ in links_to_comp]
                forces = _np.array(
                    [[-m * g[0], -m * g[1], -m * g[2]] for _, m in links_to_comp],
                    dtype=_np.float32,
                )
                self.scene.sim.rigid_solver.apply_links_external_force(
                    forces, links_idx=idxs, ref="link_com"
                )

        except Exception as e:
            if not getattr(self, "_dyn_comp_warned", False):
                self._dyn_comp_warned = True
                astribot_simu_log(
                    f"Genesis dynamics compensation apply failed (API change?): {e}",
                    level="WARN",
                )

    def _apply_mecanum(self, mec_pose):
        order = self.joint_names_list[0]  # [chassis_x, chassis_y, chassis_zrot]
        target_pose = [float(mec_pose[n]) for n in order if n in mec_pose]
        if len(target_pose) != 3:
            return
        dt = self.frame_skip * self.physics_dt
        self.mecanum_adapter.apply_pose_command(target_pose, dt=dt)

    def _setup_mecanum_adapter(self):
        self.mecanum_adapter = None
        self._mecanum_chassis_joints = set()
        param = self.param
        chassis_model = (
            param.get("chassis_model", "kinematic") if hasattr(param, "get") else "kinematic"
        )
        if chassis_model != "dynamic":
            return
        chassis_type = param.get("chassis_type", "") if hasattr(param, "get") else ""
        if chassis_type != "omni":
            return
        from simu_utils.chassis_kinematics import GenesisOmniChassis

        mec = (param.get("omni") or {}) if hasattr(param, "get") else {}
        self.mecanum_adapter = GenesisOmniChassis(
            self.robot,
            wheel_radius=mec.get("wheel_radius", 0.078),
            half_wheelbase=mec.get("half_wheelbase", 0.216),
            half_track=mec.get("half_track", 0.214),
            base_link_name=mec.get("base_link_name"),
            wheel_joint_names=mec.get("wheel_joint_names"),
            vel_scale=mec.get("vel_scale", 10.0),
            max_wheel_speed=mec.get("max_wheel_speed"),
            kd=mec.get("kd", 0.0),
            pos_deadband=mec.get("pos_deadband", 0.0),
            yaw_deadband=mec.get("yaw_deadband", 0.0),
            vel_ff_scale=mec.get("vel_ff_scale", 1.0),
        )
        if self.joint_names_list and self.joint_names_list[0]:
            self._mecanum_chassis_joints = set(self.joint_names_list[0])
        astribot_simu_log(
            "chassis_type=omni: GenesisOmniChassis enabled; the chassis 3-DOF pose "
            "command is expanded into 4 wheel-speed control_dofs_velocity targets, "
            "with no change to the SDK interface."
        )

    def _free_omni_wheel_rollers(self):
        robot = getattr(self, "robot", None)
        if robot is None:
            return
        roller_dofs = []
        for joint in robot.joints:
            name = getattr(joint, "name", "") or ""
            if "roller" in name.lower():
                di = joint.dof_idx_local
                if di is None:
                    continue
                if isinstance(di, (list, tuple)):
                    roller_dofs.extend(int(x) for x in di)
                else:
                    roller_dofs.append(int(di))
        if not roller_dofs:
            return
        import numpy as np

        zeros = np.zeros(len(roller_dofs), dtype=np.float32)
        try:
            robot.set_dofs_kp(zeros, roller_dofs)
            robot.set_dofs_kv(zeros, roller_dofs)
            astribot_simu_log(
                f"Omni-wheel rollers freed: zeroed kp/kv on {len(roller_dofs)} roller "
                "dofs (restores passive free rolling, matching MuJoCo; fixes the "
                "Genesis chassis tipping over while moving)."
            )
        except Exception as e:
            astribot_simu_log(
                f"Failed to free the rollers (set_dofs_kp/kv): {e}; "
                "the chassis may tip over while moving.",
                level="WARN",
            )

    def _apply_initial_joint_positions(self):
        init = self.param.get("initial_joint_positions") if hasattr(self.param, "get") else None
        if not init:
            return
        dof_indices = []
        values = []
        for joint_name, value in init.items():
            idx = self.joint_name_to_index.get(joint_name)
            if idx is None:
                astribot_simu_log(
                    f"initial_joint_positions: unknown joint '{joint_name}', skipped",
                    level="WARN",
                )
                continue
            dof_indices.append(idx)
            values.append(float(value))
        if dof_indices:
            self.robot.set_dofs_position(values, dof_indices, zero_velocity=True)

    def setup_joint_index_mapping(self):
        astribot_simu_log("Setup joint index mapping")

        mec_chassis = getattr(self, "_mecanum_chassis_joints", set())

        self.dof_index = [
            self.robot.get_joint(name).dof_idx_local
            for name in self.joint_names
            if name not in mec_chassis
        ]

        for name in self.joint_names:
            if name in mec_chassis:
                continue
            index = self.robot.get_joint(name).dof_idx_local
            self.joint_name_to_index[name] = index

        # Chassis dof indices for the chassis_fixed freeze logic. The
        # MJCF declares chassis joints as `astribot_chassis_x/y/zrot`
        # (slide/slide/hinge). We need their dof_idx_local to call
        # set_dofs_position([0,0,0], chassis_dof_indices) post-step.
        # Skip silently if the model doesn't declare them (e.g. a
        # gripper-only variant); only warn when chassis_fixed=True.
        self.chassis_joint_names = [
            n
            for n in ("astribot_chassis_x", "astribot_chassis_y", "astribot_chassis_zrot")
            if n in self.joint_name_to_index
        ]
        self.chassis_dof_indices = [self.joint_name_to_index[n] for n in self.chassis_joint_names]
        if self.chassis_fixed:
            if self.chassis_dof_indices:
                astribot_simu_log(
                    f"chassis_fixed=true: step() will pin chassis dofs "
                    f"{self.chassis_dof_indices} ({self.chassis_joint_names}) to 0 every step"
                )
            else:
                astribot_simu_log(
                    "chassis_fixed=true but no chassis joints found in model; freeze disabled",
                    level="WARN",
                )

    def get_ft_sensor_data(self, robot_name):
        pass

    def get_reset_status(self):
        return False

    def get_camera_image(self, camera_name="astribot_head"):
        data = {}
        render_data = None
        rgb_img = np.zeros((640, 480, 3), dtype=np.uint8)

        if camera_name == "astribot_head":
            render_data = self.head_camera.render(rgb=True)
        elif camera_name == "astribot_arm_right_effector":
            render_data = self.right_camera.render(rgb=True)
        elif camera_name == "astribot_arm_left_effector":
            render_data = self.left_camera.render(rgb=True)
        elif camera_name == "astribot_global_camera":
            render_data = self.global_camera.render(rgb=True)

        rgb_img = cv2.cvtColor(render_data, cv2.COLOR_RGB2BGR)

        # rgb_img = cv2.flip(rgb_img, 0)
        data["rgb_img"] = rgb_img
        data["depth_img"] = None
        data["point_cloud"] = None

        return data

    def _resolve_human_view(self):
        """- camera_pos = pos。"""
        import numpy as np
        import xml.etree.ElementTree as ET

        fallback = ((2.0, 0.0, 2.0), (0.0, 0.0, 1.0), 100.0)
        try:
            root = ET.parse(self.model_path).getroot()
            cam = None
            for c in root.iter("camera"):
                if c.get("name") == "human":
                    cam = c
                    break
            if cam is None:
                return fallback
            pos = np.array([float(v) for v in cam.get("pos").split()])
            xy = cam.get("xyaxes")
            fovy = float(cam.get("fovy", 45.0))
            if xy is None:
                return fallback
            vals = [float(v) for v in xy.split()]
            x_cam = np.array(vals[:3])
            y_cam = np.array(vals[3:6])
            z_cam = np.cross(x_cam, y_cam)
            view_dir = -z_cam
            view_dir = view_dir / (np.linalg.norm(view_dir) or 1.0)
            d = float(np.linalg.norm(pos - np.array([0.0, 0.0, 1.0]))) or 2.236
            lookat = pos + view_dir * d
            return (tuple(pos.tolist()), tuple(lookat.tolist()), fovy)
        except Exception as e:
            astribot_simu_log(
                f"_resolve_human_view failed, falling back to the default view: {e}",
                level="WARN",
            )
            return fallback

    def update_camera_pose(self):
        for camera_name in self.camera_names:
            if camera_name not in self.camera_extrinsics:
                continue

            pose_update = self._compute_camera_world_pose(camera_name)
            if pose_update is None:
                continue

            cam_world_pos, cam_lookat = pose_update

            normalized_name = camera_name
            if camera_name == "head_rgbd":
                normalized_name = "astribot_head"
            elif camera_name in ("left_wrist_rgbd", "astribot_arm_left_effector"):
                normalized_name = "astribot_arm_left_effector"
            elif camera_name in ("right_wrist_rgbd", "astribot_arm_right_effector"):
                normalized_name = "astribot_arm_right_effector"

            if normalized_name == "astribot_head" and hasattr(self, "head_camera"):
                self.head_camera.set_pose(pos=cam_world_pos, lookat=cam_lookat)
            elif normalized_name == "astribot_arm_left_effector" and hasattr(self, "left_camera"):
                self.left_camera.set_pose(pos=cam_world_pos, lookat=cam_lookat)
            elif normalized_name == "astribot_arm_right_effector" and hasattr(self, "right_camera"):
                self.right_camera.set_pose(pos=cam_world_pos, lookat=cam_lookat)

    def _to_numpy(self, t):
        # Genesis 1.0 returns torch tensors (cuda:0 by default with
        # gs.init(backend=gs.gpu)) for link / joint pose queries.
        # Move to host memory before np.hstack.
        if hasattr(t, "cpu"):
            t = t.cpu()
        return np.asarray(t)

    def get_site_pose(self, name):
        # B-7 follow-up (2026-06-23): Genesis 1.0 has no `site`
        # concept. The MJCF sites ('chassis', 'lidar_site', etc.)
        # either get loaded as joints (when on a body with explicit
        # joints, like the chassis_base chassis joint) or as
        # link-attached geometry. base_env calls
        # get_site_pose('chassis') once per step to publish the
        # chassis TF; that name doesn't match any of the 32
        # joints Genesis exposes. Fall back: try the same name as
        # a link, then for 'chassis' specifically fall back to
        # the first astribot_chassis_* joint (the chassis_base body
        # has 3 free joints — astribot_chassis_{x,y,zrot} — and
        # their parent frame is what the MJCF `chassis` site
        # lives on). Genesis 1.0 deprecated `joint.get_pos()` in
        # favor of operating at link-level, so we use
        # `link.get_pos()` everywhere; `joint.get_anchor_pos()`
        # returns the world-frame anchor which is what we want for
        # the chassis joint fallback. All returned tensors are
        # moved to CPU before np.hstack.
        # Guard: reset()/_get_info() can call this before setup_genesis_model
        # assigns self.robot (env-loop thread races model build). Return a safe
        # identity pose until the entity exists, instead of AttributeError-ing
        # out the whole sim loop.
        if getattr(self, "robot", None) is None:
            return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        try:
            link = self.robot.get_link(name)
            p = self._to_numpy(link.get_pos())
            q = self._to_numpy(link.get_quat())
            return np.hstack((p, q))
        except Exception:
            pass
        if name == "chassis":
            for j in self.robot.joints:
                if j.name.startswith("astribot_chassis_"):
                    p = self._to_numpy(j.get_anchor_pos())
                    q = (
                        self._to_numpy(j.get_anchor_quat())
                        if hasattr(j, "get_anchor_quat")
                        else np.array([1.0, 0.0, 0.0, 0.0])
                    )
                    return np.hstack((p, q))
        return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    def get_body_pose(self):
        pass

    def get_chassis_pose(self):
        return [0, 0, 0, 1, 0, 0, 0]

    def _read_chassis_body_twist(self):
        import numpy as _np

        try:
            def _pos_in_dofidx(jname):
                gid = self.joint_name_to_index[jname]
                return next(i for i, v in enumerate(self.dof_index) if v == gid)

            qvel = self._cached_dofs("velocity")
            vx_w = float(qvel[_pos_in_dofidx("astribot_chassis_x")])
            vy_w = float(qvel[_pos_in_dofidx("astribot_chassis_y")])
            wz = float(qvel[_pos_in_dofidx("astribot_chassis_zrot")])
            iz = self.joint_name_to_index["astribot_chassis_zrot"]
            yaw = float(self.robot.get_dofs_position([iz])[0])
        except Exception:
            return None
        c, s = _np.cos(-yaw), _np.sin(-yaw)
        return [c * vx_w - s * vy_w, s * vx_w + c * vy_w, wz]

    def _integrate_chassis_body_pose(self, dt):
        tw = self._read_chassis_body_twist()
        if tw is None or dt is None or dt <= 0:
            return
        self._chassis_int_pose[0] += tw[0] * dt
        self._chassis_int_pose[1] += tw[1] * dt
        self._chassis_int_pose[2] += tw[2] * dt

    def _rotate_chassis_vel_body_to_world(self):
        """vx_world = vx·cos(yaw) − vy·sin(yaw)
            vy_world = vx·sin(yaw) + vy·cos(yaw)
        """
        names = getattr(self, "joint_names_all", None)
        if not names:
            return
        try:
            ix = names.index("astribot_chassis_x")
            iy = names.index("astribot_chassis_y")
            iz = names.index("astribot_chassis_zrot")
        except ValueError:
            return
        if len(self.controller_mode) <= iz or self.controller_mode[ix] != 2:
            return
        import numpy as _np

        try:
            zdof = self.joint_name_to_index["astribot_chassis_zrot"]
            yaw = float(self.robot.get_dofs_position([zdof])[0])
        except Exception:
            return
        vx = float(self.joint_velocity_command_all[ix])
        vy = float(self.joint_velocity_command_all[iy])
        c, s = _np.cos(yaw), _np.sin(yaw)
        self.joint_velocity_command_all[ix] = c * vx - s * vy
        self.joint_velocity_command_all[iy] = s * vx + c * vy

    def reindex_command_data(self, command):
        left_gripper_data = 0
        right_gripper_data = 0

        for joint_name in self.joint_names:
            if "gripper" in joint_name:
                gripper_dof_index = self.joint_name_to_index[joint_name]
                gripper_index = next(
                    i for i, value in enumerate(self.dof_index) if value == gripper_dof_index
                )

                if len(command) < 32:
                    insert_size = (32 - len(command)) // 2
                    insert_list = [0] * insert_size
                    command = self.insert_values(command, 14, insert_list)
                    command = self.insert_values(command, 27, insert_list)

                if joint_name in [
                    "astribot_gripper_left_joint_L1",
                    "astribot_gripper_left_joint_L11",
                    "astribot_gripper_left_joint_R1",
                    "astribot_gripper_left_joint_R2",
                ]:
                    if joint_name == "astribot_gripper_left_joint_L1":
                        left_gripper_data = command[gripper_index]
                    command[gripper_index] = left_gripper_data / 100 * 0.93
                elif joint_name in [
                    "astribot_gripper_left_joint_L2",
                    "astribot_gripper_left_joint_R11",
                ]:
                    command[gripper_index] = -left_gripper_data / 100 * 0.93

                elif joint_name in [
                    "astribot_gripper_right_joint_L1",
                    "astribot_gripper_right_joint_L11",
                    "astribot_gripper_right_joint_R1",
                    "astribot_gripper_right_joint_R2",
                ]:
                    if joint_name == "astribot_gripper_right_joint_L1":
                        right_gripper_data = command[gripper_index]
                    command[gripper_index] = right_gripper_data / 100 * 0.93
                elif joint_name in [
                    "astribot_gripper_right_joint_L2",
                    "astribot_gripper_right_joint_R11",
                ]:
                    command[gripper_index] = -right_gripper_data / 100 * 0.93

        return command

    def reindex_string_data(self, string):
        left_gripper_data = ""
        right_gripper_data = ""

        for joint_name in self.joint_names:
            if "gripper" in joint_name:
                gripper_dof_index = self.joint_name_to_index[joint_name]
                gripper_index = next(
                    i for i, value in enumerate(self.dof_index) if value == gripper_dof_index
                )

                if len(string) < 32:
                    insert_size = (32 - len(string)) // 2
                    insert_list = [""] * insert_size
                    string = self.insert_values(string, 14, insert_list)
                    string = self.insert_values(string, 27, insert_list)

                if joint_name in [
                    "astribot_gripper_left_joint_L1",
                    "astribot_gripper_left_joint_L11",
                    "astribot_gripper_left_joint_R1",
                    "astribot_gripper_left_joint_R2",
                ]:
                    if joint_name == "astribot_gripper_left_joint_L1":
                        left_gripper_data = string[gripper_index]
                    string[gripper_index] = left_gripper_data
                elif joint_name in [
                    "astribot_gripper_left_joint_L2",
                    "astribot_gripper_left_joint_R11",
                ]:
                    string[gripper_index] = left_gripper_data

                elif joint_name in [
                    "astribot_gripper_right_joint_L1",
                    "astribot_gripper_right_joint_L11",
                    "astribot_gripper_right_joint_R1",
                    "astribot_gripper_right_joint_R2",
                ]:
                    if joint_name == "astribot_gripper_right_joint_L1":
                        right_gripper_data = string[gripper_index]
                    string[gripper_index] = right_gripper_data
                elif joint_name in [
                    "astribot_gripper_right_joint_L2",
                    "astribot_gripper_right_joint_R11",
                ]:
                    string[gripper_index] = right_gripper_data

        return string

    def _mecanum_feedback(self, joint_name, kind):
        order = self.joint_names_list[0]
        idx = order.index(joint_name)  # 0=x,1=y,2=yaw
        if kind == "position":
            return float(self.mecanum_adapter.read_chassis_pose()[idx])
        if kind == "velocity":
            return float(self.mecanum_adapter.read_chassis_twist()[idx])
        return 0.0

    def _cached_dofs(self, kind):
        tick = getattr(self, "_state_cache_tick", 0)
        cache = getattr(self, "_state_cache", None)
        if cache is None or cache.get("tick") != tick:
            cache = {"tick": tick}
            self._state_cache = cache
        if kind not in cache:
            if kind == "position":
                arr = self.robot.get_dofs_position(self.dof_index)
            elif kind == "velocity":
                arr = self.robot.get_dofs_velocity(self.dof_index)
            else:  # force
                arr = self.robot.get_dofs_force(self.dof_index)
            cache[kind] = arr.cpu().numpy().flatten()
        return cache[kind]

    def get_joint_position(self, joint_name):
        if joint_name in getattr(self, "_mecanum_chassis_joints", set()):
            return self._mecanum_feedback(joint_name, "position")
        if joint_name in ("astribot_chassis_x", "astribot_chassis_y", "astribot_chassis_zrot") \
                and self._chassis_int_pose is not None:
            return self._chassis_int_pose[
                ("astribot_chassis_x", "astribot_chassis_y", "astribot_chassis_zrot").index(
                    joint_name)]
        # read via get_dofs_position (dof-index API), NOT get_qpos.
        # dof_index holds DOF indices; qpos/dof spaces differ by 1 (implicit free
        # joint) so get_qpos would read off-by-one (chassis_zrot came back 0 → ring PCD).
        qpos = self._cached_dofs("position")
        qpos_dof_index = self.joint_name_to_index[joint_name]
        qpos_index = [
            index for index, value in enumerate(self.dof_index) if value == qpos_dof_index
        ]

        if self.robot.get_joint(joint_name).type == "free":
            return qpos[qpos_index : qpos_index + 3].item()
        elif "gripper" in joint_name:
            return abs(qpos[qpos_index].item() / 0.93 * 100)
        else:
            return qpos[qpos_index].item()

    def get_joint_positions(self, names):
        pos = []
        for i in names:
            pos.append(self.get_joint_position(i))
        return pos

    def get_joint_velocity(self, joint_name):
        if joint_name in getattr(self, "_mecanum_chassis_joints", set()):
            return self._mecanum_feedback(joint_name, "velocity")
        if (
            getattr(self, "_chassis_int_pose", None) is not None
            and joint_name in ("astribot_chassis_x", "astribot_chassis_y", "astribot_chassis_zrot")
        ):
            twist = self._read_chassis_body_twist()
            if twist is not None:
                return twist[
                    ("astribot_chassis_x", "astribot_chassis_y",
                     "astribot_chassis_zrot").index(joint_name)]

        qvel = self._cached_dofs("velocity")
        qvel_dof_index = self.joint_name_to_index[joint_name]

        qvel_index = [
            index for index, value in enumerate(self.dof_index) if value == qvel_dof_index
        ]

        if self.robot.get_joint(joint_name).type == "free":
            return qvel[qvel_index : qvel_index + 3].item()
        elif "gripper" in joint_name:
            return qvel[qvel_index].item() / 0.93 * 100
        else:
            return qvel[qvel_index].item()

    def get_joint_velocities(self, names):
        vel = []
        for i in names:
            vel.append(self.get_joint_velocity(i))
        return vel

    def get_joint_acceleration(self, joint_name):
        """Genesis 1.0 has no get_dofs_acceleration API. Return 0.0 to match
        the SDK's expected RobotJointState.acceleration array length (otherwise
        SDK's Eigen Map<VectorXd> segfaults when the array is empty).

        TODO: implement numerical differentiation (vel_now - vel_prev) / dt,
        or wait for Genesis 1.1+ to expose qd_solver.dofs_state.acc if it exists.
        """
        return 0.0

    def get_joint_accelerations(self, names):
        acc = []
        for i in names:
            acc.append(self.get_joint_acceleration(i))
        return acc

    def get_joint_torque(self, joint_name):
        if joint_name in getattr(self, "_mecanum_chassis_joints", set()):
            return self._mecanum_feedback(joint_name, "torque")
        qf = self._cached_dofs("force")
        qf_dof_index = self.joint_name_to_index[joint_name]

        qf_index = [index for index, value in enumerate(self.dof_index) if value == qf_dof_index]
        if self.robot.get_joint(joint_name).type == "free":
            return qf[qf_index : qf_index + 3].item()
        else:
            return qf[qf_index].item()

    def get_joint_torques(self, names):
        tor = []
        for i in names:
            tor.append(self.get_joint_torque(i))
        return tor

    def get_near_and_far(self):
        near = 0.01  # 1cm
        far = 100.0  # 100m
        return near, far

    def get_camera_fovy(self, camera_name):
        import math

        normalized_name = camera_name
        if camera_name == "head_rgbd":
            normalized_name = "astribot_head"

        if normalized_name == "astribot_head":
            return math.radians(60)
        elif normalized_name in ("astribot_arm_left_effector", "astribot_arm_right_effector"):
            return math.radians(30)
        else:
            return math.radians(45)

    def get_camera_transform(self, camera_name):
        import numpy as np

        trans = np.eye(4)

        return trans

    def _compute_camera_world_pose(self, camera_name):
        """Compute a camera's world pose by composing its parent link pose with the
        configured local extrinsics. Returns None when the camera has no extrinsics."""
        if camera_name not in self.camera_extrinsics:
            return None

        from simu_utils.transform_utils import (
            multiply_quaternions,
            quaternion_to_lookat,
            transform_position,
        )

        extrinsic = self.camera_extrinsics[camera_name]

        try:
            parent_link = self.robot.get_link(extrinsic.parent_link)
            parent_pos_tensor = parent_link.get_pos()
            parent_quat_tensor = parent_link.get_quat()

            # Tensor → NumPy → tuple
            import numpy as np

            if hasattr(parent_pos_tensor, "cpu"):
                parent_pos = tuple(parent_pos_tensor.cpu().numpy())
                parent_quat = tuple(parent_quat_tensor.cpu().numpy())
            else:
                parent_pos = tuple(np.array(parent_pos_tensor))
                parent_quat = tuple(np.array(parent_quat_tensor))

        except Exception as e:
            astribot_simu_log(
                f"Failed to get parent link pose for {camera_name}: {e}", level="WARN"
            )
            return None

        cam_world_pos = transform_position(extrinsic.local_pos, parent_pos, parent_quat)

        cam_world_quat = multiply_quaternions(parent_quat, extrinsic.local_quat)

        cam_lookat = quaternion_to_lookat(cam_world_pos, cam_world_quat, distance=1.0)

        return cam_world_pos, cam_lookat

    def render_single_camera(self, camera_name):
        try:

            normalized_name = camera_name
            if camera_name == "head_rgbd":
                normalized_name = "astribot_head"
            elif camera_name in ("left_wrist_rgbd", "astribot_arm_left_effector"):
                normalized_name = "astribot_arm_left_effector"
            elif camera_name in ("right_wrist_rgbd", "astribot_arm_right_effector"):
                normalized_name = "astribot_arm_right_effector"

            render_data = None
            if normalized_name == "astribot_head":
                render_data = self.head_camera.render(rgb=True, depth=True)
            elif normalized_name == "astribot_arm_right_effector":
                render_data = self.right_camera.render(rgb=True, depth=True)
            elif normalized_name == "astribot_arm_left_effector":
                render_data = self.left_camera.render(rgb=True, depth=True)
            elif normalized_name == "astribot_global_camera":
                render_data = self.global_camera.render(rgb=True, depth=True)

            if render_data is not None:
                import numpy as np

                rgb_data = None
                depth_data = None

                if isinstance(render_data, (tuple, list)):
                    if len(render_data) >= 1:
                        rgb_data = render_data[0]
                    if len(render_data) >= 2:
                        depth_data = render_data[1]
                else:
                    rgb_data = render_data

                if rgb_data is None:
                    return None
                if hasattr(rgb_data, "cpu"):
                    rgb_data = rgb_data.cpu().numpy()
                elif not isinstance(rgb_data, np.ndarray):
                    rgb_data = np.array(rgb_data)

                if rgb_data.dtype != np.uint8:
                    rgb_data = rgb_data.astype(np.uint8)

                rgb_img = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)

                depth_img = None
                if depth_data is not None:
                    if hasattr(depth_data, "cpu"):
                        depth_data = depth_data.cpu().numpy()
                    elif not isinstance(depth_data, np.ndarray):
                        depth_data = np.array(depth_data)

                    if depth_data.dtype != np.float32:
                        depth_data = depth_data.astype(np.float32)

                    depth_img = cv2.flip(depth_data, 0)

                point_cloud = None
                if depth_img is not None:
                    height, width = depth_img.shape[:2]
                    point_cloud = self.trans_depth_image_to_point_cloud(
                        depth_img, height, width, camera_name
                    )

                return {"rgb_img": rgb_img, "depth_img": depth_img, "point_cloud": point_cloud}
            return None
        except Exception as e:
            astribot_simu_log(f"Error rendering {camera_name}: {e}", level="ERROR")
            return None
