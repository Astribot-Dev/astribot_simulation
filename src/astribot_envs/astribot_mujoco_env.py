#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025, Astribot Co., Ltd.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------
# Author: Tony Wang, tonywang@astribot.com
# -----------------------------------------------------------------------------

"""
File: astribot_mujoco_env.py
Brief: mujoco simulation env
"""

import copy
import math
import os
import random
import time

import cv2
import glfw
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

from astribot_envs.astribot_base_env import AstribotBaseEnv
from simu_utils.simu_common_tools import astribot_simu_log


class AstribotMujocoEnv(AstribotBaseEnv):
    def __init__(self, param):
        super().__init__(param)

        from astribot_envs.simulation_constants import resolve_timing

        self._timing = resolve_timing(param.get("physics", {}) if hasattr(param, "get") else {})
        self.frame_skip = self._timing["frame_skip"]
        # Nominal control period: sim time advanced by one env.step().
        self.control_dt = self.frame_skip * self._timing["dt"]

        astribot_simu_log("Setup astribot base env", level="DEBUG")
        self.setup_mujoco_model_and_data()

        # register a sim-time provider so all
        # ROS publishes (joint_states, F/T, TF, cameras, IMU, LiDAR) stamp
        # messages with PHYSICS time (self.data.time) rather than wall time.
        # Historically only the Genesis backend did this (genesis_env.py:214-217)
        # because it runs at ~21% real-time and wall stamps inflated Fast-LIO's
        # IMU dt. MuJoCo runs ~100% real-time so wall vs sim were nearly
        # interchangeable — but the sensor-scheduler refactor drives sampling off
        # sim-time, so the stamp source MUST match the sampling clock, else the
        # scheduler samples on sim-time while messages carry wall stamps (two
        # clocks → bag inter-frame deltas no longer a clean 5ms/100ms). Wiring
        # this here makes MuJoCo consistent with Genesis and is the prerequisite
        # for PR-2 (scheduler) and PR-3 (camera). The lambda is evaluated lazily
        # at publish time, long after self.data exists. See
        # docs/sensor_scheduler_refactor.md §6.0 (PR-1).
        if hasattr(self, "multi_robot_ros_interface"):
            self.multi_robot_ros_interface.set_sim_time_provider(lambda: float(self.data.time))

        # now that model+data exist and thus
        # dt is known, set up the sim-time sensor scheduler. It will drive IMU/
        # LiDAR/F-T sampling on sim-time ticks in the physics loop instead of the
        # legacy update_sensor_states() every-frame path. See PR-1 comment above
        # for why we do this. base_env.setup_sensor_scheduler registers sensors;
        # step() calls scheduler.tick(data.time) below in the frame_skip loop.
        self.setup_sensor_scheduler(self.model.opt.timestep)

        astribot_simu_log("Setup mujoco renderer for human or camera", level="DEBUG")
        self.setup_mujoco_render_and_camera()

        # opt-in via ASTRIBOT_VIDEO_OUT env var (set by pipeline).
        # Captures off-screen overview camera via mujoco_renderer at 25fps →
        # ffmpeg pipe. Same pattern as Genesis backend.
        self.video_out_path = os.environ.get("ASTRIBOT_VIDEO_OUT", "")
        self.video_fps = int(os.environ.get("ASTRIBOT_VIDEO_FPS", "25"))
        self._video_renderer = None
        self._video_ffmpeg = None
        self._video_step_counter = 0
        if self.video_out_path:
            # Off-screen overview camera (no GUI window). Same viewpoint as
            # the Genesis version: (3, 0, 2.2) lookat (0.5, 0, 0.4), fov=55.
            # MuJoCo doesn't let us add cameras at runtime — we need a named
            # camera in the MJCF. Use 'overview' if present (table scene has
            # it), else fall back to 'human'. The renderer is headless
            # (render_mode='rgb_array'), separate from the main human viewer.
            camera_name = (
                "overview"
                if "overview" in [self.model.camera(i).name for i in range(self.model.ncam)]
                else "human"
            )
            self._video_renderer = MujocoRenderer(
                model=self.model,
                data=self.data,
                width=640,
                height=480,
                camera_name=camera_name,
            )
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
            astribot_simu_log(
                f"PR-G2 video: off-screen '{camera_name}' cam → {self.video_out_path} at {self.video_fps}fps"  # noqa: E501
            )

        astribot_simu_log("Setup joint qpos and qvel mapping", level="DEBUG")
        self.joint_name_to_qpos_index = self.setup_joint_qpos_mapping()
        self.joint_name_to_qvel_index = self.setup_joint_qvel_mapping()
        self.site_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
            for i in range(self.model.nsite)
        ]

        astribot_simu_log("Backup mujoco actuator", level="DEBUG")
        self.actuator_dynprm_bak = self.model.actuator_dynprm.copy()
        self.actuator_gainprm_bak = self.model.actuator_gainprm.copy()
        self.actuator_biasprm_bak = self.model.actuator_biasprm.copy()
        self.actuator_ctrlrange_bak = self.model.actuator_ctrlrange.copy()

        # NOTE: Do NOT call _apply_controller_config_to_actuators() here
        # joint_names_all is initialized later in update_joint_states()

        # joint_names_all index -> MJCF actuator index, resolved BY NAME.
        #
        # These two index spaces are not interchangeable. joint_names_all follows
        # the yaml joint_names_list (25 entries for s1: 3 virtual chassis DOFs +
        # 22 body joints), while the model has 26 actuators (4 omni wheel motors
        # + 22 body). Using a joint index directly as an actuator index shifts
        # every body actuator by one and silently drops the last one — which is
        # why the right gripper never had its gain/bias restored and drifted past
        # its 0.93 rad limit while the left gripper tracked correctly.
        #
        # Virtual chassis joints have no matching actuator (the adapter expands
        # them into wheel speeds), so they map to None and are skipped.
        self._joint_to_actuator_id = []
        for _jname in [j for grp in self.joint_names_list for j in grp]:
            _aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, _jname)
            self._joint_to_actuator_id.append(_aid if _aid >= 0 else None)

        # Chassis freeze: when sim.yaml has `chassis_fixed: true`, step()
        # pins the 3 chassis slide/hinge joints to 0 every step, regardless
        # of what the SDK sends. This decouples chassis behavior from the
        # scene: e.g. `scene: table + chassis_fixed: true` gives a tabletop
        # scene with a welded chassis.
        self.chassis_fixed = (
            bool(param.get("chassis_fixed", False)) if hasattr(param, "get") else False
        )
        self.chassis_actuator_ids = []  # actuator indices for chassis_x/y/zrot
        self.chassis_dof_indices = []  # dofadr for the same joints
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name in ("astribot_chassis_x", "astribot_chassis_y", "astribot_chassis_zrot"):
                self.chassis_actuator_ids.append(i)
                jnt = self.model.actuator_trnid[i, 0]
                self.chassis_dof_indices.append(self.model.jnt_dofadr[jnt])
        if self.chassis_fixed and self.chassis_actuator_ids:
            astribot_simu_log(
                f"chassis_fixed=true: step() will pin chassis dofs "
                f"{self.chassis_dof_indices} to 0 every step"
            )

        self.mecanum_adapter = None
        chassis_type = param.get("chassis_type", "") if hasattr(param, "get") else ""
        chassis_model = (
            param.get("chassis_model", "kinematic") if hasattr(param, "get") else "kinematic"
        )
        if chassis_model == "dynamic" and chassis_type == "omni":
            from simu_utils.chassis_kinematics import MujocoOmniChassis

            cfg = (param.get("omni") or {}) if hasattr(param, "get") else {}
            mec = cfg
            self.mecanum_adapter = MujocoOmniChassis(
                self.model,
                wheel_radius=mec.get("wheel_radius", 0.078),
                half_wheelbase=mec.get("half_wheelbase", 0.216),
                half_track=mec.get("half_track", 0.214),
                base_joint_name=mec.get("base_joint_name", "chassis_free"),
                vel_scale=mec.get("vel_scale", 10.0),
                max_wheel_speed=mec.get("max_wheel_speed", None),
                kd=mec.get("kd", 0.0),
                pos_deadband=mec.get("pos_deadband", 0.0),
                yaw_deadband=mec.get("yaw_deadband", 0.0),
                vel_ff_scale=mec.get("vel_ff_scale", 1.0),
                motor_names=mec.get("motor_names", None),
                wheel_joints=mec.get("wheel_joints", None),
            )
            self._mecanum_body_actuator_ids = [
                i for i in range(self.model.nu) if i not in self.mecanum_adapter.actuator_ids
            ]
            _a = self.mecanum_adapter._base_qadr
            self._mecanum_spawn_x = float(self.data.qpos[_a])
            self._mecanum_spawn_y = float(self.data.qpos[_a + 1])
            self._mecanum_spawn_quat = [float(self.data.qpos[_a + 3 + k]) for k in range(4)]
            astribot_simu_log(
                "chassis_type=omni: MujocoOmniChassis enabled; the chassis 3-DOF pose "
                "command is expanded into 4 wheel-speed ctrl values"
            )

        # LiDAR (M360S) — bound to lidar_site, ray-cast every step.
        # direct mj_multiRay call + LivoxGenerator('mid360') for non-repeating
        # 24000-ray scan pattern (mujoco_lidar 0.3.0 wheel is missing core_cpu/core_ti
        # sub-modules, so we bypass the wrapper library).
        self.lidar_rays = None  # (N, 3) unit vectors in site local frame (legacy)
        self.lidar_site_id = -1
        self.lidar_cutoff = 30.0
        self.lidar_step_counter = 0
        self.latest_lidar_points = None
        # per-point Livox line id (0-3), filled by _get_lidar_points
        self.latest_lidar_lines = None
        # LivoxGenerator instance + reusable mj_multiRay buffers
        self.livox_gen = None
        self.lidar_geomid_buf = None
        self.lidar_dist_buf = None
        if (
            hasattr(param, "get")
            and param.get("lidar", {}).get("enabled", False)
            and not getattr(self, "_lidar_force_disabled", False)
        ):
            self._init_lidar_rays(param["lidar"])

        (
            self.joint_names_all,
            self.controller_mode,
            self.joint_position_command_all,
            self.joint_velocity_command_all,
            self.joint_torque_command_all,
        ) = self.update_joint_states()

        # Apply controller_config gains to override MJCF (after joint_names_all is initialized)
        self._apply_controller_config_to_actuators()

        self.reset_time = time.time()
        self._apply_initial_joint_positions()
        mujoco.mj_forward(self.model, self.data)
        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()

    def _prof_mark(self, label, reset=False):
        import os

        if os.environ.get("ASTRIBOT_PROFILE_STEP") != "1":
            return
        import time as _t

        now = _t.perf_counter()
        if reset:
            self._prof_last = now
            if not hasattr(self, "_prof_acc"):
                from collections import OrderedDict

                self._prof_acc = OrderedDict()
                self._prof_n = 0
            return
        dt = (now - self._prof_last) * 1000.0  # ms
        self._prof_last = now
        self._prof_acc[label] = self._prof_acc.get(label, 0.0) + dt
        if label == "render":
            self._prof_n += 1
            if self._prof_n >= 100:
                total = sum(self._prof_acc.values())
                parts = "  ".join(f"{k}={v/self._prof_n:.1f}ms" for k, v in self._prof_acc.items())
                js_extra = ""
                if hasattr(self, "_prof_js_read"):
                    js_extra = (
                        f"  [js_read={self._prof_js_read/self._prof_n:.1f}ms "
                        f"js_pub={self._prof_js_pub/self._prof_n:.1f}ms]"
                    )
                    self._prof_js_read = 0.0
                    self._prof_js_pub = 0.0
                _raw_rtf = self.frame_skip * self.model.opt.timestep * 1000 / (total / self._prof_n)
                _pace_note = ""
                if hasattr(self, "_prof_paced_wall") and getattr(self, "_prof_paced_n", 0) > 0:
                    _paced_ms = self._prof_paced_wall / self._prof_paced_n * 1000.0
                    _sim_dt_ms = self.frame_skip * self.model.opt.timestep * 1000.0
                    _paced_rtf = _sim_dt_ms / _paced_ms if _paced_ms > 0 else 0.0
                    _pace_note = (
                        f"  |  paced={_paced_ms:.1f}ms/step "
                        f"(paced RTF~{_paced_rtf:.2f}, sim_dt={_sim_dt_ms:.0f}ms)"
                    )
                    self._prof_paced_wall = 0.0
                    self._prof_paced_n = 0
                astribot_simu_log(
                    f"[prof-step] mean over last {self._prof_n} steps: {parts}{js_extra}  "
                    f"total={total/self._prof_n:.1f}ms/step "
                    f"(raw RTF~{_raw_rtf:.2f}){_pace_note}"
                )
                self._prof_acc.clear()
                self._prof_n = 0

    def step(self, action: np.ndarray) -> tuple:
        # DEBUG: Verify this method is called
        if not hasattr(self, '_step_called'):
            self._step_called = True
            print("[COMP-DEBUG] step() method called for the first time", flush=True)
        self.update_reset_flag()
        if not self.reset_flag:
            step_begin_time = time.time()
            self._prof_mark("_begin", reset=True)
            (
                self.joint_names_all,
                self.controller_mode,
                self.joint_position_command_all,
                self.joint_velocity_command_all,
                self.joint_torque_command_all,
            ) = self.update_joint_states()
            self._prof_mark("update_joint_states")
            self.update_object_states()
            self.update_trajectory_pose()
            self.update_com_pose()
            self.update_sensor_states()
            # capture frame every 2 sim steps (50Hz/2 = 25fps)
            self._capture_video_frame()
            self._prof_mark("update_rest")

            # Read compensation configuration
            comp_cfg = self.controller_config.get("dynamics_compensation", {})
            comp_enabled = comp_cfg.get("enabled", True)
            comp_modes = comp_cfg.get("modes", [1, 2, 3])

            # Which dynamics terms to feed forward, driven by the yaml
            # `dynamics_compensation.components` block (previously hardcoded to
            # gravity-only, which silently contradicted `coriolis: true`).
            #
            # All terms are evaluated from the *measured* feedback state
            # (q, qd, qdd), matching how the real robot's middle layer computes its
            # feed-forward -- no desired acceleration is needed or used.
            #
            # `friction` stays unimplemented: the model declares no dof_frictionloss
            # (all zero) so there is no Coulomb friction to cancel; the only
            # dissipation is joint `damping`, which MuJoCo already applies.
            comp_components = comp_cfg.get("components", {})
            include_gravity = bool(comp_components.get("gravity", True))
            include_coriolis = bool(comp_components.get("coriolis", False))
            include_inertia = bool(comp_components.get("inertia", False))
            if comp_components.get("friction") and not getattr(
                self, "_warned_comp_friction", False
            ):
                self._warned_comp_friction = True
                astribot_simu_log(
                    "dynamics_compensation.components.friction=true is not "
                    "implemented (the model declares no dof_frictionloss) and "
                    "will be ignored",
                    level="WARN",
                )

            dynamics_comp = self.compute_dynamics_compensation(
                include_gravity=include_gravity,
                include_coriolis=include_coriolis,
                include_inertia=include_inertia,
            )

            self.data.qfrc_applied = np.zeros(self.model.nv)
            self.model.actuator_ctrlrange = self.actuator_ctrlrange_bak.copy()

            temp_ctrl_data = []
            for joint_id, mode in enumerate(self.controller_mode):
                joint_name = self.joint_names_all[joint_id]

                # Decide if compensation should be applied:
                # 1. Configuration enables it
                # 2. Current mode is in the enabled modes list
                # 3. Not a gripper joint
                # 4. Not a chassis joint (chassis joints are virtual, handled separately)
                should_compensate = (
                    comp_enabled
                    and mode in comp_modes
                    and "gripper" not in joint_name
                    and "chassis" not in joint_name
                )

                if should_compensate:
                    gravity_torque_ids = self.get_joint_id(joint_name)
                    gravity_torque_ids = self.model.jnt_dofadr[gravity_torque_ids]
                    # Apply compensation directly
                    self.data.qfrc_applied[gravity_torque_ids] = dynamics_comp[gravity_torque_ids]

                    # DEBUG: Log compensation vs actual torque for right arm in mode 1
                    # Print first 3 steps for each right arm joint
                    if mode == 1 and "arm_right" in joint_name:
                        if not hasattr(self, '_mode1_debug_count'):
                            self._mode1_debug_count = {}
                        if joint_name not in self._mode1_debug_count:
                            self._mode1_debug_count[joint_name] = 0

                        if self._mode1_debug_count[joint_name] < 3:
                            self._mode1_debug_count[joint_name] += 1
                            qfrc_actuator = self.data.qfrc_actuator[gravity_torque_ids]
                            qfrc_applied = self.data.qfrc_applied[gravity_torque_ids]
                            qfrc_gravity = self.data.qfrc_gravcomp[gravity_torque_ids]
                            qpos = self.data.qpos[self.model.jnt_qposadr[self.get_joint_id(joint_name)]]
                            print(f"[MODE1-DEBUG-{self._mode1_debug_count[joint_name]}] {joint_name}:", flush=True)
                            print(f"  qpos: {qpos:.3f} rad ({np.degrees(qpos):.1f}°)", flush=True)
                            print(f"  comp_applied: {qfrc_applied:.3f} Nm (our compensation)", flush=True)
                            print(f"  qfrc_actuator: {qfrc_actuator:.3f} Nm (actuator force)", flush=True)
                            print(f"  qfrc_gravcomp: {qfrc_gravity:.3f} Nm (MuJoCo gravity)", flush=True)

                if mode == 1:
                    _aid = self._joint_to_actuator_id[joint_id]
                    if _aid is not None:
                        self.set_actuator_parameters(
                            _aid,
                            gainprm=self.actuator_gainprm_bak[_aid],
                            biasprm=self.actuator_biasprm_bak[_aid],
                        )

                    # DEBUG: Print control parameters for right arm joint 2
                    if "arm_right_joint_2" == joint_name and not hasattr(self, '_mode1_ctrl_logged'):
                        self._mode1_ctrl_logged = True
                        qpos_current = self.data.qpos[self.model.jnt_qposadr[self.get_joint_id(joint_name)]]
                        target_pos = self.joint_position_command_all[joint_id]
                        error = target_pos - qpos_current
                        print(f"[MODE1-CTRL] {joint_name}:", flush=True)
                        print(f"  target_pos (command): {target_pos:.6f} rad", flush=True)
                        print(f"  current_pos (qpos):   {qpos_current:.6f} rad", flush=True)
                        print(f"  error: {error:.6f} rad", flush=True)
                        if _aid is not None:
                            print(f"  actuator_gainprm: {self.model.actuator_gainprm[_aid]}", flush=True)
                            print(f"  actuator_biasprm: {self.model.actuator_biasprm[_aid]}", flush=True)

                    temp_ctrl_data += self.vel_compensation_ctrl(
                        self.joint_position_command_all,
                        self.joint_velocity_command_all,
                        idx=joint_id,
                    )
                elif mode == 2:
                    _aid = self._joint_to_actuator_id[joint_id]
                    if _aid is not None:
                        self.set_actuator_parameters(
                            _aid,
                            gainprm=self.actuator_gainprm_bak[_aid],
                            biasprm=self.actuator_biasprm_bak[_aid],
                        )

                    now_position_status = self.get_joint_positions(self.joint_names_all)
                    temp_ctrl_data += self.vel_compensation_ctrl(
                        now_position_status, self.joint_velocity_command_all, idx=joint_id
                    )
                elif mode == 3:
                    # Torque (Zero-G) mode, aligned with the real robot's chain:
                    #
                    #   force = tau_cmd + (C+g compensation) - kd * qd
                    #
                    # i.e. the SDK torque command passes straight through, the
                    # middle layer adds the dynamics feed-forward, and a small
                    # high-frequency damping term acts against velocity with a
                    # desired velocity of zero and kp = 0 (no position feedback).
                    #
                    # Mapped onto MuJoCo's affine actuator
                    #   force = gainprm[0]*ctrl + biasprm[1]*q + biasprm[2]*qd
                    # that means gainprm[0]=1 (pass-through), biasprm[1]=0 (kp=0)
                    # and biasprm[2]=-kd.
                    #
                    # The previous code restored the *position* PD gains here while
                    # feeding a torque into ctrl, so the law became
                    #   force = kp*tau_cmd - kp*q - kd*qd
                    # -- a parasitic spring pulling the joint to q=0 that also
                    # cancelled the gravity compensation in qfrc_applied. With the
                    # SDK sending all-zero torques (examples/210) the arm dropped.
                    _aid = self._joint_to_actuator_id[joint_id]
                    if _aid is not None:
                        self.model.actuator_ctrlrange[_aid] = self.model.actuator_forcerange[
                            _aid
                        ].copy()
                        _kd = self._torque_mode_kd(joint_name)
                        _gainprm = self.actuator_gainprm_bak[_aid].copy()
                        _biasprm = self.actuator_biasprm_bak[_aid].copy()
                        _gainprm[0] = 1.0
                        _biasprm[0] = 0.0
                        _biasprm[1] = 0.0  # kp = 0: no position feedback in Zero-G
                        _biasprm[2] = -_kd
                        self.set_actuator_parameters(
                            _aid, gainprm=_gainprm, biasprm=_biasprm
                        )
                    temp_ctrl_data += [self.joint_torque_command_all[joint_id]]

            if self.mecanum_adapter is not None:
                full_ctrl = np.zeros(self.model.nu)
                chassis_target = temp_ctrl_data[:3]
                body_ctrl = temp_ctrl_data[3:]
                chassis_mode = self.controller_mode[0] if self.controller_mode else 1
                for aid, val in zip(self._mecanum_body_actuator_ids, body_ctrl):
                    full_ctrl[aid] = val
                self.data.ctrl = full_ctrl
                if not self.chassis_fixed:
                    if chassis_mode == 2:
                        twist = [-v for v in self.joint_velocity_command_all[:3]]
                        self.mecanum_adapter.apply(self.data, twist)
                    else:
                        self.mecanum_adapter.apply_pose_command(
                            self.data,
                            list(chassis_target),
                            dt=self.frame_skip * self.model.opt.timestep,
                        )
            else:
                self.data.ctrl = temp_ctrl_data

            # Chassis freeze (pre-step): when chassis_fixed=true, override
            # chassis actuator commands to 0 so the position controller
            # doesn't try to follow the SDK's chassis commands.
            if self.chassis_fixed:
                for i in self.chassis_actuator_ids:
                    self.data.ctrl[i] = 0.0
                if self.mecanum_adapter is not None:
                    for i in self.mecanum_adapter.actuator_ids:
                        self.data.ctrl[i] = 0.0

            for _ in range(self.frame_skip):
                mujoco.mj_step(self.model, self.data)

                # tick the sim-time scheduler after every physics step
                # (inside frame_skip loop, not outside). Sensors that are due
                # (IMU/LiDAR/F-T) sample+publish here on sim-time alignment.
                if self._sensor_scheduler is not None:
                    self._sensor_scheduler.tick(float(self.data.time))
            self._prof_mark("ctrl+mj_step")

            if self.mecanum_adapter is not None and not self.chassis_fixed:
                self.mecanum_adapter._data = self.data
                self.mecanum_adapter.integrate_state(self.frame_skip * self.model.opt.timestep)

            # Chassis freeze (post-step): force chassis qpos/qvel to 0.
            # Even though ctrl=0 and the position controller is holding
            # the chassis against gravity, we explicitly zero out the
            # position to defeat any numerical drift and ensure the
            # chassis stays exactly at (0, 0, 0) world-frame.
            if self.chassis_fixed:
                for d in self.chassis_dof_indices:
                    self.data.qpos[d] = 0.0
                    self.data.qvel[d] = 0.0
                if self.mecanum_adapter is not None:
                    self._freeze_mecanum_base()

            # LiDAR old throttle path migrated to the scheduler (above
            # tick). During migration stage 1 (ASTRIBOT_USE_SENSOR_SCHEDULER=0),
            # keep this for back-compat. Stage 3: delete this entire block.
            if not self._use_sensor_scheduler:
                # LiDAR ray-cast: sample hits at lidar_site after physics step.
                # Throttle by env-loop count (this block runs once per step() =
                lidar_freq = float(getattr(self, "lidar_frequency", 10.0)) or 10.0
                lidar_mod = max(1, round(self._timing["control_hz"] / lidar_freq))
                self.lidar_step_counter += 1
                if self.livox_gen is not None and (self.lidar_step_counter % lidar_mod == 0):
                    # _get_lidar_points now returns ((N,3) local, (N,) line_ids)
                    points, lines = self._get_lidar_points()
                    if points is not None and len(points) > 0:
                        self.latest_lidar_points = np.asarray(points, dtype=np.float32)
                        self.latest_lidar_lines = np.asarray(lines, dtype=np.uint8)
                    else:
                        self.latest_lidar_points = None
                        self.latest_lidar_lines = None
                    self._publish_lidar_custommsg()

            self.render()
            self._prof_mark("render")

            step_end_time = time.time()
            self.real_time_fps = 1 / (step_end_time - step_begin_time)
            self.report_realtime_factor(float(self.data.time))

            sim_time_delta = self.frame_skip * self.model.opt.timestep
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

            if os.environ.get("ASTRIBOT_PROFILE_STEP") == "1":
                if hasattr(self, "_prof_prev_begin"):
                    self._prof_paced_wall = getattr(self, "_prof_paced_wall", 0.0) + (
                        step_begin_time - self._prof_prev_begin
                    )
                    self._prof_paced_n = getattr(self, "_prof_paced_n", 0) + 1
                self._prof_prev_begin = step_begin_time

            from astribot_envs.simulation_constants import RTF_WARNING_THRESHOLD

            wall_time_delta = step_end_time - step_begin_time
            rtf = sim_time_delta / wall_time_delta if wall_time_delta > 0 else 0.0
            if not hasattr(self, "_rtf_window"):
                from collections import deque

                self._rtf_window = deque(maxlen=50)
                self._rtf_warn_counter = 0
            self._rtf_window.append(rtf)
            self.rtf_avg = sum(self._rtf_window) / len(self._rtf_window)
            self._rtf_warn_counter += 1
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
        else:
            astribot_simu_log("Reset and setup joint interface", level="DEBUG")
            self.reset()

        observation = self._get_obs()
        reward = 0
        terminated = False
        info = self._get_info()

        return observation, reward, terminated, False, info

    def _init_lidar_rays(self, lidar_cfg):
        """PR-2: Setup MID-360 LiDAR via LivoxGenerator + mj_multiRay direct.

        Pattern source: `mujoco_lidar.scan_gen.LivoxGenerator('mid360')` loads
        the Livox-provided non-repeating scan table from `mid360.npy`
        (800000 rows, 4-line interleaved). Each call to `sample_ray_angles()`
        returns the next 24000 (azimuth, elevation) pairs and advances
        `currStartIndex` so successive sweeps produce distinct angles (the
        "non-repeating" property of MID-360).

        Ray dirs are still in `lidar_site` local frame. We transform to
        world frame each step via `site_xmat @ local` (D6 reverse: world
        hits are then re-expressed in local frame for the livox_ros_driver2
        CustomMsg builder).

        Reuses the `bodyexclude` field on the lidar_site parent body to drop
        robot self-hits.
        """
        site_name = lidar_cfg.get("site_name", "lidar_site")
        self.lidar_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.lidar_site_id < 0:
            astribot_simu_log(
                f"LiDAR site {site_name!r} not found in model; LiDAR disabled",
                level="WARN",
            )
            return
        # params (override PR-1 defaults).
        self.lidar_cutoff = float(lidar_cfg.get("cutoff_dist", 40.0))  # mid360 spec
        # Min hit distance — drops self-hit residuals that slip past bodyexclude
        # (floating-point edge at site origin, hardware mount brackets, etc.).
        self.lidar_min_dist = float(lidar_cfg.get("min_dist", 0.1))  # mid360 spec
        # Body the lidar_site is attached to; mj_ray/mj_multiRay will exclude
        # this body and all descendants so rays can't self-hit.
        self.lidar_body_id = int(self.model.site_bodyid[self.lidar_site_id])

        # Lazy import: `mujoco_lidar.scan_gen` is a small (numpy-only) module
        # that ships with the 0.3.0 wheel. We do NOT use `mujoco_lidar.core_*`
        # (CPU/Taichi backends) because the wheel 0.3.0 ships without those
        # sub-modules and the underlying `MjLidarCPU` has a float32→float64
        # dtype bug for `mj_multiRay` arguments.
        try:
            from mujoco_lidar.scan_gen import LivoxGenerator
        except ImportError as e:
            astribot_simu_log(f"LivoxGenerator unavailable, LiDAR disabled: {e}", level="ERROR")
            return
        pattern = lidar_cfg.get("scan_pattern", "mid360")
        self.livox_gen = LivoxGenerator(pattern)
        n_rays = int(self.livox_gen.samples)
        # mj_multiRay output buffers
        self.lidar_geomid_buf = np.zeros(n_rays, dtype=np.int32)
        self.lidar_dist_buf = np.zeros(n_rays, dtype=np.float64)
        # Geom group filter: only test group 0 (walls, floor, table, box_red
        # are all default group 0). Robot internal visual meshes (chassis,
        # torso, head, arms, grippers) are all in group 1 and so excluded.
        # See _get_lidar_points for the full rationale.
        self._lidar_geomgroup = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        # `latest_lidar_lines` is set per-scan in _get_lidar_points; init to
        # an empty array to keep `lidar_lines` attribute always present.
        self.latest_lidar_lines = np.zeros(0, dtype=np.uint8)
        astribot_simu_log(
            f"LiDAR mj_multiRay+mid360 ready: site={site_name} rays={n_rays} pattern={pattern} "
            f"cutoff={self.lidar_cutoff}m min={self.lidar_min_dist}m geomgroup={list(self._lidar_geomgroup)}"  # noqa: E501
        )

    def _get_lidar_points(self):
        """PR-2: Cast MID-360 non-repeating rays via mj_multiRay.

        Returns ((N, 3) hit-points in lidar_site LOCAL frame, (N,) line ids).
        The local frame convention matches the livox_ros_driver2 CustomMsg
        header.frame_id ('lidar_site'), so downstream consumers (slam_node)
        get consistent points.

        Performance: 24000 rays in ~67ms on CPU (60x faster than 5000 rays in
        a Python mj_ray loop), well under the 100ms budget for 10Hz publish.

        Line id is recovered from `LivoxGenerator.currStartIndex` snapshot
        (see `simu_utils.livox_line_classifier.assign_livox_line_ids`):
        the npy is line-interleaved (every 4 rows = line 0,1,2,3 repeat), so
        (npy_index % 4) gives the line.
        """
        if self.livox_gen is None or self.lidar_site_id < 0:
            return None, None
        try:
            from simu_utils.livox_line_classifier import assign_livox_line_ids
        except ImportError:
            astribot_simu_log("livox_line_classifier unavailable", level="ERROR")
            return None, None

        n_rays = int(self.livox_gen.samples)
        # Snapshot start BEFORE sample_ray_angles mutates currStartIndex.
        line_ids_full = assign_livox_line_ids(self.livox_gen, n_rays)
        theta, phi = self.livox_gen.sample_ray_angles()  # both (n_rays,) float32

        # World-frame origin and rotation of lidar_site.
        pnt_world = self.data.site_xpos[self.lidar_site_id].copy()  # (3,)
        xmat = self.data.site_xmat[self.lidar_site_id].reshape(3, 3)
        # LivoxGenerator convention (matches MjLidarCPU.trace_rays):
        #   local = (cos(phi)*cos(theta), cos(phi)*sin(theta), sin(phi))
        # i.e. theta = azimuth (around z), phi = elevation from +z.
        local_x = np.cos(phi) * np.cos(theta)
        local_y = np.cos(phi) * np.sin(theta)
        local_z = np.sin(phi)
        local_vecs = np.stack([local_x, local_y, local_z], axis=-1)  # (n_rays, 3)
        # local -> world: world_vec = xmat @ local_vec.T, then transpose.
        world_vecs = (xmat @ local_vecs.T).T
        # Normalize (defensive; should be unit already).
        world_vecs = world_vecs / np.linalg.norm(world_vecs, axis=1, keepdims=True)

        # mj_multiRay requires (3, 1) pnt and flat (3*nray,) vec, all float64.
        pnt = pnt_world.reshape(3, 1).astype(np.float64)
        vec = world_vecs.flatten().astype(np.float64)  # length 3*n_rays
        if self.lidar_geomid_buf.shape[0] != n_rays:
            self.lidar_geomid_buf = np.zeros(n_rays, dtype=np.int32)
            self.lidar_dist_buf = np.zeros(n_rays, dtype=np.float64)

        # flg_static in MuJoCo 3.2.5 Python binding:
        #   0 = test non-static geoms only
        #   1 = test static geoms only
        #   2 = test both (this is what we want — walls + floor are static,
        #                    table + robot are non-static)
        # used flg_static=0 with the assumption it meant "both"; that
        # silently dropped static geoms (walls, floor) so only the robot
        # self-hits appeared in the bag. Discovered when walls in the
        # aloha scene produced 0 hits (2026-06-17, walls PCD regen task).
        #
        # also switch from bodyexclude=chassis_base to
        # geomgroup=[1,0,...]. The chassis mesh is in MuJoCo `geomgroup=1`
        # (visual-only), and so are all the other robot internal meshes
        # (torso, head, arms, grippers). Limiting mj_multiRay to group 0
        # alone skips the entire robot kinematic tree, not just the
        # chassis_base branch — which is exactly what we want for a
        # lidar that "doesn't see itself" (real mid360 hardware doesn't).
        # Previously bodyexclude=chassis_base only excluded the chassis
        # mesh, leaving rays to hit arm/wheel/torso meshes that occluded
        # the back wall (creating a ~70° angular gap centered behind the
        # robot). geomgroup=[1,0,0,0,0,0] closes that gap.
        try:
            mujoco.mj_multiRay(
                m=self.model,
                d=self.data,
                pnt=pnt,
                vec=vec,
                geomgroup=self._lidar_geomgroup,
                flg_static=2,
                bodyexclude=-1,  # geomgroup filter handles self-exclusion
                geomid=self.lidar_geomid_buf,
                dist=self.lidar_dist_buf,
                nray=n_rays,
                cutoff=self.lidar_cutoff,
            )
        except Exception as e:
            astribot_simu_log(f"LiDAR mj_multiRay failed: {e}", level="WARN")
            return None, None

        # mj_multiRay semantics:
        #   geomid[i] == -1  -> no hit
        #   dist[i] = hit distance  (capped at `cutoff` for hits; -1 for miss
        #                            is set by the C side to a large value,
        #                            but in our 3.2.5 binding, miss dist is
        #                            left as the initial 0 from np.zeros).
        # We treat dist == 0 OR geomid == -1 as no-hit, then filter
        # dist < min_dist as self-hit residual.
        dist = self.lidar_dist_buf
        valid = (self.lidar_geomid_buf != -1) & (dist > 0) & (dist < self.lidar_cutoff + 1e-3)
        if not np.any(valid):
            return None, None
        # Apply min_dist safety net (drops self-hit residuals).
        valid = valid & (dist >= self.lidar_min_dist)
        if not np.any(valid):
            return None, None
        hit_idx = np.where(valid)[0]
        hit_dist = dist[hit_idx]
        hit_dirs_world = world_vecs[hit_idx]
        hit_points_world = pnt_world[None, :] + hit_dirs_world * hit_dist[:, None]
        # World -> lidar_site local frame (D6).
        xmat_inv = xmat.T
        hit_points_local = (hit_points_world - pnt_world[None, :]) @ xmat_inv.T
        line_ids = line_ids_full[hit_idx].astype(np.uint8)
        return hit_points_local.astype(np.float32), line_ids

    def _sample_publish_lidar(self):
        """PR-2: LiDAR sample+publish for the sim-time scheduler (MuJoCo).
        Ray-casts at lidar_site then publishes CustomMsg. Replaces the old
        `% 5` throttle in step() — the scheduler owns the 10Hz cadence now."""
        if self.livox_gen is None:
            return
        points, lines = self._get_lidar_points()
        if points is not None and len(points) > 0:
            self.latest_lidar_points = np.asarray(points, dtype=np.float32)
            self.latest_lidar_lines = np.asarray(lines, dtype=np.uint8)
        else:
            self.latest_lidar_points = None
            self.latest_lidar_lines = None
        self._publish_lidar_custommsg()

    def _publish_lidar_custommsg(self):
        """Publish latest_lidar_points + line_ids to /livox/lidar_front as a
        `livox_ros_driver2_msgs/CustomMsg` (PR-3: switched from
        `astribot_msgs/LivoxCustomMsg` mirror to align with LioManager.cpp:751).

        Called from step() after ray-cast. The header.frame_id is set to
        'lidar_site' to match the local frame that _get_lidar_points emits.
        """
        pub = getattr(self.multi_robot_ros_interface, "lidar_custommsg_publisher", None)
        if (
            pub is None
            or self.latest_lidar_points is None
            or len(self.latest_lidar_points) == 0
            or self.latest_lidar_lines is None
        ):
            return
        try:
            from simu_utils.lidar_msg_builder import build_livox_custom_msg

            pts = np.asarray(self.latest_lidar_points, dtype=np.float32)
            lines = np.asarray(self.latest_lidar_lines, dtype=np.uint8)
            if pts.ndim != 2 or pts.shape[1] < 3 or lines.shape[0] != pts.shape[0]:
                return
            msg = build_livox_custom_msg(
                points_xyz=pts[:, :3],
                line_ids=lines,
                timebase_ns=time.time_ns(),
                frame_id="lidar_site",
                lidar_id=1,
                reflectivity=100,
            )
            # Stamp is set by the caller via get_timestamp() to stay consistent
            # with the rest of the publishers (sensor_msgs/Imu etc.).
            msg.header.stamp = self.multi_robot_ros_interface.get_timestamp()
            pub.publish(msg)
        except Exception as e:
            astribot_simu_log(f"LiDAR custommsg publish failed: {e}", level="WARN")

    def reset(self, seed=None, options=None):
        astribot_simu_log("Reset mujoco data and step", level="DEBUG")
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)

        self.reset_from_keyframe()
        self.reset_object_pose()

        if self.mecanum_adapter is not None:
            self.mecanum_adapter.reset_feedforward()

        self.reset_flag = False

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def check_joint_control_type(self):
        postion_control_ids = []
        torque_control_ids = []
        actuator_control_ids = []

        actuator_names = self.get_actuator_names()

        for actuator_name in actuator_names:
            if actuator_name:
                split = actuator_name.split("_")
                if len(split) >= 2:
                    if split[1] == "motor":
                        joint_name = actuator_name.replace("motor", "joint")
                        torque_control_ids.append(self.get_joint_id(joint_name))
                        actuator_control_ids.append(self.get_actuator_id(actuator_name))
                    else:
                        joint_name = actuator_name
                        postion_control_ids.append(self.get_joint_id(joint_name))
                        actuator_control_ids.append(self.get_actuator_id(actuator_name))
                else:
                    joint_name = actuator_name
                    postion_control_ids.append(self.get_joint_id(joint_name))

        trntype = self.model.actuator_trntype

        trnid = self.model.actuator_trnid

        ctrl_joints = []
        for i in range(trnid.shape[0]):
            if trntype[i] == 0 and trnid[i, 0] not in ctrl_joints:
                ctrl_joints.append(trnid[i, 0])

        self.controllable_joints = ctrl_joints

        astribot_simu_log("postion_control_ids: ", postion_control_ids, level="DEBUG")
        astribot_simu_log("torque_control_ids: ", torque_control_ids, level="DEBUG")
        astribot_simu_log("actuator_control_ids: ", actuator_control_ids, level="DEBUG")
        return postion_control_ids, torque_control_ids, actuator_control_ids

    def _identify_actuators(self, joint_ids):
        pos_actuator_ids = []
        torque_actuator_ids = []

        self._joint_to_actuator_map = {}
        for idx, jnt in enumerate(self.model.actuator_trnid[:, 0].tolist()):
            assert jnt == joint_ids[idx]
            actuator_name = self.get_actuator_name(idx)
            controller_type = actuator_name.split("_")[-1]
            if controller_type == "torque" or controller_type == "motor":
                torque_actuator_ids.append(idx)
                assert (
                    jnt not in self._joint_to_actuator_map
                ), "Joint {} already has an actuator assigned!".format(
                    self.model.joint_id2name(jnt)
                )
                self._joint_to_actuator_map[jnt] = idx
            elif controller_type == "position" or controller_type == "joint":
                pos_actuator_ids.append(idx)
                assert (
                    jnt not in self._joint_to_actuator_map
                ), "Joint {} already has an actuator assigned!".format(
                    self.model.joint_id2name(jnt)
                )
                self._joint_to_actuator_map[jnt] = idx
            else:
                pos_actuator_ids.append(idx)
                assert (
                    jnt not in self._joint_to_actuator_map
                ), "Joint {} already has an actuator assigned!".format(
                    self.model.joint_id2name(jnt)
                )
                self._joint_to_actuator_map[jnt] = idx
                astribot_simu_log(
                    "Unknown actuator type Ignoring. This actuator will not be controllable via PandaArm api."  # noqa: E501
                )

        return pos_actuator_ids, torque_actuator_ids

    def impedance_control(self, qpos_des, kp=5000, kd=0, qvel_des=None):
        gravity_torque = self.get_gravity_torque()

        control_torque = self.data.ctrl.copy()

        if qvel_des is None:
            qvel_des = np.zeros(self.model.nv)

        for i in range(len(qpos_des)):
            if i in self.torque_control_ids:
                index = self.torque_control_ids.index(i)
                joint_id = self.actuator_control_ids[index]
                control_torque[i] = (
                    kp * (qpos_des[i] - self.data.qpos[joint_id])
                    + kd * (qvel_des[i] - self.data.qvel[joint_id])
                    + gravity_torque[joint_id]
                )
        return control_torque

    def _get_component_name_from_joint(self, joint_name):
        """Extract component name from joint name.

        E.g., "astribot_arm_left_joint_1" -> "astribot_arm_left"
              "astribot_chassis_x" -> "astribot_chassis"
        """
        for component in self.robot_list:
            if joint_name.startswith(component):
                return component
        return "unknown"

    def _apply_controller_config_to_actuators(self):
        """Apply controller_config gains to override MJCF actuator parameters at init.

        Priority: per_joint > per_component > MJCF (respect MJCF unless overridden)

        Only applies when controller_config has per_joint or per_component entries.
        If neither is configured, MJCF values are preserved.
        """
        if not self.controller_config:
            return

        per_joint = self.controller_config.get("per_joint", {})
        per_component = self.controller_config.get("per_component", {})

        # If no overrides configured, respect MJCF
        if not per_joint and not per_component:
            return

        for joint_id, joint_name in enumerate(self.joint_names_all):
            _aid = self._joint_to_actuator_id[joint_id]
            if _aid is None:
                continue  # Virtual joint (e.g., chassis), no actuator

            component_name = self._get_component_name_from_joint(joint_name)

            # Only the position gains belong in the actuator at init.
            #
            # A MuJoCo actuator has exactly one gainprm/biasprm triple, so writing
            # the velocity and torque gains into the same slots here would make the
            # last configured mode win. That is not hypothetical: adding the mode-3
            # block (torque: {kp: 0.0, ...}) made every joint start with
            # gainprm[0]=0, i.e. no position authority at all -- mode=1 stopped
            # moving and the robot could not return to home pose.
            #
            # mode=2 and mode=3 set their own actuator parameters per step inside
            # step(), so they need nothing at init.
            for mode, mode_name in [(1, "position")]:
                gains = None

                # Priority 1: per_joint
                if joint_name in per_joint and mode_name in per_joint[joint_name]:
                    cfg = per_joint[joint_name][mode_name]
                    gains = (cfg["kp"], cfg["kd"])

                # Priority 2: per_component
                elif component_name in per_component:
                    comp_cfg = per_component[component_name]
                    if "default_gains" in comp_cfg and mode_name in comp_cfg["default_gains"]:
                        cfg = comp_cfg["default_gains"][mode_name]
                        gains = (cfg["kp"], cfg["kd"])

                # If gains found, apply to actuator
                if gains:
                    kp, kd = gains
                    # MuJoCo actuator parameter layout:
                    # gainprm[0] = Kp (position gain)
                    # biasprm[0] = bias
                    # biasprm[1] = NOT USED (setting to 1.0 causes instability)
                    # biasprm[2] = Kd (damping/velocity gain)
                    # FIXED P0-2: Kd should be in biasprm[2], not biasprm[1]
                    self.model.actuator_gainprm[_aid][0] = kp
                    self.model.actuator_biasprm[_aid][2] = -kd  # Corrected slot
                    # NOTE: biasprm[1] left as default (0) - setting to 1.0 caused oscillation
                    # Update backup so step() uses the new values
                    self.actuator_gainprm_bak[_aid] = self.model.actuator_gainprm[_aid].copy()
                    self.actuator_biasprm_bak[_aid] = self.model.actuator_biasprm[_aid].copy()

    def _torque_mode_kd(self, joint_name):
        """Damping coefficient for mode-3 (Zero-G) torque control.

        Mirrors the real robot's middle layer: a small high-frequency damping term
        with a desired velocity of zero, on top of the pass-through torque command
        and the dynamics feed-forward. kp is 0 there, so only kd is configurable.

        Priority matches ControllerConfig.get_gains: per_joint > per_component >
        default_gains. Values are currently empirical placeholders sized off the
        real controller's motor_kd_damping, not calibrated for feel.
        """
        cfg = self.controller_config
        if not isinstance(cfg, dict):
            return 0.0

        per_joint = cfg.get("per_joint", {}) or {}
        if joint_name in per_joint and "torque" in per_joint[joint_name]:
            return float(per_joint[joint_name]["torque"].get("kd", 0.0))

        component = next(
            (c for c in (self.robot_list or []) if joint_name.startswith(c)), "unknown"
        )
        per_component = cfg.get("per_component", {}) or {}
        comp_cfg = per_component.get(component, {}) or {}
        gains = comp_cfg.get("default_gains", {}) or {}
        if "torque" in gains:
            return float(gains["torque"].get("kd", 0.0))

        default_gains = cfg.get("default_gains", {}) or {}
        if "torque" in default_gains:
            return float(default_gains["torque"].get("kd", 0.0))
        return 0.0

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
        if self.render_mode == "human":
            n = getattr(self, "_human_render_every", 1)
            if getattr(self, "_use_passive_viewer", False):
                self._render_passive(n)
                return
            # The passive-viewer path sets mujoco_renderer to None, and closing that
            # window clears _use_passive_viewer. Without this guard the next call
            # falls through to the sync renderer and raises AttributeError on None.
            if self.mujoco_renderer is None:
                return
            if n <= 1:
                self.mujoco_renderer.render("human")
            else:
                self._human_render_counter = getattr(self, "_human_render_counter", 0) + 1
                if self._human_render_counter % n == 0:
                    self.mujoco_renderer.render("human")
        elif self.render_mode == "rgb_array":
            for camera_name in self.camera_names:
                camera_data = self.render_single_camera(camera_name)
                if camera_data is not None:
                    self._publish_camera_data(camera_name, camera_data)

    def _render_passive(self, n):
        viewer = getattr(self, "_passive_viewer", None)
        if viewer is None:
            return
        if not viewer.is_running():
            self._close_passive_viewer()
            self._use_passive_viewer = False
            return
        self._human_render_counter = getattr(self, "_human_render_counter", 0) + 1
        if n > 1 and self._human_render_counter % n != 0:
            return
        try:
            viewer.sync(state_only=True)
        except TypeError:
            viewer.sync()

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
            # Off-screen render via gymnasium MujocoRenderer in rgb_array mode.
            # Returns uint8 (H, W, 3); current data state is rendered.
            rgb = self._video_renderer.render("rgb_array")
            self._video_ffmpeg.stdin.write(rgb.tobytes())
        except Exception as e:
            astribot_simu_log(
                f"PR-G2 video frame capture failed: {e}",
                level="WARN",
            )

    def close(self):
        self._close_passive_viewer()
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()
        self.shutdown_async_camera_rendering()

    def render_single_camera(self, camera_name):
        try:
            if camera_name not in self.camera_dict:
                return None

            renderer = self.camera_dict[camera_name]
            rgb_img = renderer.render("rgb_array")
            depth_img = renderer.render("depth_array")

            _display = self.param.get("display", {}) or {}
            _cam_res = _display.get("camera_resolutions") or {}
            if camera_name in _cam_res:
                width, height = _cam_res[camera_name]
            else:
                width = self.width or 640
                height = self.height or 480

            target_size = (int(width), int(height))
            rgb_img = cv2.resize(rgb_img, target_size, interpolation=cv2.INTER_LINEAR)
            rgb_img = cv2.flip(rgb_img, 0)

            depth_img = cv2.resize(depth_img, target_size, interpolation=cv2.INTER_LINEAR)
            depth_img = cv2.flip(depth_img, 0)

            point_cloud = self.trans_depth_image_to_point_cloud(
                depth_img, height, width, camera_name
            )

            return {"rgb_img": rgb_img, "depth_img": depth_img, "point_cloud": point_cloud}
        except Exception as e:
            astribot_simu_log(f"Error rendering {camera_name}: {e}", level="ERROR")
            return None

    def reset_from_keyframe(self):
        try:
            self.data.qpos[:] = self.initial_qpos
            self.data.qvel[:] = self.initial_qvel
            astribot_simu_log("Reset from keyframe successfully")
        except Exception:
            astribot_simu_log("reset_from_keyframe none", level="WARN")

        astribot_simu_log(
            "Pressing the backspace key resets the robot simulator state, but can only be triggered once per second"  # noqa: E501
        )

    def reset_object_pose(self, object_name="object"):
        delta_pos = [random.uniform(0.6, 0.8), random.uniform(-0.2, 0.2), 0.05]
        twist = [0, 0, 0, 0, 0, random.uniform(-50, 50)]
        self.set_body_pose(object_name, delta_pos=delta_pos, twist=twist)

    def _resolve_passive_viewer(self):
        """ASTRIBOT_MUJOCO_PASSIVE!=0）；"""
        self._use_passive_viewer = bool(
            getattr(self, "_passive_viewer_requested", False)
            and self.render_mode == "human"
            and hasattr(mujoco.viewer, "launch_passive")
        )
        self._passive_backspace_pressed = False
        return self._use_passive_viewer

    def _passive_key_callback(self, keycode):
        if keycode == glfw.KEY_BACKSPACE:
            self._passive_backspace_pressed = True

    def _setup_passive_viewer(self):
        self._passive_viewer = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=self._passive_key_callback,
        )
        _cfg = self._resolve_human_cam_config()
        if _cfg is not None:
            try:
                _cam = self._passive_viewer.cam
                _cam.azimuth = _cfg["azimuth"]
                _cam.elevation = _cfg["elevation"]
                _cam.distance = _cfg["distance"]
                for i in range(3):
                    _cam.lookat[i] = _cfg["lookat"][i]
            except Exception:
                pass
        astribot_simu_log(
            "sim_profile=realtime: MuJoCo passive viewer started (async rendering, so "
            "the physics loop is not blocked by GLFW vsync); Backspace reset goes "
            "through key_callback. Set ASTRIBOT_MUJOCO_PASSIVE=0 for sync rendering."
        )

    def _close_passive_viewer(self):
        if getattr(self, "_passive_viewer", None) is not None:
            try:
                self._passive_viewer.close()
            except Exception:
                pass
            self._passive_viewer = None

    def _resolve_human_cam_config(self):
        g = self.model.vis.global_
        try:
            azimuth = float(g.azimuth)
            elevation = float(g.elevation)
            center = [float(c) for c in self.model.stat.center]
            distance = float(self.model.stat.extent)
        except Exception:
            return None
        return {
            "azimuth": azimuth,
            "elevation": elevation,
            "distance": distance,
            "lookat": center,
        }

    def setup_mujoco_render_and_camera(self):
        self._passive_viewer = None
        if self._resolve_passive_viewer():
            self.mujoco_renderer = None
            self._setup_passive_viewer()
            return

        self.mujoco_renderer = MujocoRenderer(
            model=self.model,
            data=self.data,
            width=self.width,
            height=self.height,
            camera_name="human",
            default_cam_config=self._resolve_human_cam_config(),
        )

        if self.render_mode == "rgb_array":
            self.camera_dict = dict()
            for camera_name in self.camera_names:
                camera_renderer = MujocoRenderer(
                    model=self.model, data=self.data, width=640, height=480, camera_name=camera_name
                )

                # ERROR::exchange left and right
                if camera_name == "left_wrist_rgbd":
                    self.camera_dict["right_wrist_rgbd"] = camera_renderer
                elif camera_name == "right_wrist_rgbd":
                    self.camera_dict["left_wrist_rgbd"] = camera_renderer
                else:
                    self.camera_dict[camera_name] = camera_renderer

    def setup_mujoco_model_and_data(self):
        astribot_simu_log("Setup mujoco model and data")
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        mjcf_dt = self.model.opt.timestep
        resolved_dt = self._timing["dt"]
        self.model.opt.timestep = float(resolved_dt)
        astribot_simu_log(
            f"backend=mujoco robot={self.param.get('robot_type', '?')} "
            f"profile={getattr(self, 'sim_profile', '?')} "
            f"physics_hz={self._timing['physics_hz']:.0f} "
            f"control_hz={self._timing['control_hz']:.0f} "
            f"frame_skip={self._timing['frame_skip']} dt={resolved_dt:.4f}s "
            f"scene={os.path.basename(os.path.dirname(self.model_path))} "
            f"(mjcf default dt={mjcf_dt}s)"
        )

        self.model_for_cal = copy.deepcopy(self.model)
        for i in range(self.model_for_cal.ngeom):
            self.model_for_cal.geom_contype[i] = 0
            self.model_for_cal.geom_conaffinity[i] = 0
        self.data_for_cal = mujoco.MjData(self.model_for_cal)

    def _apply_initial_joint_positions(self):
        init = self.param.get("initial_joint_positions") if hasattr(self.param, "get") else None
        if not init:
            return
        for joint_name, value in init.items():
            qpos_index = self.joint_name_to_qpos_index.get(joint_name)
            if qpos_index is None:
                astribot_simu_log(
                    f"initial_joint_positions: unknown joint '{joint_name}', skipped",
                    level="WARN",
                )
                continue
            self.data.qpos[qpos_index] = float(value)

    def setup_joint_qpos_mapping(self):
        joint_name_to_qpos_index = {}
        for joint_id in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            qpos_index = self.model.jnt_qposadr[joint_id]
            joint_name_to_qpos_index[joint_name] = qpos_index
        return joint_name_to_qpos_index

    def setup_joint_qvel_mapping(self):
        joint_name_to_qvel_index = {}
        for joint_id in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            qvel_index = self.model.jnt_dofadr[joint_id]
            joint_name_to_qvel_index[joint_name] = qvel_index
        return joint_name_to_qvel_index

    def get_contact_state(self):
        contact_num = self.data.ncon
        contact_geom_id_1 = list()
        contact_geom_id_2 = list()
        contact_geom_name_1 = list()
        contact_geom_name_2 = list()
        contact_pose = list()
        contact_force = list()
        contact_dist = list()

        contact_num_new = 0
        for i in range(contact_num):
            contact = self.data.contact[i]

            contact_geom1 = contact.geom1
            contact_geom2 = contact.geom2
            contact_geom1_name = str(self.get_geom_name_from_id(contact.geom1))
            contact_geom2_name = str(self.get_geom_name_from_id(contact.geom2))

            if (contact_geom1_name != "None") & (contact_geom2_name != "None"):
                contact_num_new += 1
                contact_dist.append(contact.dist)
                contact_geom_id_1.append(contact_geom1)
                contact_geom_id_2.append(contact_geom2)
                contact_geom_name_1.append(contact_geom1_name)
                contact_geom_name_2.append(contact_geom2_name)
                contact_frame = contact.frame.reshape(3, 3)
                contact_quat = self.from_matrix(contact_frame)

                contact_pose.append(contact.pos.tolist() + contact_quat.tolist())
                c_array = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, i, c_array)
                contact_force.append(c_array)

        return (
            contact_num_new,
            contact_geom_id_1,
            contact_geom_id_2,
            contact_geom_name_1,
            contact_geom_name_2,
            contact_pose,
            contact_force,
            contact_dist,
        )

    def get_sensor_data(self, sensor_name):
        """Look up sensor by exact name.

        Resolution order (name-based, not keyword-based):
          1. mjOBJ_SENSOR (gyro/accelerometer/force/torque/...) — returned as
             a flat array of length self.model.sensor_dim[sensor_id].
          2. mjOBJ_SITE (orientation only, 3x3 matrix) — returned as 9 floats
             read from data.site_xmat.

        Previously this method branched on `'imu' in sensor_name` and used
        the SITE path. That broke the new M360S LiDAR+IMU sensors
        (lidar_imu_gyro / lidar_imu_acc), which are real SENSOR objects
        bound to lidar_site, not SITE objects. The new logic is
        name-exact and works for both.
        """
        sensor_data = None
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sensor_id >= 0:
            adr = self.model.sensor_adr[sensor_id]
            dim = self.model.sensor_dim[sensor_id]
            sensor_data = self.data.sensordata[adr : adr + dim]
        else:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, sensor_name)
            if site_id >= 0:
                sensor_data = self.data.site_xmat[site_id]
        return sensor_data

    def get_geom_names_by_group(self, group):
        names = []
        groups = self.model.geom_group
        for index in range(self.model.ngeom):
            if groups[index] == group:
                names.append(self.get_geom_name_from_id(index))
        return names

    def get_geom_name_from_id(self, id):
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, id)

    def get_ft_sensor_data(self, robot_name):
        force_name = robot_name + "_force_sensor"
        force_data = self.get_sensor_data(force_name)

        torque_name = robot_name + "_torque_sensor"
        torque_data = self.get_sensor_data(torque_name)

        if force_data is None and torque_data is None:
            return None
        else:
            return np.concatenate([force_data, torque_data])

    def get_reset_status(self):
        press_status = False

        if getattr(self, "_use_passive_viewer", False):
            if getattr(self, "_passive_backspace_pressed", False):
                self._passive_backspace_pressed = False
                if time.time() - self.reset_time > 1:
                    press_status = True
            return press_status

        if self.mujoco_renderer is not None and hasattr(self.mujoco_renderer.viewer, "window"):
            if glfw.get_key(self.mujoco_renderer.viewer.window, glfw.KEY_BACKSPACE) == glfw.PRESS:
                if time.time() - self.reset_time > 1:
                    press_status = True

        return press_status

    def get_mesh_vertices(self, name):
        mesh_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MESH, name)
        total_vertnum_before = sum(self.model.mesh_vertnum[:mesh_id])
        cur_vertnum = self.model.mesh_vertnum[mesh_id]
        return self.model.mesh_vert[total_vertnum_before : total_vertnum_before + cur_vertnum]

    def get_body_names(self):
        body_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(self.model.nbody)
        ]
        return body_names

    def has_body(self, name):
        return name in self.get_body_names()

    def get_controllable_joints(self):
        trntype = self.model.actuator_trntype
        trnid = self.model.actuator_trnid

        mvbl_jnts = []
        for i in range(trnid.shape[0]):
            if trntype[i] == 0 and trnid[i, 0] not in mvbl_jnts:
                mvbl_jnts.append(trnid[i, 0])

        return sorted(mvbl_jnts)

    def get_site_pose(self, name):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        pos = self.data.site_xpos[site_id]
        mat = self.data.site_xmat[site_id].reshape(3, 3)
        quat = self.from_matrix(mat)

        return np.hstack((pos, quat))

    def get_chassis_pose(self):
        if "chassis" in self.site_names:
            chassis_pose = self.get_site_pose("chassis")
            return chassis_pose
        else:
            return [0, 0, 0, 1, 0, 0, 0]

    def get_joint_id(self, joint_name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

    def get_actuator_id(self, actuator_name):
        id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        return id

    def get_actuator_names(self):
        actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.njnt)
        ]
        return actuator_names

    _MECANUM_VIRTUAL_CHASSIS = ("astribot_chassis_x", "astribot_chassis_y", "astribot_chassis_zrot")

    def _mecanum_virtual_chassis_pos(self, joint_name):
        ip = self.mecanum_adapter._int_pose
        if joint_name == "astribot_chassis_x":
            return ip[0]
        if joint_name == "astribot_chassis_y":
            return ip[1]
        return ip[2]  # zrot → yaw（∫wz）

    def _mecanum_virtual_chassis_vel(self, joint_name):
        import numpy as np

        a = self.mecanum_adapter._base_dofadr
        qa = self.mecanum_adapter._base_qadr
        if joint_name in ("astribot_chassis_x", "astribot_chassis_y"):
            vx_w = float(self.data.qvel[a])
            vy_w = float(self.data.qvel[a + 1])
            yaw = self.mecanum_adapter._yaw_from_quat(
                self.data.qpos[qa + 3],
                self.data.qpos[qa + 4],
                self.data.qpos[qa + 5],
                self.data.qpos[qa + 6],
            )
            c, s = np.cos(-yaw), np.sin(-yaw)
            vx_b = c * vx_w - s * vy_w
            vy_b = s * vx_w + c * vy_w
            return vx_b if joint_name == "astribot_chassis_x" else vy_b
        return float(self.data.qvel[a + 5])

    def _freeze_mecanum_base(self):
        a = self.mecanum_adapter._base_qadr
        da = self.mecanum_adapter._base_dofadr
        self.data.qpos[a] = self._mecanum_spawn_x
        self.data.qpos[a + 1] = self._mecanum_spawn_y
        for k in range(4):
            self.data.qpos[a + 3 + k] = self._mecanum_spawn_quat[k]
        self.data.qvel[da] = 0.0
        self.data.qvel[da + 1] = 0.0
        self.data.qvel[da + 5] = 0.0

    def get_joint_position(self, joint_name):
        if self.mecanum_adapter is not None and joint_name in self._MECANUM_VIRTUAL_CHASSIS:
            return self._mecanum_virtual_chassis_pos(joint_name)
        qpos_index = self.joint_name_to_qpos_index[joint_name]
        joint_id = self.get_joint_id(joint_name)
        if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            return self.data.qpos[qpos_index : qpos_index + 3]
        elif "gripper" in joint_name:
            # Report the gripper on the SDK's 0-100 scale, clamped to that range.
            #
            # Only the lower bound used to be clamped, and it was clamped by writing
            # back into self.data.qpos -- a state read silently mutating the physics.
            # With no upper clamp, a joint driven past its 0.93 rad limit was reported
            # as >100 (measured 121.2 at 1.1268 rad), which is outside the interface's
            # own range and feeds the SDK a bogus planning start point.
            #
            # Clamp the returned value instead of qpos, so reading state never changes
            # the simulation.
            return min(100.0, max(0.0, float(self.data.qpos[qpos_index]) / 0.93 * 100))
        else:
            return self.data.qpos[qpos_index]

    def get_joint_positions(self, names):
        pos = []
        for i in names:
            pos.append(self.get_joint_position(i))
        return pos

    def get_joint_velocity(self, joint_name):
        if self.mecanum_adapter is not None and joint_name in self._MECANUM_VIRTUAL_CHASSIS:
            return self._mecanum_virtual_chassis_vel(joint_name)
        qvel_index = self.joint_name_to_qvel_index[joint_name]
        joint_id = self.get_joint_id(joint_name)
        if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            return self.data.qvel[qvel_index : qvel_index + 3]
        elif "gripper" in joint_name:
            return self.data.qvel[qvel_index] / 0.93 * 100
        else:
            return self.data.qvel[qvel_index]

    def get_joint_velocities(self, names):
        vel = []
        for i in names:
            vel.append(self.get_joint_velocity(i))
        return vel

    def get_joint_acceleration(self, joint_name):
        if self.mecanum_adapter is not None and joint_name in self._MECANUM_VIRTUAL_CHASSIS:
            return 0.0
        qacc_index = self.joint_name_to_qvel_index[joint_name]
        joint_id = self.get_joint_id(joint_name)
        if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            return self.data.qacc_warmstart[qacc_index : qacc_index + 3]
        elif "gripper" in joint_name:
            return self.data.qacc_warmstart[qacc_index] / 0.93 * 100
        else:
            return self.data.qacc_warmstart[qacc_index]

    def get_joint_accelerations(self, names):
        acc = []
        for i in names:
            acc.append(self.get_joint_acceleration(i))
        return acc

    def get_joint_torque(self, joint_name):
        if self.mecanum_adapter is not None and joint_name in self._MECANUM_VIRTUAL_CHASSIS:
            return 0.0
        joint_id = self.get_joint_id(joint_name)
        return self.data.qfrc_actuator[self.model.jnt_dofadr[joint_id]]

    def get_joint_torques(self, names):
        tor = []
        for i in names:
            tor.append(self.get_joint_torque(i))
        return tor

    def get_joint_applied_torque(self, joint_name):
        qtor_index = self.joint_name_to_qpos_index[joint_name]
        joint_id = self.get_joint_id(joint_name)
        if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            # Return the position part of the free joint
            return self.data.qfrc_applied[qtor_index : qtor_index + 3]
        else:
            return self.data.qfrc_applied[qtor_index]

    def get_joint_applied_torques(self):
        return self.data.qfrc_applied

    def get_gravity_torque(self):
        """Deprecated: Use compute_dynamics_compensation() instead.

        This method only computes gravity g(q), not the full C(q,qd)*qd + g(q).
        Kept for backward compatibility with old code.
        """
        return self.compute_dynamics_compensation(
            include_gravity=True, include_coriolis=False
        )

    def compute_dynamics_compensation(
        self,
        joint_indices=None,
        include_gravity=True,
        include_coriolis=True,
        include_inertia=False,
    ):
        """Dynamics feed-forward via mj_inverse on a contact-free shadow model.

        Inverse dynamics gives qfrc_inverse = M(q)*qdd + C(q,qd)*qd + g(q). Every
        term is evaluated from the MEASURED feedback state (q, qd, qdd), the same way
        the real robot's middle layer builds its feed-forward -- no desired
        acceleration is involved. Terms are selected by zeroing their driver:

            include_coriolis=False -> qd := 0   (drops C(q,qd)*qd)
            include_inertia=False  -> qdd := 0  (drops M(q)*qdd)
            include_gravity=False  -> subtract a gravity-only evaluation

        Args:
            joint_indices: DOF indices to return (None = all)
            include_gravity: include g(q)
            include_coriolis: include C(q,qd)*qd, from the measured qd
            include_inertia: include M(q)*qdd, from the measured qdd. Off by default:
                qdd is a noisy second-order signal that couples with the controller
                output within the same step (the compensation changes the
                acceleration, which changes the next compensation). Measured on a
                mode-1 step input it contributes +/-4 Nm during the transient with a
                sign flip, and decays to ~0 in steady state.

        Returns:
            compensation: (nv,) array, or the joint_indices subset.
        """
        if not (include_gravity or include_coriolis or include_inertia):
            return np.zeros(self.model.nv)

        def _inverse(use_qvel, use_qacc):
            dc = self.data_for_cal
            # Copy the FULL qpos, not qpos[jnt_qposadr]. The latter only touches
            # each joint's first coordinate, so a free joint's remaining 6 entries
            # (position + quaternion) stay at their init values. The chassis then
            # sits at a stale height/orientation in the shadow model and mj_inverse
            # returns absurd support torques (measured 8761 Nm on a joint whose real
            # gravity torque is -1.96 Nm) as soon as the robot moves.
            dc.qpos[:] = self.data.qpos
            dc.qvel[:] = self.data.qvel if use_qvel else 0.0
            dc.qacc[:] = self.data.qacc if use_qacc else 0.0
            mujoco.mj_inverse(self.model_for_cal, dc)
            return dc.qfrc_inverse.copy()

        compensation = _inverse(include_coriolis, include_inertia)

        if not include_gravity:
            # No way to disable gravity per-call, so subtract a static evaluation:
            # (selected terms + g) - g.
            compensation -= _inverse(False, False)

        return compensation[joint_indices] if joint_indices else compensation

    def get_body_position(self, body_name):
        assert self.has_body(body_name), "Name mistaken"
        return self.data.body(body_name).xpos.copy()

    def get_body_pose(self, body_name):
        if not self.has_body(body_name):
            return np.zeros(7, dtype=np.float64)
        return np.hstack(
            (self.get_body_position(body_name), self.get_body_orientation_quat(body_name))
        )

    def get_body_orientation_quat(self, body_name):
        assert self.has_body(body_name), "Name mistaken"
        return self.data.body(body_name).xquat.copy()

    def get_near_and_far(self):
        extent = self.model.stat.extent
        near = self.model.vis.map.znear * extent
        far = self.model.vis.map.zfar * extent

        return near, far

    def get_camera_fovy(self, camera_name):
        camid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        return math.radians(self.model.cam_fovy[camid])

    def get_camera_transform(self, camera_name):
        camid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

        cam_body_id = self.model.cam_bodyid[camid]
        camera_position = self.model.body_pos[cam_body_id]

        temp_trans = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        rotation_matrix = np.matmul(np.array(self.model.cam_mat0[camid]).reshape(3, 3), temp_trans)

        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = camera_position

        return transform

    def set_joint_applied_torques(self, ids, values):
        for i in range(len(ids)):
            self.data.qfrc_applied[ids[i]] = values[ids[i]]

    def set_actuator_parameters(self, actuator_id, dynprm=None, gainprm=None, biasprm=None):
        if dynprm is not None:
            self.model.actuator_dynprm[actuator_id][: len(dynprm)] = dynprm
        if gainprm is not None:
            self.model.actuator_gainprm[actuator_id][: len(gainprm)] = gainprm
        if biasprm is not None:
            self.model.actuator_biasprm[actuator_id][: len(biasprm)] = biasprm

    def set_body_pose(self, name, pose=None, twist=None, delta_pos=None):
        id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

        qposadr = self.model.jnt_qposadr[self.model.body_jntadr[id]]
        qveladr = self.model.jnt_dofadr[self.model.body_jntadr[id]]

        try:
            if delta_pos is not None:
                self.data.qpos[qposadr : qposadr + 3] += delta_pos

            if pose is not None:
                self.data.qpos[qposadr : qposadr + 7] = pose
            if twist is not None:
                self.data.qvel[qveladr : qveladr + 6] = twist
        except Exception:
            pass

    def set_camera_pose(self, camera_name, pose):
        camid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self.model.cam_pos[camid] = pose[:3]
        self.model.cam_quat[camid] = pose[3:]

    def set_mocap_pose_with_id(self, id, pose):
        self.data.mocap_pos[id] = pose[:3]
        self.data.mocap_quat[id] = pose[3:]
