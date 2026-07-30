#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025, Astribot Co., Ltd.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------
# Author: Tony Wang, tonywang@astribot.com
# -----------------------------------------------------------------------------

"""
File: chassis_kinematics.py
Brief: chassis kinematics for astribot s0
"""

import numpy as np


class ChassisKinematics:
    """Differential two-wheel chassis: SDK (v, omega) <-> motor speeds/torques."""

    def __init__(self, wheel_radius, wheel_distance):
        self.r = wheel_radius
        self.d = wheel_distance

    def motor_to_cartesian(self, motor_data):
        # Assuming motor_data is [omega_m1, omega_m2]
        T = [[self.r / 2, self.r / 2], [-self.r / self.d, self.r / self.d]]
        v = T[0][0] * motor_data[0] + T[0][1] * motor_data[1]
        omega = T[1][0] * motor_data[0] + T[1][1] * motor_data[1]
        return [v, omega]

    def cartesian_to_motor(self, cartesian_data):
        # Assuming cartesian_data is [v, omega]
        T_inv = [[1 / self.r, -self.d / (2 * self.r)], [1 / self.r, self.d / (2 * self.r)]]
        omega_m1 = T_inv[0][0] * cartesian_data[0] + T_inv[0][1] * cartesian_data[1]
        omega_m2 = T_inv[1][0] * cartesian_data[0] + T_inv[1][1] * cartesian_data[1]
        return [omega_m1, omega_m2]

    def torque_motor_to_cartesian(self, motor_torque):
        T = [[1 / self.r, 1 / self.r], [-self.d / (2 * self.r), self.d / (2 * self.r)]]
        F = T[0][0] * motor_torque[0] + T[0][1] * motor_torque[1]
        tau = T[1][0] * motor_torque[0] + T[1][1] * motor_torque[1]
        return [F, tau]

    def torque_cartesian_to_motor(self, cartesian_torque):
        T_inv = [[self.r, self.r], [-self.d / 2, self.d / 2]]
        tau_m1 = T_inv[0][0] * cartesian_torque[0] + T_inv[0][1] * cartesian_torque[1]
        tau_m2 = T_inv[1][0] * cartesian_torque[0] + T_inv[1][1] * cartesian_torque[1]
        return [tau_m1, tau_m2]


class OmniWheelKinematics:
    """Legacy 4-wheel omni kinematics kept for backward compatibility.

    New code should use OmniChassisKinematics, which derives the inverse matrix from
    the actual wheel geometry.
    """

    def __init__(self, wheel_radius, wheel_distance):
        self.r = wheel_radius
        self.d = wheel_distance

    def motor_to_cartesian(self, motor_data):
        T = np.array(
            [
                [np.cos(np.pi / 4), np.sin(np.pi / 4), self.d / 2],
                [-np.cos(np.pi / 4), np.sin(np.pi / 4), self.d / 2],
                [-np.cos(np.pi / 4), -np.sin(np.pi / 4), self.d / 2],
                [np.cos(np.pi / 4), -np.sin(np.pi / 4), self.d / 2],
            ]
        )

        cartesian_data = np.dot(T, motor_data)
        return cartesian_data

    def cartesian_to_motor(self, cartesian_data):
        T_inv = np.array(
            [
                [np.cos(np.pi / 4), -np.cos(np.pi / 4), -np.cos(np.pi / 4), np.cos(np.pi / 4)],
                [np.sin(np.pi / 4), np.sin(np.pi / 4), -np.sin(np.pi / 4), -np.sin(np.pi / 4)],
                [self.d / 2, self.d / 2, self.d / 2, self.d / 2],
            ]
        )

        motor_data = np.dot(np.linalg.pinv(T_inv), cartesian_data)
        return motor_data

    def torque_motor_to_cartesian(self, motor_torque):
        if len(motor_torque) != 4:
            raise ValueError(
                "motor_torque must be a vector of length 4 representing the torques of the 4 wheels"
            )

        T = np.array(
            [
                [1 / self.r, 1 / self.r, 1 / self.r, 1 / self.r],
                [
                    -self.d / (2 * self.r),
                    self.d / (2 * self.r),
                    -self.d / (2 * self.r),
                    self.d / (2 * self.r),
                ],
            ]
        )

        cartesian_torque = np.dot(T, motor_torque)
        return cartesian_torque

    def torque_cartesian_to_motor(self, cartesian_torque):
        T_inv = np.array(
            [[self.r, self.r, self.r, self.r], [-self.d / 2, self.d / 2, -self.d / 2, self.d / 2]]
        )

        motor_torque = np.dot(np.linalg.pinv(T_inv), cartesian_torque)
        return motor_torque


class OmniChassisKinematics:
    """Omni-wheel chassis kinematics: body twist [vx, vy, wz] -> wheel speeds.

    These are omni wheels with +/-45 degree spin axes, NOT mecanum wheels, so the
    standard mecanum formula (uniformly signed vx column) does not apply: with this
    layout, equal speed on all wheels produces pure yaw.

    Inverse matrix A, ordered RF/LF/RR/LR, where Kt = cos45/r and Kr = Kt*(L+W):
        A = [[ +Kt, +Kt, +Kr],     # RF
             [ -Kt, +Kt, +Kr],     # LF
             [ +Kt, -Kt, +Kr],     # RR
             [ -Kt, -Kt, +Kr]]     # LR
    The sign structure is what matters: vx column [+,-,+,-], vy column [+,+,-,-],
    wz column uniformly signed.
    """

    def __init__(self, wheel_radius, half_wheelbase, half_track, _A=None):
        self.r = wheel_radius
        self.L = half_wheelbase
        self.W = half_track
        if _A is not None:
            self._A = np.asarray(_A, dtype=float)
            self._Kt = self._Kr = None
            return
        self._Kt = np.cos(np.pi / 4) / wheel_radius
        self._Kr = self._Kt * (self.L + self.W)
        self._A = np.array(
            [
                [+self._Kt, +self._Kt, +self._Kr],  # RF
                [-self._Kt, +self._Kt, +self._Kr],  # LF
                [+self._Kt, -self._Kt, +self._Kr],  # RR
                [-self._Kt, -self._Kt, +self._Kr],  # LR
            ]
        )

    @classmethod
    def from_wheel_geometry(cls, wheels, wheel_radius):
        """Derive the inverse matrix from wheel geometry (generic N wheels).

        Under a body twist, wheel i's contact-point velocity is the rigid-body field
        [vx - wz*yi, vy + wz*xi]. An omni wheel only drives the component along its
        drive direction d (the perpendicular component slides on the rollers):
            omega_i = (1/r) * [ dx, dy, (xi*dy - yi*dx) ] . [vx, vy, wz]
        where d is the spin axis rotated +90 degrees, i.e. d = [-say, sax].

        Args:
            wheels: list of (px, py, spin_axis_x, spin_axis_y) — wheel position in the
                body frame (m) plus the horizontal spin-axis unit vector, ordered to
                match the wheel-name list.
            wheel_radius: wheel radius in m.

        Returns:
            An OmniChassisKinematics instance whose _A is the derived N x 3 matrix.
        """
        r = wheel_radius
        A = []
        for px, py, sax, say in wheels:
            dx, dy = -say, sax
            A.append([dx / r, dy / r, (px * dy - py * dx) / r])
        xs = [w[0] for w in wheels]
        ys = [w[1] for w in wheels]
        L = (max(xs) - min(xs)) / 2.0 if xs else 0.0
        W = (max(ys) - min(ys)) / 2.0 if ys else 0.0
        return cls(wheel_radius, L, W, _A=np.array(A))

    def cartesian_to_motor(self, cartesian_data):
        """[vx, vy, wz] -> N wheel angular velocities (rad/s), in wheel order."""
        return list(self._A @ np.asarray(cartesian_data, dtype=float))

    def motor_to_cartesian(self, motor_data):
        """Wheel angular velocities -> [vx, vy, wz] via least-squares pseudo-inverse.

        The inverse map is over-determined (3 -> N), so the forward map uses a
        pseudo-inverse; it round-trips exactly for wheel speeds the inverse produced.
        """
        twist = np.dot(np.linalg.pinv(self._A), np.asarray(motor_data, dtype=float))
        return list(twist)


class OmniChassisController:
    """Base class for the in-sim chassis controller stub.

    Converts an external 3-DOF (x, y, yaw) pose or twist command into 4 wheel speeds.
    It holds all backend-independent control logic (pose error -> deadband ->
    feedforward -> PD -> sign convention -> omni IK); subclasses supply three hooks
    for reading state and writing commands:
        _read_base_pose()      -> [x, y, yaw]
        _read_base_world_vel() -> [vx_w, vy_w, wz]
        _write_wheel_speeds([w_RF, w_LF, w_RR, w_LR])
    """

    _MOTOR_ORDER = ["RF", "LF", "RR", "LR"]

    def __init__(
        self,
        wheel_radius,
        half_wheelbase,
        half_track,
        vel_scale=10.0,
        max_wheel_speed=None,
        kd=0.0,
        pos_deadband=0.0,
        yaw_deadband=0.0,
        vel_ff_scale=1.0,
    ):
        self.kin = OmniChassisKinematics(wheel_radius, half_wheelbase, half_track)
        self.vel_scale = vel_scale
        self.max_wheel_speed = max_wheel_speed
        self.kd = kd
        self.pos_deadband = pos_deadband  # m
        self.yaw_deadband = yaw_deadband  # rad
        self.vel_ff_scale = vel_ff_scale
        self._last_target = None

        self._int_pose = [0.0, 0.0, 0.0]
        self._int_inited = False

    def _read_base_pose(self):
        """Backend hook: return the chassis pose [x, y, yaw] in the world frame."""
        raise NotImplementedError

    def _read_base_world_vel(self):
        """Backend hook: return world [vx, vy, wz] (wz feeds the D term)."""
        raise NotImplementedError

    def _write_wheel_speeds(self, wheel_speeds):
        """Backend hook: write the 4 wheel-speed commands [RF, LF, RR, LR]."""
        raise NotImplementedError

    def reset_feedforward(self):
        """Clear the feedforward history and integrated pose (called on env.reset) so a
        stale target or integral cannot cause a jump."""
        self._last_target = None
        self._int_pose = [0.0, 0.0, 0.0]
        self._int_inited = False
        self._last_wheel_speeds = None

    def integrate_state(self, dt):
        """Advance the integrated pose by the body velocity: pos += vel_body * dt.

        This mirrors the real chassis state, which integrates the body velocity
        recovered from the wheel encoders (x/y are NOT rotated by yaw; yaw is the
        integral of wz). The env calls this once per step so the chassis position
        reported to the SDK carries the same semantics as the real robot.
        """
        if dt is None or dt <= 0.0:
            return
        vx_b, vy_b, wz = self.read_chassis_twist()
        if not self._int_inited:
            self._int_pose[2] = float(self._read_base_pose()[2])
            self._int_inited = True
        self._int_pose[0] += vx_b * dt
        self._int_pose[1] += vy_b * dt
        self._int_pose[2] += wz * dt

    def read_chassis_pose(self):
        """Chassis [x, y, yaw] in body integration space, as reported to the SDK."""
        return list(self._int_pose)

    def read_chassis_twist(self):
        """Current chassis body-frame velocity [vx, vy, wz], used for feedback."""
        import numpy as np

        _, _, cur_yaw = self._read_base_pose()
        vwx, vwy, wz = self._read_base_world_vel()
        c, s = np.cos(-cur_yaw), np.sin(-cur_yaw)
        return [float(c * vwx - s * vwy), float(s * vwx + c * vwy), float(wz)]

    @staticmethod
    def _yaw_from_quat(qw, qx, qy, qz):
        """Extract the yaw angle (rad) from a quaternion."""
        import numpy as np

        return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

    def apply_twist(self, twist):
        """Translate a body-frame twist (vx, vy, wz) into wheel speeds and write them.

        Saturated at max_wheel_speed when configured. twist must be 3-dimensional, so
        passing 4 wheel speeds by mistake is rejected rather than silently misread.
        """
        if len(twist) != 3:
            raise ValueError(f"apply_twist expects 3 dims (vx,vy,wz), got {len(twist)}")
        wheel_speeds = self.kin.cartesian_to_motor(list(twist))
        if self.max_wheel_speed is not None:
            wheel_speeds = [
                max(-self.max_wheel_speed, min(self.max_wheel_speed, w)) for w in wheel_speeds
            ]
        self._write_wheel_speeds(wheel_speeds)

    def apply_pose_command(self, target_pose, dt=None):
        """Pose error -> body velocity -> omni IK -> wheel speeds (feedforward + PD).

        The loop runs entirely in body integration space, matching the real robot: the
        state is the direct integral of the body velocity recovered from wheel speeds
        (see integrate_state), and the SDK's pos_cmd lives in the same space. So
        "current" is _int_pose and the error is a plain subtraction, with NO rotation
        from world into body -- rotating would put target and current in different
        spaces, making the error direction shift with yaw and causing yaw oscillation.

        The yaw error is likewise a plain subtraction with NO arctan2 wrapping: chassis
        yaw is an unbounded odometry accumulator, not a value cycling within +/-pi.
        Wrapping breaks when the commanded speed exceeds wheel saturation, because the
        lag accumulates and, once it passes pi, arctan2 flips the error sign and the
        chassis oscillates backwards.

        Control law: v = v_ff + (-Kp * error - Kd * body_vel). All three terms are
        negated because this IK's convention is +cmd -> -motion in the world frame.

        Args:
            target_pose: [x, y, yaw] target in body integration space.
            dt: sim-time step, used to difference consecutive targets for feedforward.
        """
        import numpy as np

        if len(target_pose) != 3:
            raise ValueError(f"apply_pose_command expects 3 dims [x,y,yaw], got {len(target_pose)}")

        cur_x, cur_y, cur_yaw = self._int_pose
        tx, ty, tyaw = target_pose

        ex_b = tx - cur_x
        ey_b = ty - cur_y
        eyaw = tyaw - cur_yaw

        # Deadband: an error within threshold counts as arrived.
        if np.hypot(ex_b, ey_b) < self.pos_deadband:
            ex_b = ey_b = 0.0
        if abs(eyaw) < self.yaw_deadband:
            eyaw = 0.0

        # Velocity feedforward from differencing consecutive targets; removes the
        # steady-state lag of pure P control. Zero on the first tick or invalid dt.
        vff_x_b = vff_y_b = vff_yaw = 0.0
        if (
            self.vel_ff_scale != 0.0
            and dt is not None
            and dt > 0.0
            and self._last_target is not None
        ):
            lx, ly, lyaw = self._last_target
            vff_x_b = (tx - lx) / dt
            vff_y_b = (ty - ly) / dt
            vff_yaw = (tyaw - lyaw) / dt
        self._last_target = [float(tx), float(ty), float(tyaw)]

        vx_b = vy_b = wz_b = 0.0
        if self.kd != 0.0:
            vx_b, vy_b, wz_b = self.read_chassis_twist()

        vx = -self.vel_ff_scale * vff_x_b - self.vel_scale * ex_b - self.kd * vx_b
        vy = -self.vel_ff_scale * vff_y_b - self.vel_scale * ey_b - self.kd * vy_b
        wz = -self.vel_ff_scale * vff_yaw - self.vel_scale * eyaw - self.kd * wz_b
        self.apply_twist([vx, vy, wz])


class MujocoOmniChassis(OmniChassisController):
    """MuJoCo backend: reads the freejoint qpos/qvel and writes 4 velocity actuators.

    Actuator and joint indices are resolved by name rather than hardcoded, so a change
    in MJCF include order cannot silently shift them.
    """

    _MOTOR_NAMES = [
        "astribot_chassis_motor_RF",
        "astribot_chassis_motor_LF",
        "astribot_chassis_motor_RR",
        "astribot_chassis_motor_LR",
    ]

    def __init__(
        self,
        model,
        wheel_radius,
        half_wheelbase,
        half_track,
        base_joint_name="chassis_free",
        vel_scale=10.0,
        max_wheel_speed=None,
        kd=0.0,
        pos_deadband=0.0,
        yaw_deadband=0.0,
        vel_ff_scale=1.0,
        motor_names=None,
        wheel_joints=None,
    ):
        import mujoco

        super().__init__(
            wheel_radius,
            half_wheelbase,
            half_track,
            vel_scale=vel_scale,
            max_wheel_speed=max_wheel_speed,
            kd=kd,
            pos_deadband=pos_deadband,
            yaw_deadband=yaw_deadband,
            vel_ff_scale=vel_ff_scale,
        )
        names = motor_names or self._MOTOR_NAMES
        self.actuator_ids = []
        for name in names:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise ValueError(
                    f"{type(self).__name__}: MJCF missing actuator {name} (motor_names={names})"
                )
            self.actuator_ids.append(aid)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, base_joint_name)
        if jid < 0:
            raise ValueError(f"{type(self).__name__}: MJCF missing freejoint '{base_joint_name}'")
        self._base_qadr = model.jnt_qposadr[jid]
        self._base_dofadr = model.jnt_dofadr[jid]
        self._data = None

        # OmniChassisKinematics.from_wheel_geometry。
        if wheel_joints is not None and len(names) != 4:
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            wheels = []
            for jn in wheel_joints:
                wjid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if wjid < 0:
                    raise ValueError(f"{type(self).__name__}: MJCF missing wheel joint {jn}")
                bid = model.jnt_bodyid[wjid]
                pos = model.body_pos[bid]
                spin = data.xmat[bid].reshape(3, 3)[:, 2]
                wheels.append((float(pos[0]), float(pos[1]), float(spin[0]), float(spin[1])))
            self.kin = OmniChassisKinematics.from_wheel_geometry(wheels, wheel_radius)

    def apply(self, data, twist):
        """Bind the current MjData, then write a body twist as wheel speeds."""
        self._data = data
        self.apply_twist(twist)

    def apply_pose_command(self, data, target_pose, dt=None):
        """Bind the current MjData, then run the pose-command controller."""
        self._data = data
        super().apply_pose_command(target_pose, dt=dt)

    def _read_base_pose(self):
        """Read [x, y, yaw] from the freejoint qpos."""
        d, a = self._data, self._base_qadr
        yaw = self._yaw_from_quat(d.qpos[a + 3], d.qpos[a + 4], d.qpos[a + 5], d.qpos[a + 6])
        return [float(d.qpos[a]), float(d.qpos[a + 1]), float(yaw)]

    def _read_base_world_vel(self):
        """Read world [vx, vy, wz] from the freejoint qvel."""
        d, da = self._data, self._base_dofadr
        return [float(d.qvel[da]), float(d.qvel[da + 1]), float(d.qvel[da + 5])]

    def _write_wheel_speeds(self, wheel_speeds):
        """Write each wheel speed to its velocity actuator's ctrl slot."""
        for aid, w in zip(self.actuator_ids, wheel_speeds):
            self._data.ctrl[aid] = w


class GenesisOmniChassis(OmniChassisController):
    """Genesis backend: reads the base link pose/velocity and writes 4 wheel joints.

    Genesis has no actuator concept, so velocity targets go straight to the wheel joint
    dofs via control_dofs_velocity. Wheel joint names default to the MJCF ones
    (wheel_RF/LF/RR/LR_Joint), and the base link pose is read from the robot base link.
    """

    _WHEEL_JOINTS = ["wheel_RF_Joint", "wheel_LF_Joint", "wheel_RR_Joint", "wheel_LR_Joint"]

    def __init__(
        self,
        robot,
        wheel_radius,
        half_wheelbase,
        half_track,
        base_link_name=None,
        wheel_joint_names=None,
        vel_scale=10.0,
        max_wheel_speed=None,
        kd=0.0,
        pos_deadband=0.0,
        yaw_deadband=0.0,
        vel_ff_scale=1.0,
    ):
        super().__init__(
            wheel_radius,
            half_wheelbase,
            half_track,
            vel_scale=vel_scale,
            max_wheel_speed=max_wheel_speed,
            kd=kd,
            pos_deadband=pos_deadband,
            yaw_deadband=yaw_deadband,
            vel_ff_scale=vel_ff_scale,
        )
        self.robot = robot
        self._last_wheel_speeds = None
        names = wheel_joint_names or self._WHEEL_JOINTS
        self.wheel_dof_index = []
        for name in names:
            joint = robot.get_joint(name)
            self.wheel_dof_index.append(joint.dof_idx_local)
        if base_link_name is not None:
            self._base_link = robot.get_link(base_link_name)
        else:
            try:
                self._base_link = robot.get_link("chassis_base")
            except Exception:
                raise ValueError(
                    "GenesisOmniChassis: base_link_name not specified and the model has no "
                    "'chassis_base'. "
                    "Set config omni.base_link_name to a body that moves with the "
                    "chassis (not the world anchor)."
                )

    def _base_state(self):
        """Fetch the base link pose and velocity from Genesis (one GPU->CPU sync)."""
        import numpy as np

        link = self._base_link
        pos = np.asarray(link.get_pos().cpu()).reshape(-1)
        quat = np.asarray(link.get_quat().cpu()).reshape(-1)  # Genesis: (w,x,y,z)
        vel = np.asarray(link.get_vel().cpu()).reshape(-1)
        ang = np.asarray(link.get_ang().cpu()).reshape(-1)
        return pos, quat, vel, ang

    def _read_base_pose(self):
        """Read [x, y, yaw] from the base link pose."""
        pos, quat, _, _ = self._base_state()
        yaw = self._yaw_from_quat(quat[0], quat[1], quat[2], quat[3])
        return [float(pos[0]), float(pos[1]), float(yaw)]

    def _read_base_world_vel(self):
        """Read world [vx, vy, wz] from the base link velocities."""
        _, _, vel, ang = self._base_state()
        return [float(vel[0]), float(vel[1]), float(ang[2])]

    def _write_wheel_speeds(self, wheel_speeds):
        """Send wheel velocity targets to the wheel dofs, caching them for reissue."""
        self._last_wheel_speeds = list(wheel_speeds)
        self.robot.control_dofs_velocity(self._last_wheel_speeds, self.wheel_dof_index)

    def reissue_wheel_speeds(self):
        """Re-send the cached wheel speeds.

        Genesis needs the velocity target reissued every physics step, whereas the
        controller only recomputes it once per control tick.
        """
        speeds = getattr(self, "_last_wheel_speeds", None)
        if speeds is not None:
            self.robot.control_dofs_velocity(speeds, self.wheel_dof_index)

    def apply(self, twist):
        """Write a body twist as wheel speeds."""
        self.apply_twist(twist)


# ---------------------------------------------------------------------------
# Backward-compatible aliases. The historical "Mecanum" naming predates the
# discovery that this chassis actually uses +/-45 degree omni wheels.
# ---------------------------------------------------------------------------
MecanumKinematics = OmniChassisKinematics
MecanumChassisController = OmniChassisController
MecanumChassisAdapter = MujocoOmniChassis
MujocoMecanumChassis = MujocoOmniChassis
GenesisMecanumChassis = GenesisOmniChassis
