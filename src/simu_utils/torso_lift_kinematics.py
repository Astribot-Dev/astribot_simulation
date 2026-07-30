"""Torso lift 1-DOF <-> 2-joint linear mapping.

On robots with a segmented torso lift, the lift is built from **two physical
translational joints** in series (``astribot_torso_joint_1`` +
``astribot_torso_joint_2``, both Z-axis slides of 0-0.23 m each). The real robot
and the SDK treat the torso lift as a **single degree of freedom** L in
[0, 0.46] for both commands and states, while the simulation must drive the two
real joints.

This module provides a purely linear mapping (no frames, no integration — unlike
the omni-wheel chassis IK):

* Commands (SDK 1-DOF -> 2 joints): the joints receive L*ratio and L*(1-ratio).
  The default even split ``ratio=0.5`` gives each joint L/2, so their local
  velocities are equal (the agreed equal-rate coupling).
* States (2 joints -> SDK 1-DOF): position / velocity / acceleration / torque are
  **summed** (position = joint_1 + joint_2 in [0, 0.46], velocity likewise, which
  is self-consistent with velocity = d(position)/dt).

The mapping is backend-independent, so it lives in the ROS interface layer (the
torso component of ``robot_ros_interface``); the env and both the MuJoCo and
Genesis backends still see 2 real joints, unchanged. Compare ``ChassisKinematics``
(SDK <-> motor space transform for the wheeled base) as the analogous precedent.
"""

from typing import List


class TorsoLiftKinematics:
    """Torso lift single-DOF <-> dual-joint linear mapping.

    Args:
        ratio: fraction of the command given to the first joint (default 0.5, an
            even split). The second joint receives ``1 - ratio``. Equal-rate
            coupling requires equal joint commands, i.e. ratio=0.5.
    """

    def __init__(self, ratio: float = 0.5):
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"torso lift ratio must be in [0, 1], got {ratio}")
        self.ratio = float(ratio)

    def single_to_joints(self, single: List[float]) -> List[float]:
        """SDK single-DOF command -> 2-joint commands (split L*ratio / L*(1-ratio)).

        Takes a list of length 1 (the SDK side has dof=1) and returns a list of
        length 2, ordered to match the torso section of the config's
        ``joint_names_list``: [torso_joint_1, torso_joint_2]. An empty or all-zero
        command splits unchanged (0 -> [0, 0]).
        """
        if len(single) != 1:
            raise ValueError(f"torso lift single-DOF command must have length 1, got {len(single)}")
        value = single[0]
        return [value * self.ratio, value * (1.0 - self.ratio)]

    def joints_to_single(self, joints: List[float]) -> List[float]:
        """2-joint state -> SDK single-DOF state (summed).

        position = joint_1 + joint_2 (in [0, 0.46]); velocity / accel / torque are
        summed the same way (under equal-rate coupling velocity = 2x a single
        joint, self-consistent with d(sum)/dt). Returns a list of length 1.
        """
        if len(joints) != 2:
            raise ValueError(f"torso lift dual-joint state must have length 2, got {len(joints)}")
        return [joints[0] + joints[1]]
