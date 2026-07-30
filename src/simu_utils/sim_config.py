#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ControllerGains:
    """PD gains for a single control mode."""

    kp: float
    kd: float


@dataclass
class DefaultGains:
    """Default PD gains for three control modes."""

    position: ControllerGains = field(default_factory=lambda: ControllerGains(100.0, 10.0))
    velocity: ControllerGains = field(default_factory=lambda: ControllerGains(50.0, 5.0))
    torque: ControllerGains = field(default_factory=lambda: ControllerGains(1.0, 0.1))


@dataclass
class DynamicsCompensation:
    """Dynamics feedforward compensation configuration."""

    enabled: bool = True
    modes: List[int] = field(default_factory=lambda: [1, 2, 3])
    components: Dict[str, bool] = field(
        default_factory=lambda: {
            "gravity": True,
            "coriolis": True,
            "inertia": False,
            "friction": False,
        }
    )


@dataclass
class ControllerConfig:
    """Motor three-loop controller configuration."""

    default_gains: DefaultGains = field(default_factory=DefaultGains)
    dynamics_compensation: DynamicsCompensation = field(default_factory=DynamicsCompensation)
    per_component: Dict[str, Dict] = field(default_factory=dict)
    per_joint: Dict[str, Dict] = field(default_factory=dict)

    def get_gains(self, joint_name: str, component_name: str, mode: int) -> ControllerGains:
        """Query gains for a joint in a given mode (priority: per_joint > per_component > default).

        Args:
            joint_name: Joint name (e.g., "astribot_arm_left_joint_1")
            component_name: Component name (e.g., "astribot_arm_left")
            mode: Control mode (1=position, 2=velocity, 3=torque)

        Returns:
            ControllerGains with kp and kd
        """
        mode_name = {1: "position", 2: "velocity", 3: "torque"}.get(mode)
        if mode_name is None:
            raise ValueError(f"Invalid control mode: {mode}")

        # Priority 1: per_joint
        if joint_name in self.per_joint:
            if mode_name in self.per_joint[joint_name]:
                cfg = self.per_joint[joint_name][mode_name]
                return ControllerGains(cfg["kp"], cfg["kd"])

        # Priority 2: per_component
        if component_name in self.per_component:
            comp_cfg = self.per_component[component_name]
            if "default_gains" in comp_cfg and mode_name in comp_cfg["default_gains"]:
                cfg = comp_cfg["default_gains"][mode_name]
                return ControllerGains(cfg["kp"], cfg["kd"])

        # Priority 3: default_gains
        return getattr(self.default_gains, mode_name)

    def should_compensate(self, component_name: str, mode: int) -> bool:
        """Check if dynamics compensation is enabled for a component in a given mode.

        Args:
            component_name: Component name (e.g., "astribot_arm_left")
            mode: Control mode (1=position, 2=velocity, 3=torque)

        Returns:
            True if compensation should be applied
        """
        # Priority 1: per_component override
        if component_name in self.per_component:
            if "dynamics_compensation" in self.per_component[component_name]:
                comp_cfg = self.per_component[component_name]["dynamics_compensation"]
                enabled = comp_cfg.get("enabled", self.dynamics_compensation.enabled)
                modes = comp_cfg.get("modes", self.dynamics_compensation.modes)
                return enabled and mode in modes

        # Priority 2: global default
        return self.dynamics_compensation.enabled and mode in self.dynamics_compensation.modes


@dataclass
class PhysicsConfig:
    """Physics engine timing and gravity."""

    dt: float = 0.01

    frame_skip: int = 2

    gravity: List[float] = field(default_factory=lambda: [0.0, 0.0, -10.0])


@dataclass
class SensorFrequencies:
    """Requested sensor sampling frequencies in Hz."""

    imu: float = 200.0

    chassis_imu: float = 200.0

    lidar: float = 10.0

    force_torque: float = 50.0


@dataclass
class CameraExtrinsic:
    """Camera extrinsics: parent link, local pose and FOV."""

    parent_link: str

    local_pos: tuple[float, float, float]

    local_quat: tuple[float, float, float, float]

    fovy: float


@dataclass
class DisplayConfig:
    """Rendering and display settings."""

    mode: str = "human"

    width: int = 1920

    height: int = 1080

    fps: int = 60

    camera_resolutions: dict[str, tuple[int, int]] = None

    camera_extrinsics: dict[str, CameraExtrinsic] = None

    def __post_init__(self):
        if self.camera_resolutions is not None:
            self.camera_resolutions = {
                k: tuple(v) if isinstance(v, list) else v
                for k, v in self.camera_resolutions.items()
            }

        if self.camera_extrinsics is not None:
            converted = {}
            for name, config in self.camera_extrinsics.items():
                if isinstance(config, dict):
                    converted[name] = CameraExtrinsic(
                        parent_link=config["parent_link"],
                        local_pos=tuple(config["local_pos"]),
                        local_quat=tuple(config["local_quat"]),
                        fovy=float(config["fovy"]),
                    )
                else:
                    converted[name] = config
            self.camera_extrinsics = converted


@dataclass
class ROSConfig:
    """ROS communication settings (domain id, QoS)."""

    domain_id: int = 25

    qos_depth: int = 100

    qos_deadline_sec: float = 0.5


@dataclass
class LiDARConfig:
    """LiDAR sensor settings."""

    enabled: bool = False

    site_name: str = "lidar_site"

    cutoff_dist: float = 40.0

    min_dist: float = 0.1

    samples: int = 24000

    scan_pattern: str = "mid360"

    frequency: float = 10.0

    n_points: Optional[List[int]] = None


@dataclass
class SimConfig:
    """Full simulation configuration schema."""

    robot_type: str = ""

    robot_name: str = "astribot_whole_body"

    scene: str = "floor"

    model_path: Optional[str] = None

    simulator_type: str = "Mujoco"

    physics: PhysicsConfig = field(default_factory=PhysicsConfig)

    sensor_frequencies: SensorFrequencies = field(default_factory=SensorFrequencies)

    display: DisplayConfig = field(default_factory=DisplayConfig)

    ros: ROSConfig = field(default_factory=ROSConfig)

    lidar: LiDARConfig = field(default_factory=LiDARConfig)

    robot_list: List[str] = field(default_factory=list)

    joint_names_list: List[List[str]] = field(default_factory=list)

    vel_compensation_list: List[bool] = field(default_factory=list)

    sensor_names: List[str] = field(default_factory=list)

    camera_names: List[str] = field(default_factory=list)

    object_names: List[str] = field(default_factory=list)

    chassis_fixed: bool = False

    controller_config: ControllerConfig = field(default_factory=ControllerConfig)

    imu_topic: str = "/livox/imu_front"

    lid_topic: str = "/livox/lidar_front"

    chassis_imu_topic: str = "/astribot_whole_body/chassis_imu"

    _disabled_sensors: List[str] = field(default_factory=list)

    update_trajectory_map: Dict = field(default_factory=dict)

    def validate(self):
        """Raises:"""
        errors = []

        if not self.robot_type:
            errors.append("robot_type is required")

        if not (0.001 <= self.physics.dt <= 0.02):
            errors.append(f"physics.dt={self.physics.dt} outside valid range [0.001, 0.02]")

        if not (1 <= self.physics.frame_skip <= 10):
            errors.append(
                f"physics.frame_skip={self.physics.frame_skip} outside valid range [1, 10]"
            )

        for sensor_name in ["imu", "chassis_imu", "lidar", "force_torque"]:
            freq = getattr(self.sensor_frequencies, sensor_name)
            if freq < 0:
                errors.append(f"sensor_frequencies.{sensor_name}={freq}Hz cannot be negative")
            if freq > 1000.0:
                errors.append(
                    f"sensor_frequencies.{sensor_name}={freq}Hz is implausibly high "
                    f"(over 1000Hz); this is likely a config error"
                )

        if self.robot_list:
            if len(self.joint_names_list) != len(self.robot_list):
                errors.append(
                    f"joint_names_list length {len(self.joint_names_list)} "
                    f"!= robot_list length {len(self.robot_list)}"
                )
            if len(self.vel_compensation_list) != len(self.robot_list):
                errors.append(
                    f"vel_compensation_list length {len(self.vel_compensation_list)} "
                    f"!= robot_list length {len(self.robot_list)}"
                )

        if not (0 <= self.ros.domain_id <= 101):
            errors.append(f"ros.domain_id={self.ros.domain_id} outside valid range [0, 101]")

        valid_modes = ["human", "rgb_array", ""]
        if self.display.mode not in valid_modes:
            errors.append(
                f"display.mode='{self.display.mode}' is invalid; "
                f"valid values: {valid_modes}"
            )

        if errors:
            error_msg = "\nConfig validation failed:\n  " + "\n  ".join(errors)
            raise ValueError(error_msg)
