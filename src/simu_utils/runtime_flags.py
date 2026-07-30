#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RuntimeFlags - runtime switches

These flags are read from environment variables only; they are never persisted to
a config file. They exist for ad-hoc debugging, script automation and profiling.

How this differs from SimConfig:
- SimConfig:    describes *what the simulation is* (persisted to yaml)
- RuntimeFlags: describes *how this particular run behaves* (transient, env vars)
"""
import os
from dataclasses import dataclass


@dataclass
class RuntimeFlags:
    """Runtime switches - read from environment variables only."""

    # === Genesis debug switches ===
    genesis_headless: bool = False
    """Force Genesis headless mode (overrides display.mode)
    Env var: ASTRIBOT_GENESIS_HEADLESS=1
    Use case: cluster / CI runs, to guarantee no GUI is started
    """

    genesis_rotate: float = 0.0
    """Genesis chassis rotation injection rate (rad/s)
    Env var: ASTRIBOT_GENESIS_ROTATE=0.1745
    Use case: SDK-independent SLAM validation (bypasses the SDK<->sim channel)
    """

    genesis_profile_steps: int = 0
    """Number of steps to profile in Genesis
    Env var: ASTRIBOT_GENESIS_PROFILE_STEPS=100
    Use case: print per-stage timings for N steps, then exit
    """

    # === Video recording ===
    video_out: str = ""
    """Video output path (empty means do not record)
    Env var: ASTRIBOT_VIDEO_OUT=/path/to/output.mp4
    Use case: automatically record the simulation (used by the pipeline)
    """

    video_fps: int = 25
    """Video frame rate
    Env var: ASTRIBOT_VIDEO_FPS=30
    """

    # === Sensor scheduler (migration flag, to be removed) ===
    use_sensor_scheduler: bool = True
    """Whether to enable the sensor scheduler
    Env var: ASTRIBOT_USE_SENSOR_SCHEDULER=1

    Legacy: a PR-2 migration flag; this field will be removed in Stage 3.
    Current default: True (migration complete)
    """

    @classmethod
    def from_env(cls) -> "RuntimeFlags":
        """Build a RuntimeFlags instance from environment variables.

        Returns:
            RuntimeFlags: the set of flags read from the environment
        """
        return cls(
            genesis_headless=os.getenv("ASTRIBOT_GENESIS_HEADLESS") == "1",
            genesis_rotate=float(os.getenv("ASTRIBOT_GENESIS_ROTATE", "0") or "0"),
            genesis_profile_steps=int(os.getenv("ASTRIBOT_GENESIS_PROFILE_STEPS", "0") or "0"),
            video_out=os.getenv("ASTRIBOT_VIDEO_OUT", ""),
            video_fps=int(os.getenv("ASTRIBOT_VIDEO_FPS", "25") or "25"),
            use_sensor_scheduler=os.getenv("ASTRIBOT_USE_SENSOR_SCHEDULER", "1") == "1",
        )

    def apply_to_env(self, env_instance):
        """Apply the RuntimeFlags to an environment instance.

        Some flags must take effect during env initialization; this method keeps
        that application logic in one place.

        Args:
            env_instance: an AstribotBaseEnv instance or subclass thereof
        """
        # Genesis headless mode
        if self.genesis_headless and hasattr(env_instance, "show_viewer"):
            env_instance.show_viewer = False

        # Video recording
        if self.video_out and hasattr(env_instance, "video_out_path"):
            env_instance.video_out_path = self.video_out
            env_instance.video_fps = self.video_fps

        # Sensor scheduler
        if hasattr(env_instance, "_use_sensor_scheduler"):
            env_instance._use_sensor_scheduler = self.use_sensor_scheduler

        # Genesis-specific
        if hasattr(env_instance, "rotate_yaw_rate"):
            env_instance.rotate_yaw_rate = self.genesis_rotate
        if hasattr(env_instance, "_prof_steps"):
            if self.genesis_profile_steps > 0:
                env_instance._prof_steps = self.genesis_profile_steps

    def __repr__(self) -> str:
        """Formatted output (for logging)."""
        active_flags = []
        if self.genesis_headless:
            active_flags.append("genesis_headless")
        if self.genesis_rotate != 0:
            active_flags.append(f"genesis_rotate={self.genesis_rotate:.4f}")
        if self.genesis_profile_steps > 0:
            active_flags.append(f"profile_steps={self.genesis_profile_steps}")
        if self.video_out:
            active_flags.append(f"video_out={self.video_out}")
        if not self.use_sensor_scheduler:
            active_flags.append("sensor_scheduler=OFF")

        if not active_flags:
            return "RuntimeFlags(none active)"
        return f"RuntimeFlags({', '.join(active_flags)})"
