#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: sensor_scheduler.py
Brief: sim-time-driven sensor sampling scheduler (single-threaded).

Sensor sampling scheduler. It starts no thread of its own — tick() is called by
the physics thread after every physics step, and sim time decides which sensors
are due. Sampling therefore happens on the same thread as mj_step / scene.step,
which satisfies the single-thread state-read constraint of both MuJoCo and
Genesis and keeps the sensors time-aligned with each other.
"""
from dataclasses import dataclass
from typing import Callable, Dict

from simu_utils.simu_common_tools import astribot_simu_log


@dataclass
class _SensorEntry:
    name: str
    period: float  # sampling period (sim seconds) = 1/frequency
    sample_and_publish: Callable[[], None]
    next_time: float = 0.0  # sim_time of the next sample
    count: int = 0  # samples taken so far (for tests / diagnostics)


class SensorScheduler:
    """
    sim-time driven sensor sampling scheduler.

    Usage:
        sched = SensorScheduler(max_rate_hz=1/dt)
        sched.register('imu', 200.0, self._sample_publish_imu)
        sched.register('lidar', 10.0, self._sample_publish_lidar)
        sched.start(sim_time=0.0)
        # inside the physics thread:
        for _ in range(frame_skip):
            step_physics()
            sim_time += dt
            sched.tick(sim_time)
    """

    def __init__(self, max_rate_hz: float):
        """
        Args:
            max_rate_hz: physics step frequency (1/dt). Sampling only happens
                         after a physics step, so this is the hard upper bound
                         on any sensor's effective sample rate.
        """
        self._max_rate_hz = max_rate_hz
        self._sensors: Dict[str, _SensorEntry] = {}
        self._started = False
        self._start_time = None
        # Sensor names whose sample/publish error has already been reported, so
        # tick() warns once instead of every physics step.
        self._error_warned: set = set()

    def register(self, name: str, frequency: float, sample_and_publish: Callable[[], None]):
        """Register a sensor in the schedule.

        Args:
            name: unique sensor identifier (e.g. 'imu', 'lidar')
            frequency: target sampling frequency (Hz)
            sample_and_publish: no-arg callback — read physics state, build the
                                message, publish it
        """
        if name in self._sensors:
            raise ValueError(f"sensor '{name}' already registered")

        # Above the physics ceiling: warn and clamp to 1/dt.
        eff_freq = frequency
        if frequency > self._max_rate_hz + 1e-9:
            astribot_simu_log(
                f"[sensor-sched] '{name}' requested {frequency}Hz exceeds physics "
                f"rate {self._max_rate_hz:.1f}Hz (1/dt); clamped to {self._max_rate_hz:.1f}Hz",
                level="WARN",
            )
            eff_freq = self._max_rate_hz
        if eff_freq <= 0:
            raise ValueError(f"sensor '{name}' frequency must be > 0")

        self._sensors[name] = _SensorEntry(
            name=name,
            period=1.0 / eff_freq,
            sample_and_publish=sample_and_publish,
        )

    def start(self, sim_time: float = 0.0):
        """Anchor the first sampling instant of every sensor.

        Args:
            sim_time: current simulation time (e.g. data.time, scene.cur_t)
        """
        self._start_time = sim_time
        for e in self._sensors.values():
            # The first sample lands at start_time + period, not immediately, so a
            # 10Hz LiDAR fires exactly 10 times in 1.0s rather than 11.
            e.next_time = sim_time + e.period
        self._started = True

    def tick(self, sim_time: float):
        """Called by the physics thread after every physics step; samples and
        publishes whichever sensors are due.

        Args:
            sim_time: current simulation time (e.g. data.time, scene.cur_t)
        """
        if not self._started:
            return

        for e in self._sensors.values():
            # Allow a small float tolerance (1e-9) so boundary samples aren't missed
            if sim_time + 1e-9 >= e.next_time:
                try:
                    e.sample_and_publish()
                    e.count += 1
                except Exception as ex:
                    # Warn once per sensor: tick() runs every physics step, so a
                    # persistently failing sensor would otherwise flood the log.
                    if e.name not in self._error_warned:
                        self._error_warned.add(e.name)
                        astribot_simu_log(
                            f"[sensor-sched] '{e.name}' sample/publish error "
                            f"(further errors for this sensor are suppressed): {ex}",
                            level="WARN",
                        )

                # Advance by period rather than from now, to avoid cumulative drift
                e.next_time += e.period

                # If we fell far behind (e.g. paused in a debugger), re-anchor to now
                # instead of firing a backlog of catch-up samples
                if e.next_time < sim_time - 2 * e.period:
                    e.next_time = sim_time + e.period

    def stats(self) -> Dict[str, int]:
        """Return each sensor's sample count (for tests / diagnostics)."""
        return {name: e.count for name, e in self._sensors.items()}
