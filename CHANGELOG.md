# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-07-29

First external release. Narrowed to a **MuJoCo + Genesis dual backend** and the two
robot models **astribot_s1 / astribot_t1**, with T1 as the default.

### Added

- Genesis backend upgraded to 1.2.x: native async rendering, per-link gravity
  compensation (public API), and a pace lock (`precise_sleep`) matching MuJoCo's
  real-time behaviour.
- Dual kinematic / dynamic chassis MJCF selection via `chassis_model` plus
  `backend_overrides`: Genesis uses kinematic virtual joints
  (`chassis_x/y/zrot`) for stability, MuJoCo uses real roller dynamics.
- Centralised timing: `physics_hz` / `control_hz` in `config/simulation.yaml` are
  the single source of truth, `resolve_timing()` derives `frame_skip`, and both
  backends run at 200 / 100 Hz.
- `scripts/verify_env.sh` environment self-check: package versions, project
  import, all four robot x backend config resolutions, and a launch smoke test
  for both backends (Genesis in its own headless process).
- `scripts/install.sh` one-shot install: auto-installs Miniconda when absent,
  creates the `astribot_simulation` env, and builds `astribot_msgs`.
- Periodic real-time-factor logging, with the interval set by
  `ASTRIBOT_RTF_REPORT_SEC` (default 10 s).

### Changed

- CLI reduced to `--backend {mujoco,genesis}`; the robot is selected by a
  positional argument or `$ROBOT_TYPE`.
- Robot ids renamed to `astribot_s1` / `astribot_t1`, with the config and scene
  directories renamed to match.
- Conda environment name unified to `astribot_simulation`.
- Default robot is now **T1** (`env.sh` exports `ROBOT_TYPE=T1`).
- The `astribot_descriptions` submodule now points at a public GitHub repository
  over https, tracking branch `aos`. It previously used an internal SSH URL that
  external users could not clone.
- `config/` reduced to `simulation.yaml` plus one `sim.yaml` per robot. Every
  field is documented in English with its meaning, accepted values and range.
- Logging cleaned up: INFO keeps only the startup banner, `ASTRIBOT_DEBUG=1`
  enables full DEBUG output.
- Renamed `mecanum` to `omni` throughout: the wheels are +/-45 degree omni
  wheels, not mecanum.
- Dependency versions aligned: mujoco >= 3.5.0, genesis-world >= 1.2.0,
  numpy 1.26.4, gymnasium 1.1.1.
- All comments, docstrings and runtime strings under `src/` and the install
  scripts are now English.
- Version unified to 1.2.0 across `pyproject.toml` and the `astribot_msgs`
  package manifests.

### Fixed

- Two places used a joint index where an actuator index was required.
  `set_actuator_parameters()` received an index into the joint list (25 entries
  for s1) rather than the actuator array (26 entries), shifting every body
  actuator by one and leaving the last one (the right gripper) without its gain
  restored. `get_gravity_torque()` indexed `qvel`/`qacc` by joint ID instead of
  dof address, and tendon-driven joints were skipped entirely, so stale velocity
  leaked into the computed "gravity" torque.
- 29 call sites passed the log level as a positional argument to
  `astribot_simu_log()`, so WARN and ERROR messages were printed as INFO.
- Gripper position reporting clamped only the lower bound, and did so by writing
  back into `data.qpos` -- a state read mutating the physics. Values beyond the
  joint limit were reported outside the interface's own 0-100 range. The returned
  value is now clamped and the physics state is left untouched.
- After the MuJoCo passive viewer window was closed, `render()` still called the
  now-`None` `mujoco_renderer` and raised `AttributeError`.
- The FastDDS interface whitelist in `env.sh` only checked whether the variable
  was set, never whether the pinned address still existed. After a network
  change the stale whitelist bound DDS to a vanished interface, so topics were
  published but no subscriber could discover them. Stale whitelists are now
  detected and discarded.
- Ctrl+C no longer floods the console with `ExternalShutdownException` and
  `publisher's context is invalid` tracebacks from the physics and spin threads;
  these are shutdown races, not failures.
- Genesis omni chassis tipping and drifting: rollers freed, feedforward timing
  aligned, and a feedback index mismatch corrected.
- Genesis freejoint robots floated away because material gravity compensation
  was zeroed for the whole model.
- Chassis yaw oscillation: state and closed loop moved into body integration
  space to match the real robot's semantics.
- The omni-wheel inverse-kinematics translation gain was derived incorrectly as
  `1/(r*sin45)`, twice the geometrically correct `cos45/r`, making open-loop
  velocity control translate 2x too fast and rotate 0.7x too slow.

### Known Issues

- **Gripper can intermittently jam.** On the MuJoCo backend a gripper may fail to
  reopen after closing (either side, depending on startup timing). The mechanism
  is understood: the four-bar linkage's ring of equality constraints sits at the
  edge of numerical stability at a 5 ms step and gets forced open by the
  actuator, wedging the linkage. Not yet fixed; restart the simulation to
  recover.

### Removed

- ManiSkill and Isaac Lab backends are excluded (not mature enough to ship).
- Unreleased and legacy robot configs and scenes removed (a1_v02, t1_v2).
- Internal tests, development docs, debug and SLAM scripts, and CI / lint
  configuration are no longer part of the release tree.

## [1.1.0] - 2026-03-30

### Fixed

- Fixed `handle_object_pose_command` missing `response` parameter causing ROS2 service crash (F821)
- Fixed bare `except:` clauses to use `except Exception:` in MuJoCo and Isaac Lab envs (E722)
- Fixed boolean comparison style: `== False` to `not`, `== True` to direct usage, `!= None` to `is not None` (E711/E712)
- Removed duplicate `import sapien` in ManiSkill env (F811)
- Removed unused variable assignments across codebase (F841)

### Changed

- Migrated project source code to `src/` layout, updated all internal imports to `src.` prefix
- Consolidated `requirements.txt` and `requirements-dev.txt` into a single `requirements.txt` aligned with `scripts/install.sh`
- Bumped version to 1.1.0 across all version files (pyproject.toml, src/version.py, astribot_msgs package manifests, CMakeLists)
- Removed material/texture references from "without_texture" MJCF model files
- Fixed MJCF include chain so without-texture main file references without-texture model

### Removed

- Removed auto-generated intermediate documentation from `docs/` directory
- Removed `requirements-dev.txt` (consolidated into `requirements.txt`)

## [0.1.0] - 2026-03-30

### Added

- Initial release with MuJoCo, Genesis, ManiSkill, Isaac Lab support
- ROS1 and ROS2 integration
- Multi-robot control interface
- Factory pattern for simulator backends
