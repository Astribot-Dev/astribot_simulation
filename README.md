# Astribot Simulation

[中文文档 / Chinese](README_zh.md)

**A unified robot simulation platform** built on top of **MuJoCo** and **Genesis**. This project provides a common abstraction layer that lets you switch seamlessly between the two simulators with a consistent API, intended as a research tool for Astribot users.

<p align="center">
  <img src="docs/system.png" alt="Astribot Simulation System" width="600">
</p>

---

## Demo

<p align="center">
  <b>MuJoCo Simulation</b><br>
  <img src="docs/mujoco.gif" alt="MuJoCo Demo" width="400">
</p>

---

## Features

- **Dual-backend environments**: switch between **MuJoCo** and **Genesis** with the single `--backend` CLI flag.
- **Two robot models**: `astribot_s1` and `astribot_t1`.
- **Photorealistic rendering**: high-fidelity meshes with texture maps and PBR materials
  (OBJ / DAE / STL); scenes include ground reflectance, multiple light sources and
  configurable cameras. The Genesis backend supports GPU rasterized rendering and can
  output RGB / depth / point clouds.
- **Zero-force drag (Zero-G)**: `mode=3` torque control matches the real robot's chain --
  the torque command is passed straight through, plus dynamics feed-forward, plus a small
  damping term (kp=0). When the SDK sends all-zero torques the arm holds its pose, so it
  can be dragged by hand for teaching.
- **Dynamics feed-forward compensation**: `M(q)*qdd + C(q,qd)*qd + g(q)`, all computed from
  the measured feedback state, configurable term by term.
- **Scenes decoupled from robot descriptions**: scenes live on their own under
  `astribot_scenes/` and reference the robot model in `astribot_descriptions/` through
  MJCF's native `<include>`; the model files contain no scene content. Adding a scene means
  creating one directory, dropping in two MJCF files and changing a single `scene:` config
  line -- **no edits to the robot description files or to any code**, and no impact on
  other scenes or robots.
- **Pairs with the Astribot SDK**: the simulation reuses the real robot's ROS topics,
  message types and joint naming. Setting `ASTRIBOT_ON_SIMULATION_MODE=1` on the SDK side
  is all it takes to point the same control code at the simulation instead of hardware,
  with **no changes to application logic**; the SDK's `examples/` run against the
  simulation as-is. Messages published by the simulation carry a `frame_id="simulation"`
  marker, and if a real robot is detected publishing state in the same ROS domain, the
  simulation disables itself to avoid both writing at once.
- **ROS / ROS2 integration**: supports both ROS1 and ROS2 interfaces, bridging simulation and hardware.
- **Plug-and-play robot models**: supports **URDF** and **MJCF** formats.
- **Config-driven**: per-robot YAML, with shared parameters centralized in `config/simulation.yaml`.
- **Real-time first**: publishes only control-related topics (joint states + chassis IMU), RTF around 1.0.

---

## System Requirements

- **OS:** Ubuntu 20.04 / 22.04 LTS
- **Middleware:** ROS Noetic (20.04) / ROS2 Humble (22.04)
- **Python:** 3.10 or 3.11 (prepared by the install script inside a Conda environment)
- **Physics engine versions:**

  | Backend | Package | Required version | Notes |
  | ---- | ---- | -------- | ---- |
  | MuJoCo | `mujoco` | **>= 3.5.0** | Async rendering needs `state_only` (3.3.4+); verified on 3.11.0 |
  | Genesis | `genesis-world` | **>= 1.2.0, < 2.0.0** | The MJCF loading API in 1.0 / 0.2.x is incompatible; verified on 1.2.3 |

| Hardware / Software | MuJoCo (minimum)              | Genesis / dual-backend (recommended) |
| ------------------- | ----------------------------- | ------------------------------------ |
| CPU                 | 4+ cores                      | Intel i5-14600F or higher            |
| RAM                 | 8 GB                          | 16 GB or more                        |
| GPU                 | Not required (CPU-only works) | NVIDIA RTX 2080 Ti or higher         |
| GPU driver          | --                            | NVIDIA driver >= 535                 |
| CUDA                | --                            | >= 12.0                              |
| Disk                | 5 GB                          | 15 GB or more (includes torch/CUDA)  |

> **Note:** the MuJoCo backend runs on CPU alone; the **Genesis backend requires an NVIDIA GPU** (it uses CUDA/Taichi internally). If you only use MuJoCo, the GPU-related rows can be ignored.

---

## Installation

There is only one way to install: run `install.sh` on the host machine (or inside a container you have already entered). The script prepares **all** dependencies, including the Conda environment itself.

### Prerequisites

All you need beforehand is ROS installed and sourced -- this must be done **before** running the install script, otherwise the ROS messages will not build correctly:

```bash
# Ubuntu 22.04 (ROS2 Humble)
source /opt/ros/humble/setup.bash

# or Ubuntu 20.04 (ROS1 Noetic)
source /opt/ros/noetic/setup.bash
```

### One-shot Install

```bash
# 1. Clone the repository
git clone https://github.com/Astribot-Dev/astribot_simulation.git
cd astribot_simulation

# 2. Init submodules and pull the LFS mesh files (MuJoCo will not start without this)
git submodule update --init --recursive
git submodule foreach git lfs pull

# 3. Install (the script installs Miniconda automatically if Conda is missing)
bash scripts/install.sh
```

`install.sh` performs, in order:

1. Checks system / Python / GPU / CUDA
2. **Downloads and installs Miniconda if no Conda is detected**
3. Creates the Conda environment **`astribot_simulation`** (Python 3.11; override with `ASTRIBOT_SIMU_PY_VERSION=3.10`)
4. Installs dependencies: numpy / mujoco / genesis-world / gymnasium / torch(+CUDA) / opencv / transforms3d, etc.
5. **Builds `astribot_msgs`** (cmake/make on ROS1, colcon on ROS2)
6. Installs this project in editable mode

You are ready to go once it finishes -- no other scripts need to be run by hand.

### Verifying the Install

```bash
conda activate astribot_simulation
source /opt/ros/humble/setup.bash
source env.sh
bash scripts/verify_env.sh
```

All green, exit code 0. The script validates key package versions, that the project imports, that all 4 `robot x backend` config combinations parse correctly, and launches MuJoCo and Genesis separately as a startup smoke test (Genesis in its own process, headless). Add `--no-launch` if you only want the version checks without the launch smoke test.

---

## Quick Start

```bash
# 1. Activate the environment (in every new terminal)
conda activate astribot_simulation
source /opt/ros/humble/setup.bash      # /opt/ros/noetic/setup.bash on ROS1

# 2. Set the project environment variables (exports ASTRIBOT_SIMU_ROOT / ROBOT_TYPE / PYTHONPATH)
source env.sh                          # T1 by default; or --robot s1

# 3. Launch
python3 astribot_simulation.py astribot_t1
```

> You do **not** need to run `scripts/build.sh` separately -- `install.sh` already built the ROS messages. You only need to rebuild after you **modify a `.msg/.srv/.action` under `astribot_msgs/`**, or after a `git clean` removed the generated artifacts:
>
> ```bash
> bash scripts/build.sh
> ```
>
> Launching without building yields `ImportError: cannot import name 'RawRequest' from 'astribot_msgs.srv'`.

### Choosing the Robot and Backend

```bash
python3 astribot_simulation.py astribot_t1                      # T1 + MuJoCo (default backend)
python3 astribot_simulation.py astribot_s1                      # S1 + MuJoCo
python3 astribot_simulation.py astribot_t1 --backend genesis     # T1 + Genesis
python3 astribot_simulation.py astribot_s1 --backend genesis     # S1 + Genesis
```

When the robot is omitted it falls back to the `$ROBOT_TYPE` environment variable (exported by `env.sh --robot`). The CLI argument takes precedence, so switching robots does not require re-sourcing:

```bash
source env.sh                                 # ROBOT_TYPE=T1
python3 astribot_simulation.py astribot_s1    # switch straight to S1
```

Press **Backspace** during simulation to reset the robot state.

### Published Topics and Real-time Factor

This release **publishes only control-related topics**: per-component joint states
(`/astribot_*/joint_space_states`) and the chassis IMU (`/astribot_whole_body/chassis_imu`).
Cameras, LiDAR, the LiDAR-integrated IMU and force/torque sensors are **not supported** --
see [Sensors](#sensors) below.

This is a deliberate trade to prioritize the real-time factor: measured RTF for S1 + MuJoCo
is around 0.99 (sim time 1:1 with wall clock). The simulation also throttles the human-view
window to about 17Hz and sleeps back to 1:1 when it runs faster than real time, so sim time
never races ahead of the wall clock and desynchronizes from VR teleoperation or the real robot.

Available runtime environment variables:

| Environment variable          | Effect                                             |
| ----------------------------- | -------------------------------------------------- |
| `ASTRIBOT_PACE_REALTIME=0`  | Disable the real-time pacing lock (run as fast as possible offline) |
| `ASTRIBOT_HUMAN_RENDER_HZ`  | Set the human-view window frame rate (default about 17Hz) |
| `ASTRIBOT_MUJOCO_PASSIVE=0` | Fall back to synchronous rendering on MuJoCo       |
| `ASTRIBOT_RTF_REPORT_SEC`   | RTF report interval in seconds (default 10, `0` disables) |
| `ASTRIBOT_DEBUG=1`          | Enable DEBUG-level logging                         |

---

## Detailed Usage

### Building the ROS Messages (first run, or after cleaning `astribot_msgs/`)

The custom service / action types under `astribot_msgs/` are generated at build time from the
`*.srv` / `*.action` files. The generated artifacts land in `astribot_msgs/install/` and are
**not committed to the repository**. If you skip this step, launching `astribot_simulation.py` shows:

```
ImportError: cannot import name 'RawRequest' from 'astribot_msgs.srv' (unknown location)
```

You only need to build once per clone (or again after adding new message types, switching ROS
version, or `git clean`-ing `astribot_msgs/{build,install,log}/`):

```bash
# The ROS environment and the conda environment must be activated first
source /opt/ros/humble/setup.bash             # or /opt/ros/noetic/setup.bash for ROS1
conda activate astribot_simulation

# build.sh symlinks the correct CMakeLists / package.xml for the current ROS version,
# then runs colcon build (ROS2) or cmake/make (ROS1)
bash scripts/build.sh
```

**`install.sh` already includes this step**, so a fresh install needs no manual build. You only need it after modifying message definitions or cleaning.

### Config File Lookup Order

Each `(robot_type, backend)` pair maps to one yaml under `config/astribot_<robot>/`:

| Backend     | YAML file    | Notes                                                                     |
| ----------- | ------------ | ------------------------------------------------------------------------- |
| `mujoco`  | `sim.yaml` | The single source of truth                                                |
| `genesis` | `sim.yaml` | The same file; per-backend differences live in `backend_overrides.<backend>` |

Both backends share the same per-robot `sim.yaml`; there are no longer separate per-backend files.

---

## Project Structure

```
astribot_simulation/
├── astribot_simulation.py         # Main entry point
├── env.sh                         # Environment activation (ROS / ASTRIBOT_SIMU_ROOT / ROBOT_TYPE)
├── pyproject.toml                 # Project metadata / version
├── requirements.txt               # Python dependencies
│
├── src/
│   ├── astribot_envs/             # Per-backend environment wrappers
│   │   ├── astribot_base_env.py       # Abstract base class (inherits gym.Env)
│   │   ├── astribot_mujoco_env.py     # MuJoCo backend
│   │   ├── astribot_genesis_env.py    # Genesis backend
│   │   ├── astribot_envs_factory.py   # Factory: loads config, builds the gym env
│   │   └── simulation_constants.py    # Timing derivation and constants
│   │
│   ├── simu_utils/                # Common utilities
│   │   ├── simu_common_tools.py       # Poses, point clouds, logging
│   │   ├── robot_ros_interface.py     # ROS1/ROS2 transceiver
│   │   ├── chassis_kinematics.py      # Omni-wheel kinematics and chassis controller stubs
│   │   ├── config_loader.py           # Six-layer config merge
│   │   ├── sim_config.py              # Config dataclasses (schema)
│   │   └── sensor_scheduler.py        # Sim-time-driven sensor sampling
│   │
│   └── sim_assets_tools/          # Mesh / texture processing tools
│
├── config/                        # YAML configs (see below)
├── astribot_scenes/               # Scene MJCF (s1 / t1)
├── astribot_descriptions/         # Robot models (git submodule + Git LFS)
├── astribot_msgs/                 # Custom ROS messages / services / actions
├── scripts/                       # Install, build and verify scripts
└── docs/                          # Images and diagrams
```

---

## Architecture

`astribot_simulation` uses a factory pattern to support multiple backends, keeping cohesion high and coupling low.

### Class Diagram

<p align="center">
  <img src="docs/class_diagram.png" alt="Astribot Simulation Class Diagram" width="900">
</p>

### Design Notes

- **Unified API**: every backend environment inherits `AstribotBaseEnv` and is compatible with `gym.Env`.
- **Factory dispatch**: `AstribotEnvsFactory` dynamically loads the backend from the `simulator_type` derived from `--backend`.
- **Sim-to-real bridge**: `MultiRobotRosInterface` keeps simulation state strictly aligned with the ROS messages, making deployment to hardware seamless.

> **About the other backends:** environment modules for ManiSkill and Isaac Lab are still in the tree, but they are **immature and disabled in this release** -- they are neither installed nor exposed. `--backend` only accepts `mujoco` and `genesis`.

---

## Configuration Files

### File Layout

```
config/
├── simulation.yaml            # Shared: robot-independent physics/display/ROS/controller defaults
├── astribot_s1/
│   ├── sim.yaml               # S1 robot config
│   └── sensor_calib.json      # Camera intrinsics/extrinsics (optional, not enabled in this release)
└── astribot_t1/
    └── sim.yaml               # T1 robot config
```

The config has two layers, deep-merged at load time: the shared layer holds robot-independent
defaults, and the per-robot layer carries only what is specific to that robot (joint tables,
chassis geometry, scene). Each field's meaning, allowed values and range are also written as
**English comments** inside the yaml, so you can read them straight from the file.

### Shared Config (`config/simulation.yaml`)

| Field                         | Meaning                                                     |
| ----------------------------- | ----------------------------------------------------------- |
| `physics.physics_hz`        | Physics step rate (default 200). **The single frequency source**; `dt = 1/physics_hz` |
| `physics.control_hz`        | Control rate (default 50). `frame_skip = physics_hz / control_hz` is derived from it |
| `physics.gravity`           | Gravity vector, default `[0, 0, -9.81]`                   |
| `display.mode`              | `human` (GUI window) / `rgb_array` (off-screen) / `''` (no rendering) |
| `display.width/height/fps`  | Window resolution and refresh rate                          |
| `ros.domain_id`             | `ROS_DOMAIN_ID`; must match the SDK side to communicate   |
| `ros.qos_depth`             | ROS2 QoS queue depth                                        |
| `controller_config`         | **Global defaults** for controller gains and dynamics compensation, see below |
| `imu_topic` / `lid_topic` / `chassis_imu_topic` | Sensor topic names (placeholder fields, see the Sensors section) |
| `sensor_frequencies`        | Per-sensor sampling rates (placeholder fields)              |

> **Only change these two fields for timing**: `physics_hz` and `control_hz` are the single
> source of truth. `frame_skip` and `dt` are both derived by `resolve_timing()` and shared by
> the two backends -- do not hardcode them anywhere else.

### Per-robot Config (`config/<robot>/sim.yaml`)

| Field                     | Meaning                                                           |
| ------------------------- | ----------------------------------------------------------------- |
| `robot_type`            | Robot identifier (`s1` / `t1`), matching the directory name   |
| `robot_name`            | ROS topic prefix, default `astribot_whole_body`                  |
| `scene`                 | Scene name, resolved to `astribot_scenes/<robot>/<scene>/scene.xml` |
| `simulator_type`        | Backend class (`Mujoco` / `Genesis`), normally overridden by `--backend` |
| `chassis_type`          | Chassis type: `omni` (omni-directional wheels)                  |
| `chassis_fixed`         | `true` locks the chassis pose; more stable when only tuning the arms |
| `omni.*`                | Chassis wheel geometry: `wheel_radius`, `half_wheelbase`, `half_track`, etc. |
| `robot_list`            | Component list; determines which ROS topic groups are published    |
| `joint_names_list`      | **Joint names grouped by component**; the order must map 1:1 onto the MJCF actuator order |
| `vel_compensation_list` | Whether velocity feed-forward is enabled per component             |
| `controller_config`     | Gains and compensation for this robot (overrides the shared layer) |
| `object_names`          | Free-floating scene objects whose poses are published              |
| `backend_overrides`     | Override any of the fields above per backend, see below             |

> **The `joint_names_list` order is a hard constraint**: it maps one-to-one onto the MJCF
> actuator order, chassis first. A misordered list writes commands to the wrong joints.

Merge priority (low to high):

```
SimConfig defaults < simulation.yaml < per-robot sim.yaml < backend_overrides.<backend> < environment variables
```

`backend_overrides` lives in the per-robot `sim.yaml` and lists only the fields that differ from the default for that backend. For example on S1: MuJoCo uses `chassis_model: dynamic` (real roller physics) and raises `control_hz` to 100; Genesis uses `chassis_model: kinematic` (virtual joints, avoiding instability from high-speed roller contacts).

The scene is not a CLI argument; it comes from the `scene` field in the per-robot `sim.yaml` and resolves to `astribot_scenes/<robot>/<scene>/scene[_kinematic].xml`.

### Controller Config (`controller_config`)

Three-level priority: `per_joint` > `per_component.default_gains` > `default_gains`.

```yaml
controller_config:
  default_gains:                    # Global fallback
    position: {kp: 100.0, kd: 10.0}
    torque:   {kp: 0.0,   kd: 0.1}

  per_joint:                        # Exact per-joint config (highest priority)
    astribot_arm_right_joint_1:
      position: {kp: 100.0, kd: 10.0}   # mode=1
      torque:   {kp: 0.0,   kd: 2.0}    # mode=3, kp must be 0

  dynamics_compensation:
    enabled: true
    modes: [1, 2, 3]                # Which control modes get compensation
    components:
      gravity: true                 # g(q)
      coriolis: true                # C(q,qd)*qd
      inertia: false                # M(q)*qdd (off by default, see the Joint Space Command section)
      friction: false               # Not implemented (the model has no Coulomb friction)
```

| Field               | Meaning                                                              |
| ------------------- | -------------------------------------------------------------------- |
| `position.kp`     | Position gain -> MuJoCo `gainprm[0]`; also sets `biasprm[1] = -kp` |
| `position.kd`     | Velocity damping -> absolute value of MuJoCo `biasprm[2]`. **Should match the MJCF** |
| `torque.kp`       | Position gain for mode=3, **must be 0** (real-robot semantics)    |
| `torque.kd`       | Small damping for mode=3, **must be configured per joint** (it differs per joint on the real robot) |
| `modes`           | Modes in which compensation applies; a mode not listed gets no feed-forward |
| `components.*`    | Individual switches for the three dynamics terms                     |

> **`position.kd` is velocity damping, not the position feedback coefficient.** The three
> MJCF `biasprm` slots are, in order: bias, position feedback (`-kp`), velocity damping
> (`-kd`). Historically kd was filled with the kp value, inflating damping by 10x; it is now
> aligned with the real MJCF values.

> ### About the sensor fields (important)
>
> `camera_names` and `sensor_names` in the per-robot `sim.yaml`, and `sensor_frequencies`,
> `imu_topic`, `lid_topic` in `simulation.yaml`, are all **placeholder fields kept only to
> keep the schema stable**.
>
> **This release does not support cameras or LiDAR.** These fields being empty or at their
> defaults does not mean "fill them in and it will work". The corresponding code paths are
> not sufficiently validated, so **do not modify these fields to try to enable sensors** --
> that leads to unpredictable simulation behaviour. The only sensor that actually works is
> the chassis IMU (`astribot_chassis_base_imu_gyro` in `sensor_names`).
>
> Sensor support will be provided properly in a later release.

---

## Sensors

> **This release does not support sensors.** The simulation paths for cameras, LiDAR, the
> LiDAR-integrated IMU and force/torque sensors are not mature, so they are **uniformly
> disabled** in the release, with no switch to turn them on (`--profile` has been removed
> from the CLI). Please do not try to enable them by editing the config -- those code paths
> are not sufficiently validated and may make the simulation unstable or the data untrustworthy.

| Sensor                  | MuJoCo            | Genesis           |
| ----------------------- | ----------------- | ----------------- |
| Camera RGB              | Not supported     | Not supported     |
| Camera depth            | Not supported     | Not supported     |
| Camera point cloud      | Not supported     | Not supported     |
| LiDAR point cloud       | Not supported     | Not supported     |
| LiDAR-integrated IMU    | Not supported     | Not supported     |
| Force / torque sensor   | Not supported     | Not supported     |
| **Chassis IMU**   | **Supported** | **Supported** |

**The only usable sensor topic** is the chassis IMU:

- `/astribot_whole_body/chassis_imu` -- `std_msgs/Float64MultiArray`, 9 values
  (rpy, angular velocity, linear acceleration), semantically aligned with the real robot, 200Hz by default.

The `camera_names: []` and `sensor_names: [...]` fields in the per-robot `sim.yaml` are kept as **schema placeholders**; please leave them as they are. Camera and LiDAR support will come in a later release.

## Joint Space Command

To control the joints directly, the robot exposes joint-space command topics (e.g. `/astribot_arm_left/joint_space_command`). The meaning of the command depends on the control mode:

| `mode` | Control mode | Command dimension | Meaning | Feed-forward |
| ------ | ------------ | ----------------- | ------- | ------------ |
| `1` | Position control | 7-14 | **First 7 values**: target joint positions; **last 7 values**: target joint velocities (velocity feed-forward) | Yes |
| `2` | Velocity control | 7 | Target velocity per joint | Yes |
| `3` | Torque control (zero-force drag / Zero-G) | 7 | **Additional** torque per joint, usually all zeros | Yes |

### Dynamics Feed-forward Compensation

All three modes add a dynamics feed-forward term, controlled by `controller_config.dynamics_compensation`:

```
compensation = M(q)*qdd + C(q, qd)*qd + g(q)
```

Every term is computed from the **measured feedback state** `q, qd, qdd` (matching the real
robot's middle layer; no desired acceleration is used). Terms can be toggled individually via
`components`: `gravity` and `coriolis` are on by default, `inertia` is off by default (`qdd`
is a noisy second-order signal that couples with the controller output within the same step;
enabling it measurably slows convergence).

### Torque Control (mode=3) and Zero-force Drag

On the real robot, zero-force drag works like this: the torque command from the SDK is
**passed straight through**, the middle layer adds the dynamics feed-forward, and a small
damping term is applied (desired velocity zero, **position gain kp = 0**):

```
joint torque = torque command + dynamics compensation - kd * qd
```

The simulation implements the same chain, so when the SDK sends all-zero torques the arm
**holds its current pose** instead of dropping -- that is zero-force drag: push the arm and it
moves along, release it and it stays where you left it.

The damping `kd` is configured **per joint** through the `torque` section of `controller_config` (`kp` must be 0):

```yaml
astribot_arm_right_joint_1:
  position: {kp: 100.0, kd: 10.0}   # mode=1
  torque:   {kp: 0.0,   kd: 2.0}    # mode=3: kp must be 0
```

> **Note:** the current `torque.kd` values are empirical, chosen to match the real robot's
> order of magnitude but **not calibrated for feel**, so the zero-force drag damping does not
> match the real robot exactly. The control chain semantics are aligned, but sim-to-real
> force-control accuracy is not guaranteed.

**Tip:** when using the **Astribot SDK** you generally do not need to worry about these details -- refer to the SDK documentation. When sending joint commands directly, make sure the command dimension matches the control mode.

---

## Scene Files (`astribot_scenes/`)

The `astribot_scenes/` directory holds MuJoCo MJCF scene definitions organized by robot model and environment type. Each scene contains the ground, walls, lighting and optional props (such as a table).

### Directory Layout

```
astribot_scenes/
├── s1/                    # S1 robot scenes
│   ├── floor/                 # Empty floor (basic testing)
│   ├── table/                 # Room with a table (manipulation tasks)
│   └── warehouse/             # Warehouse environment (SLAM verification)
└── t1/                    # T1 robot scenes
    ├── floor/
    ├── table/
    └── warehouse/
```

### Scene Variants

Each scene directory contains two MJCF files:

- **`scene.xml`** -- the full scene with a **dynamic chassis** (real wheels, roller contacts, full physics)
- **`scene_kinematic.xml`** -- the **kinematic chassis** scene (virtual joints, visual-only wheels, faster and more stable)

The active variant is selected via `backend_overrides.mujoco.chassis_model` in `config/<robot>/sim.yaml`:

```yaml
backend_overrides:
  mujoco:
    chassis_model: dynamic      # uses scene.xml
    # chassis_model: kinematic  # uses scene_kinematic.xml
```

### Available Scenes

|          Robot | Scene         | Description                             |
| -------------: | ------------- | --------------------------------------- |
| **s1** | `floor`     | Empty flat ground, good for basic mobility and arm testing |
|                | `table`     | 3-wall room + table, for manipulation and SLAM verification |
|                | `warehouse` | 30x30m warehouse-style layout with perimeter walls and obstacles, for SLAM patrol verification |
| **t1** | `floor`     | Empty flat ground                       |
|                | `table`     | Room with a table                       |
|                | `warehouse` | Same layout as s1, with the t1 robot model |

### Scene Selection

The scene is selected via the `scene` parameter in `config/<robot>/sim.yaml`:

```yaml
scene: floor      # default: empty floor
# scene: table    # room with a table
# scene: warehouse # warehouse environment (supported on both s1 and t1)
```

The scene MJCF is included into the robot's top-level MJCF like this:

```xml
<include file="../../astribot_scenes/<robot>/<scene>/scene.xml"/>
```

### Adding a Custom Scene

To create a new scene:

1. Create a directory under `astribot_scenes/<robot>/<scene_name>/`
2. Add `scene.xml` and `scene_kinematic.xml` (copy an existing scene as a template)
3. Adjust the ground, walls, props and lighting as needed
4. Update `config/<robot>/sim.yaml` to reference your scene

---

## Robot Descriptions, Submodule and Git LFS

`astribot_descriptions/` is a **git submodule** (see [.gitmodules](.gitmodules)) whose mesh files are managed with **Git LFS**. **The vast majority of startup failures come from a bad submodule state or missing LFS objects.**

Source and tracking branch:

| Item | Value                                                       |
| ---- | ----------------------------------------------------------- |
| Repo | `https://github.com/Astribot-Dev/astribot_descriptions.git` |
| Branch | `aos`                                                     |
| Transport | https (**no SSH key needed**)                          |

### First-time Setup

```bash
# 1. Init the submodule
git submodule update --init --recursive

# 2. Pull the LFS objects (this is what actually downloads the mesh files)
git submodule foreach git lfs pull
```

If you already cloned without LFS, just run step 2 -- the ~130-byte pointer files get replaced with the real binary STL/DAE.

### Updating the Submodule

```bash
# Check out the commit pinned by the parent repo (recommended: that commit is a verified version)
git submodule update --init --recursive
git submodule foreach git lfs pull

# Or check out the latest tip of the tracking branch aos
git submodule update --remote --recursive
git submodule foreach git lfs pull
```

After `--remote`, remember to commit the new submodule SHA in the parent repo if you want to freeze the pointer. For day-to-day use prefer the former: the SHA pinned by the parent repo is the one verified against this release of the simulation code, whereas following the branch tip may pull in unverified model changes.

### Pre-launch Self-check

```bash
# Should print nothing -- any output means LFS was not fully pulled
find astribot_descriptions -name "*.STL" -size -1k | head

# Files that are only ~130 bytes are just LFS pointers:
#   git submodule foreach git lfs pull
```

---

## Scripts

The release ships only three scripts:

| Script                    | Purpose                                                                        |
| ------------------------- | ------------------------------------------------------------------------------ |
| `scripts/install.sh`    | **Full install**: Miniconda (if missing) + Conda environment + all dependencies + ROS message build |
| `scripts/build.sh`      | **Rebuild `astribot_msgs` on its own**, only needed after changing message definitions |
| `scripts/verify_env.sh` | **Environment self-check**: version validation + config parsing + dual-backend launch smoke test |

> Miniconda is installed automatically by `install.sh`; the installer is no longer shipped with the repository.

---

## Documentation

- Environment wrappers: [`src/astribot_envs/`](src/astribot_envs/)
- Config examples: [`config/astribot_s1/`](config/astribot_s1/), [`config/astribot_t1/`](config/astribot_t1/)
- Utility modules: [`src/simu_utils/`](src/simu_utils/)

---

## Troubleshooting

| Symptom                                          | Cause and fix                                                                        |
| ------------------------------------------------ | ------------------------------------------------------------------------------------ |
| `ImportError: cannot import name 'RawRequest'` | ROS messages not built -> `bash scripts/build.sh`                                 |
| MuJoCo reports missing mesh files / model fails to load | LFS not pulled -> `git submodule foreach git lfs pull`                     |
| `ModuleNotFoundError: rclpy`                   | Wrong activation order -> `conda activate` first, then `source /opt/ros/.../setup.bash` |
| Genesis reports `no valid context` / fails to start | No GPU or no display -> set `ASTRIBOT_GENESIS_HEADLESS=1`, or use `--backend mujoco` |
| RTF noticeably below 1.0                         | Lower the window frame rate with `ASTRIBOT_HUMAN_RENDER_HZ`, or close the GUI window (`display.mode: ''`) |
| Robot state looks wrong, topic data is garbled   | Multiple simulation processes may be competing for the same topics -> `pkill -f astribot_simulation.py` and restart a single instance |

---

## Contributing

Issues and pull requests are welcome. Please make sure `bash scripts/verify_env.sh` passes before submitting.

---

## License

BSD 3-Clause License, see [LICENSE](LICENSE).

---

## Acknowledgements

Thanks to the [MuJoCo](https://mujoco.org/) and [Genesis](https://genesis-world.readthedocs.io/) teams.
