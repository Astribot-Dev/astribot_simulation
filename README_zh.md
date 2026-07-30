# Astribot Simulation 中文文档

[English version](README.md)

**统一的机器人仿真平台**，构建于 **MuJoCo** 与 **Genesis** 之上。本项目提供统一抽象层，让你用一致的 API 在两种仿真器之间无缝切换，面向 Astribot 用户作为研究工具使用。

<p align="center">
  <img src="docs/system.png" alt="Astribot Simulation System" width="600">
</p>

---

## Demo

<p align="center">
  <b>MuJoCo 仿真</b><br>
  <img src="docs/mujoco.gif" alt="MuJoCo Demo" width="400">
</p>

---

## 特性

- **双后端环境**：命令行 `--backend` 一个参数切换 **MuJoCo** / **Genesis**。
- **两种机型**：`astribot_s1` 与 `astribot_t1`。
- **逼真渲染**：带纹理贴图与 PBR 材质的高精度网格模型（OBJ / DAE / STL），场景含地面反射、
  多光源与可配置相机；Genesis 后端支持 GPU 光栅化渲染，可输出 RGB / 深度 / 点云。
- **零力拖动（Zero-G）**：`mode=3` 力矩控制对齐真机链路 —— 力矩命令直接透传 + 动力学
  前馈补偿 + 小阻尼项（kp=0）。SDK 发全 0 力矩时手臂保持姿态，可徒手拖动示教。
- **动力学前馈补偿**：`M(q)·q̈ + C(q,q̇)·q̇ + g(q)`，全部由实测反馈状态计算，逐项可配。
- **场景与模型描述解耦**：场景独立存放于 `astribot_scenes/`，通过 MJCF 原生 `<include>`
  引用 `astribot_descriptions/` 里的机器人模型；模型文件不含任何场景内容。新增场景只需
  建一个目录、放两个 MJCF、改一行 `scene:` 配置，**无需改动模型描述文件或代码**，
  也不影响其它场景与机型。
- **与 Astribot SDK 配套**：仿真复用真机的 ROS 话题、消息类型与关节命名，SDK 侧只需设
  `ASTRIBOT_ON_SIMULATION_MODE=1` 即可把同一套控制代码从真机切到仿真，**无需改动业务
  逻辑**；SDK 的 `examples/` 可直接对仿真运行。仿真发布的消息带 `frame_id="simulation"`
  标记，一旦检测到真机在同一 ROS 域内发布状态，仿真会自动禁用自身以免双写冲突。
- **ROS / ROS2 集成**：同时支持 ROS1 与 ROS2 接口，连通仿真与真机。
- **即插即用机器人模型**：支持 **URDF** 与 **MJCF** 格式。
- **配置驱动**：按机型组织的 YAML，公共参数集中在 `config/simulation.yaml`。
- **实时优先**：只发布控制相关话题（关节状态 + 底盘 IMU），RTF≈1.0。

---

## 系统要求

- **操作系统：** Ubuntu 20.04 / 22.04 LTS
- **中间件：** ROS Noetic（20.04）/ ROS2 Humble（22.04）
- **Python：** 3.10 或 3.11（由安装脚本在 Conda 环境中准备）
- **物理引擎版本：**

  | 后端 | 包名 | 要求版本 | 说明 |
  | ---- | ---- | -------- | ---- |
  | MuJoCo | `mujoco` | **≥ 3.5.0** | 异步渲染需要 `state_only`（3.3.4+）；已验证 3.11.0 |
  | Genesis | `genesis-world` | **≥ 1.2.0, < 2.0.0** | 1.0/0.2.x 的 MJCF 加载 API 不兼容；已验证 1.2.3 |

| 硬件 / 软件 | MuJoCo（最低可用）    | Genesis / 双后端（推荐）    |
| ----------- | --------------------- | --------------------------- |
| CPU         | 4 核以上              | Intel i5-14600F 或更高      |
| 内存        | 8 GB                  | 16 GB 以上                  |
| GPU         | 非必需（可纯 CPU 跑） | NVIDIA RTX 2080 Ti 或更高   |
| GPU 驱动    | —                    | NVIDIA driver ≥ 535        |
| CUDA        | —                    | ≥ 12.0                     |
| 磁盘        | 5 GB                  | 15 GB 以上（含 torch/CUDA） |

> **说明：** MuJoCo 后端在纯 CPU 上即可运行；**Genesis 后端必须有 NVIDIA GPU**（内部走 CUDA/Taichi）。若只用 MuJoCo，GPU 相关项可忽略。

---

## 安装

只有一种安装方式：在宿主机（或已进入的容器）里跑 `install.sh`。脚本会准备好**全部**依赖，包括 Conda 环境本身。

### 前置条件

只需先装好 ROS 并 source 它 —— 这一步必须在跑安装脚本**之前**完成，否则 ROS 消息无法正确编译：

```bash
# Ubuntu 22.04（ROS2 Humble）
source /opt/ros/humble/setup.bash

# 或 Ubuntu 20.04（ROS1 Noetic）
source /opt/ros/noetic/setup.bash
```

### 一键安装

```bash
# 1. 克隆仓库
git clone https://github.com/Astribot-Dev/astribot_simulation.git
cd astribot_simulation

# 2. 初始化 submodule 并拉取 LFS 网格文件（缺这步 MuJoCo 一定起不来）
git submodule update --init --recursive
git submodule foreach git lfs pull

# 3. 安装（Conda 缺失时脚本会自动装 Miniconda）
bash scripts/install.sh
```

`install.sh` 依次完成：

1. 检查系统 / Python / GPU / CUDA
2. **未检测到 Conda 时自动下载安装 Miniconda**
3. 创建 Conda 环境 **`astribot_simulation`**（Python 3.11，可用 `ASTRIBOT_SIMU_PY_VERSION=3.10` 覆盖）
4. 安装依赖：numpy / mujoco / genesis-world / gymnasium / torch(+CUDA) / opencv / transforms3d 等
5. **编译 `astribot_msgs`**（ROS1 走 cmake/make，ROS2 走 colcon）
6. 以 editable 方式安装本项目

装完即可用，无需再手工跑其它脚本。

### 验证安装

```bash
conda activate astribot_simulation
source /opt/ros/humble/setup.bash
source env.sh
bash scripts/verify_env.sh
```

全绿退出码 0。该脚本校验关键包版本、项目可导入、4 组「机型 × 后端」配置解析正确，并分别拉起 MuJoCo 与 Genesis 做启动冒烟（Genesis 单独进程 + 无头模式）。仅想看版本不跑启动冒烟时加 `--no-launch`。

---

## 快速开始

```bash
# 1. 激活环境（每个新终端都要）
conda activate astribot_simulation
source /opt/ros/humble/setup.bash      # ROS1 用 /opt/ros/noetic/setup.bash

# 2. 设置项目环境变量（导出 ASTRIBOT_SIMU_ROOT / ROBOT_TYPE / PYTHONPATH）
source env.sh                          # 默认 T1；或 --robot s1

# 3. 启动
python3 astribot_simulation.py astribot_t1
```

> **不需要**再单独跑 `scripts/build.sh` —— `install.sh` 已经编译过 ROS 消息。只有在你**修改了 `astribot_msgs/` 下的 `.msg/.srv/.action`**，或 `git clean` 删掉了生成产物之后，才需要重新编译：
>
> ```bash
> bash scripts/build.sh
> ```
>
> 若跳过编译就启动，会报 `ImportError: cannot import name 'RawRequest' from 'astribot_msgs.srv'`。

### 选择机型与后端

```bash
python3 astribot_simulation.py astribot_t1                      # T1 + MuJoCo（默认后端）
python3 astribot_simulation.py astribot_s1                      # S1 + MuJoCo
python3 astribot_simulation.py astribot_t1 --backend genesis     # T1 + Genesis
python3 astribot_simulation.py astribot_s1 --backend genesis     # S1 + Genesis
```

机型省略时回退环境变量 `$ROBOT_TYPE`（由 `env.sh --robot` 导出），命令行参数优先级更高，因此切机型无需重新 source：

```bash
source env.sh                                 # ROBOT_TYPE=T1
python3 astribot_simulation.py astribot_s1    # 直接切到 S1
```

仿真过程中按 **Backspace** 复位机器人状态。

### 发布话题与实时率

本版本**只发布控制相关话题**：各组件的关节状态（`/astribot_*/joint_space_states`）与底盘 IMU
（`/astribot_whole_body/chassis_imu`）。相机、LiDAR、LiDAR 一体 IMU 与力矩传感器均**不支持**，
详见下文[传感器](#传感器)。

这样做是为了优先保证实时率：实测 S1 + MuJoCo 的 RTF≈0.99（sim 时间与墙钟 1:1）。仿真同时会把人眼
窗口渲染节流到约 17Hz，并在跑得比实时快时主动 sleep 到 1:1，避免仿真时间超前墙钟而与 VR 遥操
或真机同步失配。

可用的运行期环境变量：

| 环境变量                      | 作用                                    |
| ----------------------------- | --------------------------------------- |
| `ASTRIBOT_PACE_REALTIME=0`  | 关掉实时节拍锁（离线时尽可能快跑）      |
| `ASTRIBOT_HUMAN_RENDER_HZ`  | 指定人眼窗口帧率（默认约 17Hz）         |
| `ASTRIBOT_MUJOCO_PASSIVE=0` | MuJoCo 回退同步渲染                     |
| `ASTRIBOT_RTF_REPORT_SEC`   | RTF 打印间隔秒数（默认 10，`0` 关闭） |
| `ASTRIBOT_DEBUG=1`          | 打开 DEBUG 级日志                       |

---

## 详细用法

### 编译 ROS 消息（首次运行，或清理 `astribot_msgs/` 后）

`astribot_msgs/` 下的自定义 service / action 类型是从 `*.srv` / `*.action` 文件在编译时生成的，
生成产物位于 `astribot_msgs/install/`，**不提交到仓库**。如果你跳过这步，启动 `astribot_simulation.py` 时会看到：

```
ImportError: cannot import name 'RawRequest' from 'astribot_msgs.srv' (unknown location)
```

每个克隆只需编译一次（或在你添加新消息类型、切换 ROS 版本、`git clean` 清理 `astribot_msgs/{build,install,log}/` 后重新编译）：

```bash
# 必须先激活 ROS 环境和 conda 环境
source /opt/ros/humble/setup.bash             # 或 /opt/ros/noetic/setup.bash for ROS1
conda activate astribot_simulation

# build.sh 会根据当前 ROS 版本符号链接正确的 CMakeLists / package.xml
# 然后运行 colcon build（ROS2）或 cmake/make（ROS1）
bash scripts/build.sh
```

**`install.sh` 已经包含了这一步**，所以全新安装后无需再手动编译。只有修改消息定义或清理后才需要。

### 配置文件查找顺序

每个 `(robot_type, backend)` 对应 `config/astribot_<robot>/` 下的一个 yaml：

| 后端        | YAML 文件      | 说明                                                          |
| ----------- | -------------- | ------------------------------------------------------------- |
| `mujoco`  | `sim.yaml`   | 唯一配置源                                                    |
| `genesis` | `sim.yaml`   | 同一个文件；两后端的差异写在 `backend_overrides.<backend>` 里 |

两个后端共用同一份机型 `sim.yaml`，不再有按后端拆分的独立文件。

---

## 项目结构

```
astribot_simulation/
├── astribot_simulation.py         # 主入口
├── env.sh                         # 环境激活（ROS / ASTRIBOT_SIMU_ROOT / ROBOT_TYPE）
├── pyproject.toml                 # 项目元数据 / 版本
├── requirements.txt               # Python 依赖
│
├── src/
│   ├── astribot_envs/             # 各后端环境封装
│   │   ├── astribot_base_env.py       # 抽象基类（继承 gym.Env）
│   │   ├── astribot_mujoco_env.py     # MuJoCo 后端
│   │   ├── astribot_genesis_env.py    # Genesis 后端
│   │   ├── astribot_envs_factory.py   # 工厂：加载配置、生成 gym env
│   │   └── simulation_constants.py    # 频率推导与常量
│   │
│   ├── simu_utils/                # 通用工具
│   │   ├── simu_common_tools.py       # 姿态、点云、日志
│   │   ├── robot_ros_interface.py     # ROS1/ROS2 收发器
│   │   ├── chassis_kinematics.py      # 全向轮运动学与底盘控制器桩
│   │   ├── config_loader.py           # 六层配置合并
│   │   ├── sim_config.py              # 配置数据类（schema）
│   │   └── sensor_scheduler.py        # 仿真时间驱动的传感器采样
│   │
│   └── sim_assets_tools/          # 网格 / 贴图处理工具
│
├── config/                        # YAML 配置（见下文）
├── astribot_scenes/               # 场景 MJCF（s1 / t1）
├── astribot_descriptions/         # 机器人模型（git submodule + Git LFS）
├── astribot_msgs/                 # 自定义 ROS 消息 / 服务 / 动作
├── scripts/                       # 安装、编译、验证脚本
└── docs/                          # 图片与示意图
```

---

## 架构

`astribot_simulation` 采用工厂模式支持多后端，保持高内聚低耦合。

### 类图

<p align="center">
  <img src="docs/class_diagram.png" alt="Astribot Simulation Class Diagram" width="900">
</p>

### 设计要点

- **统一 API**：所有后端环境继承 `AstribotBaseEnv`，与 `gym.Env` 兼容。
- **工厂分发**：`AstribotEnvsFactory` 按 `--backend` 推导出的 `simulator_type` 动态载入对应后端。
- **仿真到真机的桥梁**：`MultiRobotRosInterface` 让仿真状态与 ROS 消息严格对齐，便于无缝部署到真机。

> **关于其它后端：** 代码里还留有 ManiSkill 与 Isaac Lab 的环境模块，但**尚不成熟，当前版本已屏蔽**，不做安装也不对外暴露。`--backend` 只接受 `mujoco` 与 `genesis`。

---

## 配置文件

### 文件结构

```
config/
├── simulation.yaml            # 公共配置：与机型无关的物理/显示/ROS/控制器默认值
├── astribot_s1/
│   ├── sim.yaml               # S1 机型配置
│   └── sensor_calib.json      # 相机内外参（可选，本版本未启用）
└── astribot_t1/
    └── sim.yaml               # T1 机型配置
```

配置分两层，加载时深合并：公共层放与机型无关的默认值，机型层只写该机型特有的内容
（关节表、底盘几何、场景）。每个字段的含义、可选值与取值范围也以**英文注释**写在 yaml
里，可直接打开文件对照。

### 公共配置（`config/simulation.yaml`）

| 字段                          | 含义                                                        |
| ----------------------------- | ----------------------------------------------------------- |
| `physics.physics_hz`        | 物理步频（默认 200）。**唯一频率源**，`dt = 1/physics_hz` |
| `physics.control_hz`        | 控制频率（默认 50）。`frame_skip = physics_hz / control_hz` 由此推导 |
| `physics.gravity`           | 重力向量，默认 `[0, 0, -9.81]`                            |
| `display.mode`              | `human`（GUI 窗口）/ `rgb_array`（离屏）/ `''`（不渲染） |
| `display.width/height/fps`  | 窗口分辨率与刷新率                                          |
| `ros.domain_id`             | `ROS_DOMAIN_ID`，与 SDK 侧保持一致才能通信                |
| `ros.qos_depth`             | ROS2 QoS 队列深度                                           |
| `controller_config`         | 控制器增益与动力学补偿的**全局默认值**，见下文         |
| `imu_topic` / `lid_topic` / `chassis_imu_topic` | 传感器话题名（占位字段，见「传感器」一节） |
| `sensor_frequencies`        | 各传感器采样率（占位字段）                                  |

> **频率只改这两个字段**：`physics_hz` 与 `control_hz` 是单一数据源，`frame_skip` 与
> `dt` 全部由 `resolve_timing()` 推导，两个后端共用，请勿在别处硬编码。

### 机型配置（`config/<robot>/sim.yaml`）

| 字段                      | 含义                                                              |
| ------------------------- | ----------------------------------------------------------------- |
| `robot_type`            | 机型标识（`s1` / `t1`），与目录名对应                         |
| `robot_name`            | ROS 话题前缀，默认 `astribot_whole_body`                        |
| `scene`                 | 场景名，解析为 `astribot_scenes/<robot>/<scene>/scene.xml`       |
| `simulator_type`        | 后端类（`Mujoco` / `Genesis`），通常由 `--backend` 覆盖       |
| `chassis_type`          | 底盘类型：`omni`（全向轮）                                      |
| `chassis_fixed`         | `true` 时锁死底盘位姿，只调手臂时更稳定                         |
| `omni.*`                | 底盘轮几何：`wheel_radius`、`half_wheelbase`、`half_track` 等 |
| `robot_list`            | 部件列表，决定发布哪些 ROS 话题组                                 |
| `joint_names_list`      | **按部件分组的关节名**，顺序必须与 MJCF actuator 顺序 1:1 对应 |
| `vel_compensation_list` | 各部件是否启用速度前馈                                            |
| `controller_config`     | 该机型的增益与补偿配置（覆盖公共层）                              |
| `object_names`          | 场景中需要发布位姿的自由物体                                      |
| `backend_overrides`     | 按后端覆盖上述任意字段，见下文                                    |

> **`joint_names_list` 顺序是硬约束**：它与 MJCF 的 actuator 顺序一一对应，底盘在最前。
> 顺序错位会导致命令写到错误的关节上。

合并优先级（低 → 高）：

```
SimConfig 默认值 < simulation.yaml < 机型 sim.yaml < backend_overrides.<backend> < 环境变量
```

`backend_overrides` 写在机型 `sim.yaml` 里，只列出该后端与默认值不同的字段。例如 S1：MuJoCo 用 `chassis_model: dynamic`（真实辊子物理）并把 `control_hz` 提到 100；Genesis 用 `chassis_model: kinematic`（虚拟关节，避免高速辊子接触失稳）。

场景不是命令行参数，由机型 `sim.yaml` 的 `scene` 字段决定，解析为 `astribot_scenes/<机型>/<场景>/scene[_kinematic].xml`。

### 控制器配置（`controller_config`）

三级优先级：`per_joint` > `per_component.default_gains` > `default_gains`。

```yaml
controller_config:
  default_gains:                    # 全局兜底
    position: {kp: 100.0, kd: 10.0}
    torque:   {kp: 0.0,   kd: 0.1}

  per_joint:                        # 单关节精确配置（最高优先级）
    astribot_arm_right_joint_1:
      position: {kp: 100.0, kd: 10.0}   # mode=1
      torque:   {kp: 0.0,   kd: 2.0}    # mode=3，kp 必须为 0

  dynamics_compensation:
    enabled: true
    modes: [1, 2, 3]                # 在哪些控制模式下启用补偿
    components:
      gravity: true                 # g(q)
      coriolis: true                # C(q,q̇)·q̇
      inertia: false                # M(q)·q̈（默认关，见「关节空间命令」一节）
      friction: false               # 未实现（模型无库仑摩擦）
```

| 字段                | 含义                                                                 |
| ------------------- | -------------------------------------------------------------------- |
| `position.kp`     | 位置增益 → MuJoCo `gainprm[0]`；同时决定 `biasprm[1] = -kp`      |
| `position.kd`     | 速度阻尼 → MuJoCo `biasprm[2]` 的绝对值。**取值应与 MJCF 一致** |
| `torque.kp`       | mode=3 的位置增益，**必须为 0**（真机语义）                      |
| `torque.kd`       | mode=3 的小阻尼，**需逐关节配置**（真机各关节不同）              |
| `modes`           | 补偿生效的模式列表；不含某模式则该模式无前馈                         |
| `components.*`    | 逐项开关补偿的三个动力学分量                                         |

> **`position.kd` 是速度阻尼，不是位置反馈系数。** MJCF 的 `biasprm` 三个槽位依次是
> 偏置、位置反馈（`-kp`）、速度阻尼（`-kd`）。历史上曾把 kd 误填成 kp 的值，导致阻尼
> 放大 10 倍，现已对齐 MJCF 真值。

> ### 关于传感器字段（重要）
>
> 机型 `sim.yaml` 里的 `camera_names`、`sensor_names`，以及 `simulation.yaml` 里的
> `sensor_frequencies`、`imu_topic`、`lid_topic`，都只是**保持 schema 稳定的占位字段**。
>
> **本版本不支持相机与 LiDAR**，这些字段留空或留默认值并不代表"可以自行填上就能启用"。
> 相关代码路径尚未充分验证，**请勿修改这些字段去尝试开启传感器** —— 那会导致仿真行为不可预期。
> 唯一实际生效的传感器是底盘 IMU（`sensor_names` 里的 `astribot_chassis_base_imu_gyro`）。
>
> 传感器支持会在后续版本正式提供。

---

## 传感器

> **本版本不支持传感器。** 相机、LiDAR、LiDAR 一体 IMU 与力矩传感器的仿真链路尚不成熟，
> 发布版**统一关闭**，且没有提供任何开启开关（CLI 已移除 `--profile`）。请不要尝试通过修改
> 配置来启用 —— 这些代码路径未经充分验证，可能导致仿真不稳定或数据不可信。

| 传感器             | MuJoCo         | Genesis        |
| ------------------ | -------------- | -------------- |
| 相机 RGB           | 不支持         | 不支持         |
| 相机深度           | 不支持         | 不支持         |
| 相机点云           | 不支持         | 不支持         |
| LiDAR 点云         | 不支持         | 不支持         |
| LiDAR 一体 IMU     | 不支持         | 不支持         |
| 力 / 力矩传感器    | 不支持         | 不支持         |
| **底盘 IMU** | **支持** | **支持** |

**唯一可用的传感器话题**是底盘 IMU：

- `/astribot_whole_body/chassis_imu` —— `std_msgs/Float64MultiArray`，9 维
  （rpy、角速度、线加速度），与真机语义对齐，默认 200Hz。

机型 `sim.yaml` 里 `camera_names: []` 与 `sensor_names: [...]` 两个字段作为
**schema 占位**保留，请保持原样不要改动。相机与 LiDAR 支持将在后续版本提供。

## 关节空间命令

如需直接控制关节，机器人对外暴露关节空间命令话题（如 `/astribot_arm_left/joint_space_command`）。命令含义随控制模式不同：

| `mode` | 控制模式 | 命令维度 | 含义 | 前馈补偿 |
| ------ | -------- | -------- | ---- | -------- |
| `1` | 位置控制 | 7–14 | **前 7 维** 目标关节位置；**后 7 维** 目标关节速度（速度前馈） | ✅ |
| `2` | 速度控制 | 7 | 各关节目标速度 | ✅ |
| `3` | 力矩控制（零力拖动 / Zero-G） | 7 | 各关节的**附加**力矩，通常发全 0 | ✅ |

### 动力学前馈补偿

三种模式都会叠加动力学前馈补偿，由 `controller_config.dynamics_compensation` 控制：

```
补偿量 = M(q)·q̈ + C(q, q̇)·q̇ + g(q)
```

所有项都从**实测反馈状态** `q, q̇, q̈` 计算（与真机中间层一致，不使用期望加速度）。
可通过 `components` 逐项开关；`gravity` 与 `coriolis` 默认开启，`inertia` 默认关闭
（`q̈` 是噪声较大的二阶量，且与控制器输出同期耦合，实测开启后收敛变慢）。

### 力矩控制（mode=3）与零力拖动

真机的零力拖动语义是：SDK 下发的力矩命令**直接透传**，中间层叠加动力学前馈，并施加
一个小阻尼项（期望速度为零、**位置增益 kp = 0**）：

```
关节力矩 = 力矩命令 + 动力学补偿 − kd · q̇
```

仿真按同一链路实现，所以 SDK 发全 0 力矩时手臂会**保持当前姿态**而不下坠 —— 这就是
零力拖动：外力推动手臂时它顺势移动，撤去外力后停在原处。

阻尼 `kd` 通过 `controller_config` 的 `torque` 段**逐关节**配置（`kp` 必须为 0）：

```yaml
astribot_arm_right_joint_1:
  position: {kp: 100.0, kd: 10.0}   # mode=1
  torque:   {kp: 0.0,   kd: 2.0}    # mode=3：kp 必须为 0
```

> **注意：** 当前 `torque.kd` 是照真机量级选取的经验值，**未按手感标定**，因此零力拖动
> 的阻尼手感与真机不完全一致。控制链路语义已对齐，但 sim-to-real 的力控精度不作保证。

**提示：** 使用 **Astribot SDK** 时通常无需关心这些细节，参见 SDK 文档。直接发关节命令时，请确保命令维度与控制模式对齐。

---

## 场景文件（`astribot_scenes/`）

`astribot_scenes/` 目录包含按机器人型号和环境类型组织的 MuJoCo MJCF 场景定义。每个场景包含地面、墙壁、光照以及可选的道具（如桌子）。

### 目录结构

```
astribot_scenes/
├── s1/                    # S1 机器人场景
│   ├── floor/                 # 空地面（基础测试）
│   ├── table/                 # 带桌子的房间（操作任务）
│   └── warehouse/             # 仓库环境（SLAM 验证）
└── t1/                    # T1 机器人场景
    ├── floor/
    ├── table/
    └── warehouse/
```

### 场景变体

每个场景目录包含两个 MJCF 文件：

- **`scene.xml`** — **动力学底盘**完整场景（真实轮子、滚轮接触、完整物理）
- **`scene_kinematic.xml`** — **运动学底盘**场景（虚拟关节、仅视觉轮子、更快更稳定）

活动变体通过 `config/<robot>/sim.yaml` 中的 `backend_overrides.mujoco.chassis_model` 选择：

```yaml
backend_overrides:
  mujoco:
    chassis_model: dynamic      # 使用 scene.xml
    # chassis_model: kinematic  # 使用 scene_kinematic.xml
```

### 可用场景

|       机器人 | 场景          | 描述                                    |
| -----------: | ------------- | --------------------------------------- |
| **s1** | `floor`     | 空平地，适合基础移动和手臂测试          |
|              | `table`     | 三面墙房间 + 桌子，用于操作和 SLAM 验证 |
|              | `warehouse` | 30×30m 仓库式布局，含围墙与障碍物，用于 SLAM 巡航验证 |
| **t1** | `floor`     | 空平地                                  |
|              | `table`     | 带桌子的房间                            |
|              | `warehouse` | 同 s1 布局，机器人模型换为 t1           |

### 场景选择

场景通过 `config/<robot>/sim.yaml` 中的 `scene` 参数选择：

```yaml
scene: floor      # 默认：空地面
# scene: table    # 带桌子的房间
# scene: warehouse # 仓库环境（s1 / t1 均支持）
```

场景 MJCF 通过以下方式包含到机器人的顶层 MJCF 中：

```xml
<include file="../../astribot_scenes/<robot>/<scene>/scene.xml"/>
```

### 添加自定义场景

创建新场景的步骤：

1. 在 `astribot_scenes/<robot>/<scene_name>/` 下创建目录
2. 添加 `scene.xml` 和 `scene_kinematic.xml`（从现有场景复制作为模板）
3. 根据需要修改地面、墙壁、道具、光照
4. 更新 `config/<robot>/sim.yaml` 引用你的场景

---

## 机器人描述文件、Submodule 与 Git LFS

`astribot_descriptions/` 是一个 **git submodule**（见 [.gitmodules](.gitmodules)），网格文件用 **Git LFS** 管理。**绝大多数启动失败都源于 submodule 状态错误或 LFS 对象缺失。**

来源与跟踪分支：

| 项目 | 值                                                            |
| ---- | ------------------------------------------------------------- |
| 仓库 | `https://github.com/Astribot-Dev/astribot_descriptions.git` |
| 分支 | `aos`                                                       |
| 传输 | https（**无需 SSH key**）                               |

### 首次设置

```bash
# 1. 初始化 submodule
git submodule update --init --recursive

# 2. 拉取 LFS 对象（这一步才真正下载网格文件）
git submodule foreach git lfs pull
```

如果你已经在没有 LFS 的情况下 clone 过，只需补跑第 2 步 —— 约 130 字节的指针文件会被替换成真正的二进制 STL/DAE。

### 更新 submodule

```bash
# 切到父仓 pin 的 commit（推荐：该 commit 是经过验证的版本）
git submodule update --init --recursive
git submodule foreach git lfs pull

# 或切到跟踪分支 aos 的最新 tip
git submodule update --remote --recursive
git submodule foreach git lfs pull
```

`--remote` 之后若想固化指针，记得在父仓提交新的 submodule SHA。日常使用建议用前者：父仓 pin 的 SHA 是与本版本仿真代码配套验证过的，跟随分支 tip 有可能引入未验证的模型改动。

### 启动前自检

```bash
# 应当输出空——若有输出说明 LFS 没拉全
find astribot_descriptions -name "*.STL" -size -1k | head

# 文件普遍只有 ~130 字节说明只是 LFS 指针：
#   git submodule foreach git lfs pull
```

---

## 脚本

发布版只包含三个脚本：

| 脚本                      | 用途                                                                           |
| ------------------------- | ------------------------------------------------------------------------------ |
| `scripts/install.sh`    | **完整安装**：Miniconda（缺失时）+ Conda 环境 + 全部依赖 + 编译 ROS 消息 |
| `scripts/build.sh`      | **单独重编译 `astribot_msgs`**，仅在改了消息定义后需要                 |
| `scripts/verify_env.sh` | **环境自检**：版本校验 + 配置解析 + 双后端启动冒烟                       |

> Miniconda 已由 `install.sh` 自动安装，不再随仓库分发安装包。

---

## 文档

- 环境封装：[`src/astribot_envs/`](src/astribot_envs/)
- 配置示例：[`config/astribot_s1/`](config/astribot_s1/)、[`config/astribot_t1/`](config/astribot_t1/)
- 工具模块：[`src/simu_utils/`](src/simu_utils/)

---

## 故障排查

| 现象                                             | 原因与解决                                                                           |
| ------------------------------------------------ | ------------------------------------------------------------------------------------ |
| `ImportError: cannot import name 'RawRequest'` | ROS 消息未编译 →`bash scripts/build.sh`                                           |
| MuJoCo 报网格文件找不到 / 模型加载失败           | LFS 未拉取 →`git submodule foreach git lfs pull`                                  |
| `ModuleNotFoundError: rclpy`                   | 激活顺序错了 → 先`conda activate`，再 `source /opt/ros/.../setup.bash`          |
| Genesis 报`no valid context` / 无法启动        | 无 GPU 或无显示 → 设`ASTRIBOT_GENESIS_HEADLESS=1`，或改用 `--backend mujoco`    |
| RTF 明显低于 1.0                                 | 用`ASTRIBOT_HUMAN_RENDER_HZ` 降低窗口帧率，或关闭 GUI 窗口（`display.mode: ''`） |
| 机器人状态异常、话题数据错乱                     | 可能有多个仿真进程抢同一批话题 →`pkill -f astribot_simulation.py` 后重启单实例    |

---

## 贡献

欢迎提交 Issue 与 Pull Request。提交前请确保 `bash scripts/verify_env.sh` 通过。

---

## 许可证

BSD 3-Clause License，见 [LICENSE](LICENSE)。

---

## 致谢

感谢 [MuJoCo](https://mujoco.org/) 与 [Genesis](https://genesis-world.readthedocs.io/) 团队。
