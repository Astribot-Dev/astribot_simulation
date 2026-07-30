#!/bin/bash

set -e  # Exit on error

# ------------------------------
# Configurable: conda env name and Python version
# ------------------------------
#   ASTRIBOT_SIMU_CONDA_ENV  Conda env name (default: astribot_simulation)
#   ASTRIBOT_SIMU_PY_VERSION Python version  (default: 3.11; allowed: 3.10 / 3.11)
# Override examples:
#   ASTRIBOT_SIMU_CONDA_ENV=astribot_simulation bash scripts/install.sh
#   ASTRIBOT_SIMU_PY_VERSION=3.10 bash scripts/install.sh
ASTRIBOT_SIMU_CONDA_ENV="${ASTRIBOT_SIMU_CONDA_ENV:-astribot_simulation}"
ASTRIBOT_SIMU_PY_VERSION="${ASTRIBOT_SIMU_PY_VERSION:-3.11}"

echo "=========================================="
echo "  Astribot Simulation Installation"
echo "=========================================="

# ------------------------------
# Step 0: System environment check
# ------------------------------
echo ""
echo "Checking system environment..."
echo "----------------------------------------"

# Check OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "  OS:       $PRETTY_NAME"
    if [[ "$ID" != "ubuntu" ]]; then
        echo "  ⚠ Non-Ubuntu OS detected. This project is tested on Ubuntu 20.04/22.04."
    fi
else
    echo "  ⚠ Cannot detect OS version"
fi

# Check Python
if ! command -v python &> /dev/null; then
    if command -v python3 &> /dev/null; then
        echo "  ⚡ 'python' not found. Creating symlink to python3..."
        sudo ln -sf "$(which python3)" /usr/bin/python
        echo "  ✓ Symlink created: /usr/bin/python -> $(which python3)"
    else
        echo "  ❌ Neither 'python' nor 'python3' found. Please install Python 3.10."
        exit 1
    fi
fi
PYTHON_VER=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python:   $PYTHON_VER"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d. -f1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo "  GPU:      $GPU_NAME ($GPU_MEM)"
    echo "  Driver:   $DRIVER_VERSION"

    if [ -n "$DRIVER_MAJOR" ] && [ "$DRIVER_MAJOR" -lt 535 ] 2>/dev/null; then
        echo ""
        echo "  ⚠ WARNING: NVIDIA driver $DRIVER_VERSION < 535 (minimum required)"
        echo "    Update driver: https://www.nvidia.com/Download/index.aspx"
        echo "    MuJoCo will still work, but Genesis may fail."
    fi
else
    echo "  GPU:      ❌ nvidia-smi not found"
    echo ""
    echo "  ⚠ WARNING: No NVIDIA GPU detected. GPU-accelerated simulators will not work."
    echo "    Only MuJoCo (CPU mode) will be available."
    echo "    Install NVIDIA driver: https://www.nvidia.com/Download/index.aspx"
fi

# Check CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | cut -d, -f1)
    echo "  CUDA:     $CUDA_VERSION"
else
    echo "  CUDA:     not found (bundled with PyTorch, may still work)"
fi

echo "----------------------------------------"

# ------------------------------
# Step 1: Set ASTRIBOT_SIMU_ROOT
# ------------------------------
if [ -n "$ZSH_VERSION" ]; then
    export ASTRIBOT_SIMU_ROOT="$(cd "$(dirname "$(realpath "$0")")/.." && pwd)"
    SHELL_RC="$HOME/.zshrc"
else
    export ASTRIBOT_SIMU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    SHELL_RC="$HOME/.bashrc"
fi

echo "ASTRIBOT_SIMU_ROOT: $ASTRIBOT_SIMU_ROOT"
echo ""
echo "Setting up large temporary and pip cache directories..."

TMP_BASE="$ASTRIBOT_SIMU_ROOT/.tmp"

mkdir -p "$TMP_BASE/tmp"
mkdir -p "$TMP_BASE/pip-cache"

export TMPDIR="$TMP_BASE/tmp"
export TEMP="$TMP_BASE/tmp"
export TMP="$TMP_BASE/tmp"

export PIP_CACHE_DIR="$TMP_BASE/pip-cache"
export PIP_NO_INPUT=1

echo "✓ TMPDIR set to: $TMPDIR"
echo "✓ PIP_CACHE_DIR set to: $PIP_CACHE_DIR"

# ------------------------------
# Step 2: Check ROS environment (optional)
# ------------------------------
if [ -n "$ROS_VERSION" ]; then
    if [ "$ROS_VERSION" == "1" ]; then
        echo "✓ Detected ROS1"
    elif [ "$ROS_VERSION" == "2" ]; then
        echo "✓ Detected ROS2 ($ROS_DISTRO)"
    else
        echo "⚠ Unknown ROS version: $ROS_VERSION"
    fi
else
    echo ""
    echo "⚠ ROS not detected. ROS messages will be built when you first source env.sh"
    echo "  To use ROS: source /opt/ros/humble/setup.bash (or noetic)"
fi

# ------------------------------
# Step 3: Setup Python environment
# ------------------------------
if ! command -v conda &> /dev/null; then
    # Auto-install Miniconda. A conda env is how this project isolates its Python
    # deps from the system/ROS interpreters, so it is required rather than optional.
    echo ""
    echo "Conda not found. Installing Miniconda to $HOME/miniconda3 ..."
    MINICONDA_INSTALLER="$TMP_BASE/miniconda.sh"
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    if ! curl -fsSL "$MINICONDA_URL" -o "$MINICONDA_INSTALLER"; then
        echo "❌ Failed to download Miniconda from $MINICONDA_URL"
        echo "   Install it manually, then re-run this script:"
        echo "     wget $MINICONDA_URL -O miniconda.sh && bash miniconda.sh -b"
        exit 1
    fi
    bash "$MINICONDA_INSTALLER" -b -p "$HOME/miniconda3"
    rm -f "$MINICONDA_INSTALLER"
    export PATH="$HOME/miniconda3/bin:$PATH"
    # Make `conda activate` usable in this non-interactive shell and in future ones.
    "$HOME/miniconda3/bin/conda" init bash > /dev/null 2>&1 || true
    echo "✓ Miniconda installed at $HOME/miniconda3"
fi

if ! command -v conda &> /dev/null; then
    echo "❌ conda still not on PATH after installation; aborting."
    exit 1
else
    echo ""
    echo "Conda detected. Setting up environment..."
    eval "$(conda shell.bash hook)"

    # Check if environment already exists
    if conda env list | awk '{print $1}' | grep -qx "$ASTRIBOT_SIMU_CONDA_ENV"; then
        echo "✓ Conda environment '$ASTRIBOT_SIMU_CONDA_ENV' already exists"
        conda activate "$ASTRIBOT_SIMU_CONDA_ENV"
    else
        echo "Creating new Conda environment '$ASTRIBOT_SIMU_CONDA_ENV' with Python $ASTRIBOT_SIMU_PY_VERSION..."
        conda create -n "$ASTRIBOT_SIMU_CONDA_ENV" python="$ASTRIBOT_SIMU_PY_VERSION" -y
        conda activate "$ASTRIBOT_SIMU_CONDA_ENV"
        echo "✓ Conda environment created and activated"
    fi

    # Sanity-check active Python version is in [3.10, 3.12)
    ACTIVE_PY=$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    case "$ACTIVE_PY" in
        3.10|3.11) echo "✓ Active Python: $ACTIVE_PY" ;;
        *) echo "❌ Active Python $ACTIVE_PY is outside [3.10, 3.12). Re-create env with ASTRIBOT_SIMU_PY_VERSION=3.11."; exit 1 ;;
    esac

    # Isolate conda env from external Python path pollution.
    #   - PYTHONNOUSERSITE=1: ignore ~/.local/lib/pythonX.Y/site-packages
    #     (user-site residues like a stray genesis-world 0.2.1 would otherwise
    #     shadow the conda-installed 1.0.0).
    #   - unset PYTHONPATH: ROS2 / astribot_sdk setup.bash injects /opt/ros
    #     and other paths that pull in the system Python's site-packages.
    #     We want this install to be deterministic regardless of what was
    #     sourced earlier in the shell.
    export PYTHONNOUSERSITE=1
    if [ -n "${PYTHONPATH:-}" ]; then
        echo "  ⚠ Unsetting PYTHONPATH for isolation (was: ${PYTHONPATH})"
        unset PYTHONPATH
    fi

    # Verify the conda env Python is actually being used by checking sys.prefix.
    ACTIVE_PREFIX=$(python -c 'import sys; print(sys.prefix)')
    EXPECTED_PREFIX="${CONDA_PREFIX:-/dev/null}"
    if [ "$ACTIVE_PREFIX" != "$EXPECTED_PREFIX" ]; then
        echo "❌ python sys.prefix ($ACTIVE_PREFIX) != CONDA_PREFIX ($EXPECTED_PREFIX)."
        echo "   This usually means PYTHONHOME or shell init order is wrong."
        exit 1
    fi
    echo "✓ Python isolation verified: sys.prefix=$ACTIVE_PREFIX"
fi

# ------------------------------
# Step 4: Install Python packages
# ------------------------------
echo ""
echo "Installing Python packages..."
echo "----------------------------------------"

# Helper: run pip and suppress harmless dependency resolver warnings
# These warnings appear because PYTHONNOUSERSITE=1 hides ~/.local/ packages
# whose metadata is still referenced by conda-installed packages.
# All missing deps are installed in subsequent steps.
pip_install() {
    pip install "$@" 2>&1 | grep -v "ERROR: pip's dependency resolver" | grep -v "which is not installed" | grep -v "which is incompatible"
    return ${PIPESTATUS[0]}
}

echo "[1/6] Upgrading pip, setuptools, wheel..."
pip_install --upgrade pip "setuptools<80" wheel

echo "[2/6] Installing core dependencies..."
# transforms3d backs ROS2's tf_transformations, which astribot_base_env imports for
# quaternion_from_matrix. It used to arrive indirectly as a mani_skill dependency;
# now that ManiSkill is no longer installed it must be requested explicitly, or the
# sim fails at import with ModuleNotFoundError: No module named 'transforms3d'.
pip_install numpy==1.26.4 pyyaml glfw distro docutils importlib-metadata \
    pexpect requests lxml psutil decorator open3d ipywidgets pytz colorama six \
    transforms3d
# Note: Do not install matplotlib here - use system version (3.5.1) to avoid conflicts

echo "[3/6] Installing simulators (gymnasium, mujoco, genesis)..."
# Genesis 1.x dropped Python 3.8/3.9; require 3.10 or 3.11 (gated above).
# Genesis also requires `xacro` for its URDF/Xacro parser path. In ROS2 envs
# this happens to be available via /opt/ros/humble/lib/python3.10/site-packages,
# but in an isolated conda env (PYTHONPATH unset) we must install it from PyPI.
# Pin is pessimistic (<2.0.0) but allows minor upgrades — Genesis 1.x has stayed
# API-stable for our usage (gs.morphs / gs.surfaces / control_dofs_*).
pip_install gymnasium==1.1.1 "mujoco>=3.5.0" "mujoco-lidar>=0.3.0" "genesis-world>=1.2.0,<2.0.0" xacro "libigl<2.6.0" pyglet "pin-pink>=3.1.0"

# Fix numpy version (may be upgraded by dependencies)
pip_install --force-reinstall numpy==1.26.4

echo "[4/6] Installing OpenCV, cv_bridge and PyTorch..."
pip_install opencv-python cv_bridge
pip_install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
echo "  ✓ PyTorch 2.6.0 installed"

pip_install --upgrade --force-reinstall coverage

echo "✓ All Python packages installed"

# ------------------------------
# Step 5: Build ROS messages
# ------------------------------
echo ""
echo "Building ROS messages..."
echo "----------------------------------------"

cd "$ASTRIBOT_SIMU_ROOT/astribot_msgs"

if [ "$ROS_VERSION" == "1" ]; then
    echo "Setting up ROS1 message build..."
    pip_install rospkg

    rm -f CMakeLists.txt package.xml
    ln -sf CMakeLists_ros1.txt CMakeLists.txt
    ln -sf package_ros1.xml package.xml

    mkdir -p build && cd build
    cmake -DPYTHON_EXECUTABLE=$(which python3) ../
    make -j$(nproc)
    echo "✓ ROS1 messages built successfully"

elif [ "$ROS_VERSION" == "2" ]; then
    echo "Setting up ROS2 message build..."

    # Install colcon build tools in conda env (system colcon is not available in conda)
    pip_install colcon-common-extensions catkin_pkg lark empy==3.3.4 pytz

    # ament_package, rosidl_* and friends are only shipped via apt as part of
    # /opt/ros/$ROS_DISTRO/lib/python3.10/site-packages and
    # /opt/ros/$ROS_DISTRO/local/lib/python3.10/dist-packages — they are NOT on
    # PyPI. ROS2 setup.bash adds both to PYTHONPATH; we replicate that here as
    # a .pth file inside the conda env's site-packages so the conda Python
    # picks them up WITHOUT polluting PYTHONPATH (which would shadow conda's
    # own packages, e.g. numpy / torch / Genesis).
    #
    # .pth entries:
    #   - apply only when this conda env is the active interpreter,
    #   - rank AFTER conda env's own site-packages (no shadowing),
    #   - require zero env-var hygiene from the user.
    if command -v conda &> /dev/null && [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
        ROS_SITE_A="/opt/ros/$ROS_DISTRO/lib/python3.10/site-packages"
        ROS_SITE_B="/opt/ros/$ROS_DISTRO/local/lib/python3.10/dist-packages"
        if [ -d "$ROS_SITE_A" ] && [ -d "$ROS_SITE_B" ]; then
            CONDA_SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
            PTH_FILE="$CONDA_SITE/zzz_ros2_${ROS_DISTRO}.pth"
            {
                echo "$ROS_SITE_A"
                echo "$ROS_SITE_B"
            } > "$PTH_FILE"
            echo "  ✓ Linked ROS2 sites to conda env via $PTH_FILE"
            python -c "import ament_package, rosidl_adapter" \
                || { echo "❌ ament_package / rosidl_adapter not importable after .pth link"; exit 1; }
            echo "  ✓ ament_package + rosidl_adapter importable from conda Python"
        else
            echo "❌ ROS2 site dirs not found:"
            echo "   - $ROS_SITE_A : $([ -d "$ROS_SITE_A" ] && echo OK || echo MISSING)"
            echo "   - $ROS_SITE_B : $([ -d "$ROS_SITE_B" ] && echo OK || echo MISSING)"
            echo "   Is ros-$ROS_DISTRO-ros-base installed?"
            exit 1
        fi
    fi

    rm -f CMakeLists.txt package.xml
    ln -sf CMakeLists_ros2.txt CMakeLists.txt
    ln -sf package_ros2.xml package.xml

    # Clean stale build artifacts and env vars to avoid path conflicts
    rm -rf build install log
    unset AMENT_PREFIX_PATH 2>/dev/null || true

    # Ensure /opt/ros/humble is in CMAKE_PREFIX_PATH and AMENT_PREFIX_PATH
    # (may be overridden by other ROS workspaces in .bashrc)
    if [[ ":$CMAKE_PREFIX_PATH:" != *":/opt/ros/$ROS_DISTRO:"* ]]; then
        export CMAKE_PREFIX_PATH="/opt/ros/$ROS_DISTRO:$CMAKE_PREFIX_PATH"
    fi
    if [[ ":$AMENT_PREFIX_PATH:" != *":/opt/ros/$ROS_DISTRO:"* ]]; then
        export AMENT_PREFIX_PATH="/opt/ros/$ROS_DISTRO:$AMENT_PREFIX_PATH"
    fi

    # Build ROS2 messages, suppress harmless CMake warnings about conda libpython
    colcon build 2>&1 | grep -Ev "WARNING:colcon|CMake Warning|Cannot generate a safe runtime|runtime library \[|may be hidden by|Some of these libraries|Call Stack|ament_execute_extensions|rosidl_generate_interfaces|rosidl_generator_py_generate_interfaces|CMakeLists\.txt:49|^---$|^\s*$"
    # Verify build succeeded
    if [ ! -d "install/astribot_msgs/lib/python3.10/site-packages/astribot_msgs/srv" ]; then
        echo "❌ ROS2 message build failed. Re-running without output filter..."
        rm -rf build install log
        colcon build
    fi
    echo "✓ ROS2 messages built successfully"

    # Restore numpy for simulation runtime (build tools may have changed it)
    pip_install --force-reinstall numpy==1.26.4
fi

# ------------------------------
# Step 6: Install astribot_simulation itself (editable / src layout)
# ------------------------------
# pyproject.toml uses [tool.setuptools.packages.find] where=["src"], so
# without `pip install -e .` Python cannot find astribot_envs / simu_utils.
# Editable mode means edits to src/ are picked up without re-installing.
echo ""
echo "[6/6] Installing astribot_simulation (editable, src layout)..."
if command -v conda &> /dev/null && [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    pip_install -e "$ASTRIBOT_SIMU_ROOT" \
        || { echo "❌ pip install -e . failed"; exit 1; }
    # Sanity: importable from a fresh interpreter run
    python -c "from astribot_envs.astribot_envs_factory import AstribotEnvsFactory" \
        && echo "✓ astribot_envs importable" \
        || { echo "❌ astribot_envs not importable after install"; exit 1; }
else
    echo "  (skipped — no conda env active; install with 'pip install -e .' manually)"
fi

# ------------------------------
# Installation Complete
# ------------------------------
echo ""
echo "=========================================="
echo "  Installation Completed Successfully!"
echo "=========================================="
echo ""
if command -v conda &> /dev/null; then
    echo "To start using Astribot Simulation:"
    echo "  1. Open a new terminal or run: source $SHELL_RC"
    echo "  2. Source ROS environment:"
    echo "       source /opt/ros/humble/setup.bash   # ROS2"
    echo "       source /opt/ros/noetic/setup.bash   # ROS1"
    echo "  3. Activate environment: conda activate $ASTRIBOT_SIMU_CONDA_ENV"
    echo "  4. Navigate to: cd $ASTRIBOT_SIMU_ROOT"
    echo "  5. Run simulation: source env.sh && python3 astribot_simulation.py"
else
    echo "To start using Astribot Simulation:"
    echo "  1. Open a new terminal or run: source $SHELL_RC"
    echo "  2. Source ROS environment:"
    echo "       source /opt/ros/humble/setup.bash   # ROS2"
    echo "       source /opt/ros/noetic/setup.bash   # ROS1"
    echo "  3. Navigate to: cd $ASTRIBOT_SIMU_ROOT"
    echo "  4. Run simulation: source env.sh && python3 astribot_simulation.py"
fi
echo ""
echo "Welcome to Astribot Simulation!"
echo ""
