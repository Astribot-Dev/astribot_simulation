#!/bin/bash

set -e  # Exit on error

echo "=========================================="
echo "  Astribot Simulation Installation"
echo "=========================================="

# ------------------------------
# Step 0: Ensure 'python' exists (for container environments)
# ------------------------------
if ! command -v python &> /dev/null; then
    if command -v python3 &> /dev/null; then
        echo "⚡ 'python' not found. Creating symlink to python3..."
        sudo ln -sf "$(which python3)" /usr/bin/python
        echo "✓ Symlink created: /usr/bin/python -> $(which python3)"
    else
        echo "❌ Neither 'python' nor 'python3' found. Please install Python 3.10."
        exit 1
    fi
else
    echo "✓ 'python' command exists"
fi

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
    echo ""
    echo "Conda not found. Using system Python..."
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

    if [ "$PYTHON_MAJOR" -ne 3 ] || [ "$PYTHON_MINOR" -lt 10 ] || [ "$PYTHON_MINOR" -gt 10 ]; then
        echo "❌ Python 3.10 is required. Current version: $PYTHON_VERSION"
        exit 1
    fi
    echo "✓ Python $PYTHON_VERSION detected (compatible)"
else
    echo ""
    echo "Conda detected. Setting up environment..."
    eval "$(conda shell.bash hook)"

    # Check if environment already exists
    if conda env list | grep -q "^astribot_simu "; then
        echo "✓ Conda environment 'astribot_simu' already exists"
        conda activate astribot_simu
    else
        echo "Creating new Conda environment 'astribot_simu' with Python 3.10..."
        conda create -n astribot_simu python=3.10 -y
        conda activate astribot_simu
        echo "✓ Conda environment created and activated"
    fi
fi

# ------------------------------
# Step 4: Install Python packages
# ------------------------------
echo ""
echo "Installing Python packages..."
echo "----------------------------------------"

echo "[1/5] Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

echo "[2/5] Installing core dependencies..."
pip install numpy==1.26.4 pyyaml glfw distro docutils importlib-metadata \
    pexpect requests lxml psutil decorator open3d ipywidgets pytz colorama six
# Note: Do not install matplotlib here - use system version (3.5.1) to avoid conflicts

echo "[3/5] Installing simulators (gymnasium, mujoco, genesis, maniskill)..."
# Install simulators - mani_skill has conflicting gymnasium dependency, so install it separately
pip install gymnasium==1.1.1 mujoco==3.2.5 genesis-world==0.2.1 "libigl<2.6.0" pyglet "pin-pink>=3.1.0"

# Install mani_skill dependencies manually to avoid gymnasium version conflict
echo "  Installing mani_skill dependencies..."
pip install sapien==3.0.0b1 trimesh transforms3d h5py mplib==0.1.1 rtree \
    huggingface-hub fast-kinematics GitPython tabulate dacite pynvml \
    pytorch_kinematics==0.7.5 tyro

# Install mani_skill without dependencies to avoid gymnasium version conflict
pip install --no-deps mani_skill==3.0.0b20

# Fix numpy version (may be upgraded by dependencies)
pip install --force-reinstall numpy==1.26.4

echo "[4/5] Installing OpenCV and cv_bridge..."
pip install opencv-python cv_bridge

echo "[5/5] Installing Isaac Lab and PyTorch..."
# Isaac Lab requires specific PyTorch versions, install carefully to avoid conflicts
if pip show isaaclab &> /dev/null; then
    echo "  Isaac Lab detected, using compatible versions..."

    # Install all compatible versions in one go to avoid conflicts
    # Isaac Lab 2.1.0 requires: torch==2.5.1, pillow==11.0.0, numpy<2
    pip install --force-reinstall \
        torch==2.5.1 \
        torchvision==0.20.1 \
        torchaudio==2.5.1 \
        numpy==1.26.4 \
        pillow==11.0.0

    # Reinstall Isaac Lab to ensure all components are correct
    pip install --no-deps --force-reinstall isaaclab==2.1.0
    pip install isaaclab[isaacsim,all]==2.1.0 --extra-index-url https://pypi.nvidia.com

    echo "  ✓ Isaac Lab compatible versions installed (torch 2.5.1, pillow 11.0.0, numpy 1.26.4)"
else
    echo "  Isaac Lab not found, using latest PyTorch for other simulators..."
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

    echo "  ✓ PyTorch 2.6.0 installed"
fi

pip install --upgrade --force-reinstall coverage

echo "✓ All Python packages installed"

# ------------------------------
# Step 6: Build ROS messages
# ------------------------------
echo ""
echo "Building ROS messages..."
echo "----------------------------------------"

cd "$ASTRIBOT_SIMU_ROOT/astribot_msgs"

if [ "$ROS_VERSION" == "1" ]; then
    echo "Setting up ROS1 message build..."
    pip install rospkg

    rm -f CMakeLists.txt package.xml
    ln -sf CMakeLists_ros1.txt CMakeLists.txt
    ln -sf package_ros1.xml package.xml

    mkdir -p build && cd build
    cmake -DPYTHON_EXECUTABLE=$(which python3) ../
    make -j$(nproc)
    echo "✓ ROS1 messages built successfully"

elif [ "$ROS_VERSION" == "2" ]; then
    echo "Setting up ROS2 message build..."
    pip install numpy==1.22.4
    pip install catkin_pkg lark empy==3.3.4 pytz

    rm -f CMakeLists.txt package.xml
    ln -sf CMakeLists_ros2.txt CMakeLists.txt
    ln -sf package_ros2.xml package.xml

    colcon build
    echo "✓ ROS2 messages built successfully"
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
    echo "  2. Activate environment: conda activate astribot_simu"
    echo "  3. Navigate to: cd $ASTRIBOT_SIMU_ROOT"
    echo "  4. Run simulation: source env.sh && python3 astribot_simulation.py"
else
    echo "To start using Astribot Simulation:"
    echo "  1. Open a new terminal or run: source $SHELL_RC"
    echo "  2. Navigate to: cd $ASTRIBOT_SIMU_ROOT"
    echo "  3. Run simulation: source env.sh && python3 astribot_simulation.py"
fi
echo ""
echo "Welcome to Astribot Simulation!"
echo ""
