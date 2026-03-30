#!/bin/bash

if [ -n "$ZSH_VERSION" ]; then
    export ASTRIBOT_SIMU_ROOT="$(dirname "$(realpath "$0")")"
    SHELL_RC="$HOME/.zshrc"
    echo "export ASTRIBOT_SIMU_ROOT=\"${ASTRIBOT_SIMU_ROOT}\"" >> "$SHELL_RC"
    echo "source ${ASTRIBOT_SIMU_ROOT}/env.sh" >> "$SHELL_RC"
else
    export ASTRIBOT_SIMU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    SHELL_RC="$HOME/.bashrc"
    echo "export ASTRIBOT_SIMU_ROOT=\"${ASTRIBOT_SIMU_ROOT}\"" >> "$SHELL_RC"
    echo "source ${ASTRIBOT_SIMU_ROOT}/env.sh" >> "$SHELL_RC"
fi

if ! command -v conda &> /dev/null; then
    echo "Conda not found. Proceeding with the base Python environment..."
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

    if [ "$PYTHON_MAJOR" -gt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; }; then
        echo "Python version $PYTHON_VERSION detected. Python >= 3.11 is not supported. Exiting."
        exit 1
    else
        echo "Python version $PYTHON_VERSION detected. Proceeding with installation..."
    fi
else
    echo "Conda is installed."
    eval "$(conda shell.bash hook)"
    echo "Conda environment successfully initialized."

    echo "Creating and activating Conda environment..."
    conda create -n astribot_simu python=3.10 -y
    if [ $? -eq 0 ]; then
        conda activate astribot_simu
    else
        echo "Failed to create Conda environment. Exiting."
        exit 1
    fi
fi

sudo apt install python3-pip -y
pip install "setuptools==64.0.0"
pip install imageio
pip install pyyaml glfw matplotlib distro docutils importlib-metadata pexpect distro requests lxml psutil decorator open3d
pip install numpy==1.22.4
pip install gymnasium==1.1.1

# Install mujoco only
pip install mujoco==3.2.5

# opencv
pip install opencv-python
pip install cv_bridge

if [ -z "$ROS_VERSION" ]; then
    echo "ROS_VERSION environment variable is not set."
    exit 1
fi

if [ "$ROS_VERSION" == "1" ]; then
    echo "Detected ROS1"
    if [ -z "$ASTRIBOT_SIMU_ROOT" ]; then
        echo "ASTRIBOT_SIMU_ROOT environment variable is not set."
        exit 1
    fi
    pip install rospkg
    cd "$ASTRIBOT_SIMU_ROOT/astribot_msgs"
    rm -rf CMakeLists.txt
    rm -rf package.xml
    ln -s CMakeLists_ros1.txt CMakeLists.txt
    ln -s package_ros1.xml package.xml
    mkdir -p build && cd build
    cmake -DPYTHON_EXECUTABLE=/usr/bin/python3 ../
    make -j8
elif [ "$ROS_VERSION" == "2" ]; then
    echo "Detected ROS2"
    if [ -z "$ASTRIBOT_SIMU_ROOT" ]; then
        echo "ASTRIBOT_SIMU_ROOT environment variable is not set."
        exit 1
    fi
    pip install catkin_pkg
    pip install lark
    pip install empy==3.3.4
    pip install pytz
    cd "$ASTRIBOT_SIMU_ROOT/astribot_msgs"
    rm -rf CMakeLists.txt
    rm -rf package.xml
    ln -s CMakeLists_ros2.txt CMakeLists.txt
    ln -s package_ros2.xml package.xml
    colcon build
else
    echo "Unknown ROS version: $ROS_VERSION"
    exit 1
fi

if command -v conda &> /dev/null; then
    echo "Installation completed successfully, Please run 
    	
    	'conda activate astribot_simu'
    
    to begin."
else
    echo "Installation completed successfully!"
fi

echo "Welcome to Astribot Simulation (Mujoco Only)!"
