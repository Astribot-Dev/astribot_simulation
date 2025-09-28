#!/bin/bash
install_ros() {
    # 2. 添加 ROS 公钥 & 源
    # 添加 ROS 公钥
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg

    UBUNTU_CODENAME=$(lsb_release -cs)
    ARCH=$(dpkg --print-architecture)

    # 写入多个源，apt 会自动选择可用的
    sudo tee /etc/apt/sources.list.d/ros.list > /dev/null <<EOF
deb [arch=${ARCH} signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] https://mirrors.tuna.tsinghua.edu.cn/ros/ubuntu ${UBUNTU_CODENAME} main
deb [arch=${ARCH} signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] https://mirrors.aliyun.com/ros/ubuntu ${UBUNTU_CODENAME} main
deb [arch=${ARCH} signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu ${UBUNTU_CODENAME} main
EOF

    sudo apt update && sudo apt install -y \
        ros-noetic-desktop-full \
        python3-flake8-docstrings \
        python3-pip \
        python3-pytest-cov \
        ros-dev-tools \
        libspdlog-dev \
        libyaml-cpp-dev \
        ros-noetic-hpp-fcl ros-noetic-ruckig ros-noetic-pinocchio ros-noetic-trac-ik ros-noetic-moveit-ros-planning-interface \
        libfmt-dev \
        vim
}

if [ -n "$ZSH_VERSION" ]; then
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

if [ -n "$ZSH_VERSION" ]; then
    export ASTRIBOT_SIMU_ROOT="$(dirname "$(realpath "$0")")"
else
    export ASTRIBOT_SIMU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

source /opt/ros/noetic/setup.bash

if [ $? -eq 0 ]; then
    echo "ROS setup.bash 执行成功"
else
    echo "ROS setup.bash 执行失败"
    install_ros
    source /opt/ros/noetic/setup.bash
fi


if [ -z "$ROS_VERSION" ]; then
    echo "ROS_VERSION environment variable is not set."
    exit 1
fi

if [ "$ROS_VERSION" == "1" ]; then
    cd "$ASTRIBOT_SIMU_ROOT/astribot_msgs"
    pip install catkin_pkg rospkg empy==3.3.4
    catkin_make install
elif [ "$ROS_VERSION" == "2" ]; then
    cd "$ASTRIBOT_SIMU_ROOT/astribot_msgs"
    colcon build --symlink-install
else
    echo "Unknown ROS version: $ROS_VERSION"
    exit 1
fi

#!/bin/bash

sudo apt install python3-pip -y
pip install --upgrade pip setuptools wheel
pip install numpy pyyaml glfw matplotlib distro docutils importlib-metadata pexpect distro requests lxml psutil decorator open3d
pip install gymnasium==1.1.1

pip install mujoco==3.2.5

pip install genesis-world==0.2.1

pip install mani_skill==3.0.0b20

# pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

# upgrade gym
pip install --upgrade gymnasium
pip install "libigl<2.6.0"

# opencv
pip install opencv-python
pip install cv_bridge

# torch
pip install torch==2.6.0
pip install torchvision==0.21.0
pip install torchaudio==2.6.0


if [ "$ROS_VERSION" == "1" ]; then
    echo "Detected ROS1"
    if [ -z "$ASTRIBOT_SIMU_ROOT" ]; then
        echo "ASTRIBOT_SIMU_ROOT environment variable is not set."
        exit 1
    fi
    pip install rospkg
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

echo "Welcome to Astribot Simulation !"

