if [ -n "$ZSH_VERSION" ]; then
    export ASTRIBOT_SIMU_ROOT="$(dirname "$(realpath "$0")")"
else
    export ASTRIBOT_SIMU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
    # Install colcon build tools in conda env (system colcon is not available in conda)
    pip install colcon-common-extensions catkin_pkg lark empy==3.3.4 pytz

    # Link the ROS2 site-packages into the conda env via a .pth file.
    # Without this, colcon fails inside a conda env with
    # "ModuleNotFoundError: ament_package" -- the ROS2 build tooling lives in
    # /opt/ros site-packages, which the conda interpreter does not see.
    # install.sh does the same thing; keep both in sync.
    if command -v conda &> /dev/null && [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
        ROS_SITE_A="/opt/ros/$ROS_DISTRO/lib/python3.10/site-packages"
        ROS_SITE_B="/opt/ros/$ROS_DISTRO/local/lib/python3.10/dist-packages"
        if [ -d "$ROS_SITE_A" ] && [ -d "$ROS_SITE_B" ]; then
            CONDA_SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
            {
                echo "$ROS_SITE_A"
                echo "$ROS_SITE_B"
            } > "$CONDA_SITE/zzz_ros2_${ROS_DISTRO}.pth"
            python -c "import ament_package, rosidl_adapter" \
                || { echo "ERROR: ament_package / rosidl_adapter not importable"; exit 1; }
            echo "  Linked ROS2 site-packages into the conda env"
        fi
    fi

    cd "$ASTRIBOT_SIMU_ROOT/astribot_msgs"
    rm -f CMakeLists.txt package.xml
    ln -sf CMakeLists_ros2.txt CMakeLists.txt
    ln -sf package_ros2.xml package.xml
    # Clean stale build artifacts to avoid CMakeCache path conflicts
    rm -rf build install log
    unset AMENT_PREFIX_PATH 2>/dev/null || true
    # Ensure /opt/ros is in CMAKE_PREFIX_PATH and AMENT_PREFIX_PATH
    # (may be overridden by other ROS workspaces in .bashrc)
    if [[ ":$CMAKE_PREFIX_PATH:" != *":/opt/ros/$ROS_DISTRO:"* ]]; then
        export CMAKE_PREFIX_PATH="/opt/ros/$ROS_DISTRO:$CMAKE_PREFIX_PATH"
    fi
    if [[ ":$AMENT_PREFIX_PATH:" != *":/opt/ros/$ROS_DISTRO:"* ]]; then
        export AMENT_PREFIX_PATH="/opt/ros/$ROS_DISTRO:$AMENT_PREFIX_PATH"
    fi
    colcon build 2>&1 | grep -Ev "WARNING:colcon|CMake Warning|Cannot generate a safe runtime|runtime library \[|may be hidden by|Some of these libraries|Call Stack|ament_execute_extensions|rosidl_generate_interfaces|rosidl_generator_py_generate_interfaces|CMakeLists\.txt:49|^---$|^\s*$"
else
    echo "Unknown ROS version: $ROS_VERSION"
    exit 1
fi
