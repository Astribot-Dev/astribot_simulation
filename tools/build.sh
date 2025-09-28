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

if [ "$ROS_VERSION" == "1" ]; then
    echo "Detected ROS1"
    if [ -z "$ASTRIBOT_SIMU_ROOT" ]; then
        echo "ASTRIBOT_SIMU_ROOT environment variable is not set."
        exit 1
    fi
    pip install rospkg
    cd "$ASTRIBOT_SIMU_ROOT/astribot_msgs"
    # rm -rf CMakeLists.txt
    # rm -rf package.xml
    # ln -s CMakeLists_ros1.txt CMakeLists.txt
    # ln -s package_ros1.xml package.xml
    #mkdir -p build && cd build
    #cmake -DPYTHON_EXECUTABLE=/usr/bin/python3 ../
    #make -j8
    catkin_make install
elif [ "$ROS_VERSION" == "2" ]; then
    echo "Detected ROS2"
    if [ -z "$ASTRIBOT_SIMU_ROOT" ]; then
        echo "ASTRIBOT_SIMU_ROOT environment variable is not set."
        exit 1
    fi
    pip install catkin_pkg
    pip install empy
    cd "$ASTRIBOT_SIMU_ROOT/astribot_msgs"
    # rm -rf CMakeLists.txt
    # rm -rf package.xml
    # ln -s CMakeLists_ros2.txt CMakeLists.txt
    # ln -s package_ros2.xml package.xml
    colcon build --symlink-install
else
    echo "Unknown ROS version: $ROS_VERSION"
    exit 1
fi