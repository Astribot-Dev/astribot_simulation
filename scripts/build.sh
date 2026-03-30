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
    pip install catkin_pkg
    pip install empy
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
