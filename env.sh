# 获取当前脚本的目录
if [ -n "$ZSH_VERSION" ]; then
    # Zsh 语法
    export ASTRIBOT_SIMU_ROOT=$(dirname "$0:A")
else
    # Bash 语法
    export ASTRIBOT_SIMU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
PYTHONPATH="$PYTHONPATH":"$ASTRIBOT_SIMU_ROOT"
export PYTHONWARNINGS="ignore"

# 根据 ROS_VERSION 执行不同的命令
if [ "$ROS_VERSION" == "1" ]; then
    # echo "Detected ROS1"
    if [ -z "$ASTRIBOT_SIMU_ROOT" ]; then
        echo "ASTRIBOT_SIMU_ROOT environment variable is not set."
        exit 1
    fi
    if [ -n "$ZSH_VERSION" ]; then
        # Zsh 语法
        source $ASTRIBOT_SIMU_ROOT/astribot_msgs/build/devel/setup.zsh
    else
        # Bash 语法
        source $ASTRIBOT_SIMU_ROOT/astribot_msgs/build/devel/setup.bash
    fi

elif [ "$ROS_VERSION" == "2" ]; then
    # echo "Detected ROS2"
    if [ -z "$ASTRIBOT_SIMU_ROOT" ]; then
        echo "ASTRIBOT_SIMU_ROOT environment variable is not set."
        exit 1
    fi
    if [ -n "$ZSH_VERSION" ]; then
        # Zsh 语法
        source $ASTRIBOT_SIMU_ROOT/astribot_msgs/install/setup.zsh
    else
        # Bash 语法
        source $ASTRIBOT_SIMU_ROOT/astribot_msgs/install/setup.bash
    fi

else
    echo "Unknown ROS version: $ROS_VERSION"
    exit 1
fi


export ROS_DOMAIN_ID=25
