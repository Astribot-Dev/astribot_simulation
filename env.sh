# Get the directory of current script
if [ -n "$ZSH_VERSION" ]; then
    # Zsh syntax
    export ASTRIBOT_SIMU_ROOT=$(dirname "$0:A")
else
    # Bash syntax
    export ASTRIBOT_SIMU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
PYTHONPATH="$PYTHONPATH":"$ASTRIBOT_SIMU_ROOT"
export PYTHONWARNINGS="ignore"

# Execute different commands based on ROS_VERSION
if [ "$ROS_VERSION" == "1" ]; then
    # echo "Detected ROS1"
    if [ -z "$ASTRIBOT_SIMU_ROOT" ]; then
        echo "ASTRIBOT_SIMU_ROOT environment variable is not set."
        exit 1
    fi
    if [ -n "$ZSH_VERSION" ]; then
        # Zsh syntax
        source $ASTRIBOT_SIMU_ROOT/astribot_msgs/build/devel/setup.zsh
    else
        # Bash syntax
        source $ASTRIBOT_SIMU_ROOT/astribot_msgs/build/devel/setup.bash
    fi

elif [ "$ROS_VERSION" == "2" ]; then
    # echo "Detected ROS2"
    if [ -z "$ASTRIBOT_SIMU_ROOT" ]; then
        echo "ASTRIBOT_SIMU_ROOT environment variable is not set."
        exit 1
    fi
    if [ -n "$ZSH_VERSION" ]; then
        # Zsh syntax
        source $ASTRIBOT_SIMU_ROOT/astribot_msgs/install/setup.zsh
    else
        # Bash syntax
        source $ASTRIBOT_SIMU_ROOT/astribot_msgs/install/setup.bash
    fi

else
    echo "Unknown ROS version: $ROS_VERSION"
    exit 1
fi


export ROS_DOMAIN_ID=25
