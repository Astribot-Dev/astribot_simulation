# 获取当前脚本的目录
if [ -n "$ZSH_VERSION" ]; then
    # Zsh 语法
    export ASTRIBOT_SIMU_ROOT=$(dirname "$0:A")
else
    # Bash 语法
    export ASTRIBOT_SIMU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
PYTHONPATH="$PYTHONPATH":"$ASTRIBOT_SIMU_ROOT"
# export PYTHONPATH="$PYTHONPATH:$ASTRIBOT_SIMU_ROOT:$ASTRIBOT_SIMU_ROOT/third_party"

source $ASTRIBOT_SIMU_ROOT/astribot_msgs/install/setup.bash
