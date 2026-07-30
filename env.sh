# -----------------------------------------------------------------------------
# Astribot simulation -- one-source environment activation.
#
# Usage:
#   source env.sh                  # ROBOT_TYPE=T1 (default)
#   source env.sh --robot t1       # explicit T1
#   source env.sh --robot s1       # S1
#
# What this script does:
#   * exports ASTRIBOT_ON_SIMULATION_MODE=1   (the SDK auto-detects this and switches
#                                        to local interface + ROS2 comm backend)
#   * exports ROBOT_TYPE (uppercase)     (consumed by both the simulator and the
#                                        SDK robot_profile loader)
#   * sets ROS2 DDS defaults             (ROS_DOMAIN_ID, RMW, FastDDS whitelist)
#   * sources astribot_msgs setup        (ROS1 or ROS2, picked by $ROS_VERSION)
#
# Re-sourcing is idempotent: helper functions early-return on second call and
# all exports use ${VAR:-default} so prior user overrides are preserved.
# -----------------------------------------------------------------------------

# -------------------- 1. Argument parsing --------------------
_astribot_sim_robot="t1"
while [ $# -gt 0 ]; do
    case "$1" in
        --robot)
            if [ -z "${2:-}" ]; then
                echo "[env.sh] ERROR: --robot requires a value (s1|t1)" >&2
                unset _astribot_sim_robot
                return 1 2>/dev/null || exit 1
            fi
            _astribot_sim_robot="$2"
            shift 2
            ;;
        --robot=*)
            _astribot_sim_robot="${1#--robot=}"
            shift
            ;;
        *)
            # Unknown args ignored (allows forward compatibility)
            shift
            ;;
    esac
done

# Lowercase for validation, uppercase for export (SDK robot_profile.py uppercases anyway)
_astribot_sim_robot_lc=$(echo "${_astribot_sim_robot}" | tr 'A-Z' 'a-z')
case "${_astribot_sim_robot_lc}" in
    s1|t1) ;;
    *)
        echo "[env.sh] ERROR: unsupported robot type '${_astribot_sim_robot}'. Use s1 or t1." >&2
        unset _astribot_sim_robot _astribot_sim_robot_lc
        return 1 2>/dev/null || exit 1
        ;;
esac
_astribot_sim_robot_uc=$(echo "${_astribot_sim_robot_lc}" | tr 'a-z' 'A-Z')

# -------------------- 2. Core exports --------------------
export ASTRIBOT_ON_SIMULATION_MODE=1
export ROBOT_TYPE="${_astribot_sim_robot_uc}"

# Resolve script root (zsh / bash)
if [ -n "$ZSH_VERSION" ]; then
    export ASTRIBOT_SIMU_ROOT=$(dirname "$0:A")
else
    export ASTRIBOT_SIMU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# PYTHONPATH: simulator package layout
case ":${PYTHONPATH:-}:" in
    *":${ASTRIBOT_SIMU_ROOT}/src:"*) ;;
    *) export PYTHONPATH="${ASTRIBOT_SIMU_ROOT}/src:${PYTHONPATH:-}" ;;
esac
export PYTHONWARNINGS="ignore"

# Mask ~/.local user-site: leftovers there (e.g. numpy 1.22.4) shadow the conda
# environment's own numpy 1.26.4, making matplotlib/genesis fail with
# "requires numpy>=1.23".
# **Only mask it while a conda env is active** -- a conda env already carries every
# dependency, so user-site is unnecessary. The system Python (SDK side, no conda)
# does rely on ~/.local packages such as msgpack, and masking unconditionally would
# break the SDK with ModuleNotFoundError. Hence the $CONDA_PREFIX check.
if [ -n "${CONDA_PREFIX:-}" ]; then
    export PYTHONNOUSERSITE=1
fi

# -------------------- 3. ROS setup --------------------
if [ "$ROS_VERSION" = "1" ]; then
    if [ -n "$ZSH_VERSION" ]; then
        [ -f "${ASTRIBOT_SIMU_ROOT}/astribot_msgs/build/devel/setup.zsh" ] && \
            source "${ASTRIBOT_SIMU_ROOT}/astribot_msgs/build/devel/setup.zsh"
    else
        [ -f "${ASTRIBOT_SIMU_ROOT}/astribot_msgs/build/devel/setup.bash" ] && \
            source "${ASTRIBOT_SIMU_ROOT}/astribot_msgs/build/devel/setup.bash"
    fi
else
    # Default to ROS2 (also covers ROS_VERSION="2" and unset)
    # B-5: PR-8D-fix discovered that env.sh sources the sim-embedded
    # astribot_msgs mirror (${ASTRIBOT_SIMU_ROOT}/astribot_msgs/install/...)
    # but scripts/build.sh in PR-1-era installs the canonical copy to
    # ws/src/astribot_msgs/install (the D3-F source of truth). If the
    # sim-embedded path doesn't exist, fall back to the ws/src install.
    _astribot_msgs_setup=""
    if [ -n "$ZSH_VERSION" ]; then
        if [ -f "${ASTRIBOT_SIMU_ROOT}/astribot_msgs/install/setup.zsh" ]; then
            _astribot_msgs_setup="${ASTRIBOT_SIMU_ROOT}/astribot_msgs/install/setup.zsh"
        fi
    else
        if [ -f "${ASTRIBOT_SIMU_ROOT}/astribot_msgs/install/setup.bash" ]; then
            _astribot_msgs_setup="${ASTRIBOT_SIMU_ROOT}/astribot_msgs/install/setup.bash"
        fi
    fi
    if [ -z "$_astribot_msgs_setup" ]; then
        # Fallback: ws/src/astribot_msgs (D3-F source of truth, installed by
        # `bash scripts/build.sh` in the ws).
        _ws_src_msgs="/home/alexzhu/Code/astribot_ws/src/astribot_msgs/install"
        if [ -n "$ZSH_VERSION" ]; then
            [ -f "${_ws_src_msgs}/setup.zsh" ] && _astribot_msgs_setup="${_ws_src_msgs}/setup.zsh"
        else
            [ -f "${_ws_src_msgs}/setup.bash" ] && _astribot_msgs_setup="${_ws_src_msgs}/setup.bash"
        fi
        [ -n "$_astribot_msgs_setup" ] && \
            echo "[env.sh] B-5 fallback: using ws/src astribot_msgs at ${_ws_src_msgs}"
    fi
    [ -n "$_astribot_msgs_setup" ] && source "$_astribot_msgs_setup"
fi

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-25}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

# -------------------- 4. FastDDS interface whitelist --------------------
# Restrict DDS to the robot subnet (192.168.0.x). Prevents slow discovery when
# WiFi or other interfaces are active. Skipped on the robot itself.
# Mirrors astribot_sdk/env.sh.
#
# The whitelist pins DDS to one specific address, so a file generated on a
# different network silently breaks discovery: publishers still come up, but no
# subscriber ever finds them and `ros2 topic list` looks empty. An already-set
# FASTRTPS_DEFAULT_PROFILES_FILE is therefore validated against the interfaces
# that exist right now, and dropped when it no longer matches.
_astribot_sim_setup_fastdds_whitelist() {
    local robot_ip="192.168.0.10"
    local local_ip
    local_ip=$(ip -4 addr show 2>/dev/null | grep -oP '192\.168\.0\.\d+' | head -1)

    # Discard a stale whitelist: either this host left the 192.168.0.x subnet, or
    # its address changed since the file was written.
    if [ -n "${FASTRTPS_DEFAULT_PROFILES_FILE}" ]; then
        case "${FASTRTPS_DEFAULT_PROFILES_FILE}" in
            */astribot_fastdds_whitelist_*.xml)
                local pinned_ip="${FASTRTPS_DEFAULT_PROFILES_FILE##*whitelist_}"
                pinned_ip="${pinned_ip%.xml}"
                if [ "${pinned_ip}" != "${local_ip}" ]; then
                    echo "[env.sh] WARNING: dropping stale DDS whitelist pinned to" \
                         "${pinned_ip} (current 192.168.0.x address:" \
                         "${local_ip:-none}). A stale whitelist makes topics" \
                         "undiscoverable." >&2
                    unset FASTRTPS_DEFAULT_PROFILES_FILE
                    rm -f "/tmp/astribot_fastdds_whitelist_${pinned_ip}.xml" 2>/dev/null
                else
                    return  # still valid for this host
                fi
                ;;
            *)
                return  # user-supplied profile: leave it alone
                ;;
        esac
    fi

    [ -z "${local_ip}" ] && return
    [ "${local_ip}" = "${robot_ip}" ] && return

    local xml="/tmp/astribot_fastdds_whitelist_${local_ip}.xml"
    cat > "${xml}" <<EOF
<?xml version="1.0" encoding="UTF-8" ?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <transport_descriptors>
    <transport_descriptor>
      <transport_id>CustomUDPTransport</transport_id>
      <type>UDPv4</type>
      <interfaceWhiteList>
        <address>${local_ip}</address>
        <address>127.0.0.1</address>
      </interfaceWhiteList>
    </transport_descriptor>
  </transport_descriptors>
  <participant profile_name="participant_profile" is_default_profile="true">
    <rtps>
      <useBuiltinTransports>false</useBuiltinTransports>
      <userTransports>
        <transport_id>CustomUDPTransport</transport_id>
      </userTransports>
    </rtps>
  </participant>
</profiles>
EOF
    export FASTRTPS_DEFAULT_PROFILES_FILE="${xml}"
}
_astribot_sim_setup_fastdds_whitelist

# -------------------- 5. Summary --------------------
echo "[env.sh] Astribot simulation environment activated"
echo "[env.sh]   ASTRIBOT_SIMU_ROOT:    ${ASTRIBOT_SIMU_ROOT}"
echo "[env.sh]   ROBOT_TYPE:            ${ROBOT_TYPE}"
echo "[env.sh]   ASTRIBOT_ON_SIMULATION_MODE=${ASTRIBOT_ON_SIMULATION_MODE}  (SDK shall be able to connect simulation)"
echo "[env.sh]   ROS_DOMAIN_ID=${ROS_DOMAIN_ID}, RMW=${RMW_IMPLEMENTATION}"
[ -n "${FASTRTPS_DEFAULT_PROFILES_FILE}" ] && \
    echo "[env.sh]   DDS whitelist:         ${FASTRTPS_DEFAULT_PROFILES_FILE}"

# -------------------- 6. Cleanup --------------------
unset _astribot_sim_robot _astribot_sim_robot_lc _astribot_sim_robot_uc
unset -f _astribot_sim_setup_fastdds_whitelist
