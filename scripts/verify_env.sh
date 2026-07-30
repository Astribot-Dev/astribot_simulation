#!/bin/bash
# =============================================================================
# Astribot Simulation environment verification
#
# Repeatable self-check: prints and validates key Python package versions, smoke
# imports the project, and launches each supported backend briefly. Run it after
# installing to confirm the environment is correct, or to diagnose version drift.
#
# Usage:
#   conda activate astribot_simulation
#   source /opt/ros/humble/setup.bash        # or noetic, for the launch smoke test
#   source env.sh --robot s1
#   bash scripts/verify/verify_env.sh
#
# Options:
#   --no-launch    Skip the backend launch smoke tests (imports/versions only)
#
# Exit codes: 0 all passed; 1 warnings; 2 fatal errors. Suitable for CI gating.
#
# Note: PYTHONNOUSERSITE=1 is exported below to mask ~/.local user-site packages.
# Leftovers there (e.g. an old numpy) shadow the conda environment's correct
# versions and produce misleading version readings — a trap hit on this project.
# =============================================================================

set -o pipefail
export PYTHONNOUSERSITE=1

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[0;33m'; NC='\033[0m'
EXPECTED_ENV="astribot_simulation"

RUN_LAUNCH=1
for arg in "$@"; do
    case "$arg" in
        --no-launch) RUN_LAUNCH=0 ;;
        -h|--help) sed -n '3,19p' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg (try --help)"; exit 1 ;;
    esac
done

echo "=========================================="
echo "  Astribot Simulation environment check"
echo "=========================================="

PY="$(command -v python3 || command -v python)"
if [ -z "$PY" ]; then
    echo -e "${RED}FAIL python3 not found${NC}"; exit 2
fi
echo "  python:       $PY"
echo "  version:      $($PY --version 2>&1)"
echo "  CONDA_PREFIX: ${CONDA_PREFIX:-(no conda env active)}"

ENV_WARN=0
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo -e "  ${YLW}WARN not inside a conda env; expected '${EXPECTED_ENV}'${NC}"
    ENV_WARN=1
elif [ "$(basename "$CONDA_PREFIX")" != "$EXPECTED_ENV" ]; then
    echo -e "  ${YLW}WARN active env is '$(basename "$CONDA_PREFIX")', expected '${EXPECTED_ENV}'${NC}"
    ENV_WARN=1
fi
echo "  ASTRIBOT_SIMU_ROOT: ${ASTRIBOT_SIMU_ROOT:-(unset — source env.sh)}"
echo "  ROS_VERSION:        ${ROS_VERSION:-(unset)}"
echo "------------------------------------------"

# Version checks run in one embedded python process: faster and consistent.
"$PY" - <<'PYEOF'
import importlib
import sys

fail = 0
warn = 0

RED = "\033[0;31m"
GRN = "\033[0;32m"
YLW = "\033[0;33m"
NC = "\033[0m"


def chk(mod, want=None, cmp="ge", label=None, optional=False):
    """Import mod, print its version, and validate against want/cmp."""
    global fail, warn
    label = label or mod
    try:
        m = importlib.import_module(mod)
    except Exception as e:
        if optional:
            print(f"  {YLW}WARN {label}: not installed ({e}){NC}")
            warn += 1
        else:
            print(f"  {RED}FAIL {label}: import failed — {e}{NC}")
            fail += 1
        return None
    ver = getattr(m, "__version__", "?")
    ok = True
    if want is not None and ver != "?":
        def parse(v):
            return [int(x) for x in str(v).split("+")[0].split(".")[:3] if x.isdigit()]
        if cmp == "eq":
            ok = str(ver).split("+")[0] == want
        else:
            ok = parse(ver) >= parse(want)
    mark = f"{GRN}OK  {NC}" if ok else f"{YLW}WARN{NC}"
    extra = "" if want is None else f" (need {cmp} {want})"
    print(f"  {mark} {label}: {ver}{extra}")
    if want is not None and not ok:
        warn += 1
    return m


print("Core dependencies:")
chk("numpy", want="1.26.4", cmp="eq")
chk("gymnasium", want="1.1.1", cmp="eq")

print("Simulation backends:")
chk("mujoco", want="3.5.0", cmp="ge")
chk("genesis", want="1.2.0", cmp="ge", label="genesis-world")
chk("mujoco_lidar", want="0.3.0", cmp="ge", label="mujoco-lidar", optional=True)

print("Torch / GPU:")
t = chk("torch", want="2.6.0", cmp="ge")
if t is not None:
    try:
        avail = t.cuda.is_available()
        mark = f"{GRN}OK  {NC}" if avail else f"{YLW}WARN{NC}"
        print(f"  {mark} CUDA available: {avail} (cuda {t.version.cuda})")
        if not avail:
            warn += 1
    except Exception as e:
        print(f"  {YLW}WARN CUDA probe failed: {e}{NC}")
        warn += 1

print("Support libraries:")
for mod, lbl in [("open3d", "open3d"), ("cv2", "opencv-python"), ("scipy", "scipy")]:
    chk(mod, label=lbl, optional=True)

print("-" * 42)
print("Project import:")
try:
    import os
    sys.path.insert(0, os.path.join(os.getenv("ASTRIBOT_SIMU_ROOT", "."), "src"))
    from astribot_envs.astribot_envs_factory import AstribotEnvsFactory  # noqa: F401
    print(f"  {GRN}OK  {NC} astribot_envs.astribot_envs_factory imports")
except Exception as e:
    print(f"  {YLW}WARN astribot_envs import failed: {e}{NC}")
    print("       (source env.sh first to set PYTHONPATH / ASTRIBOT_SIMU_ROOT)")
    warn += 1

print("Config resolution:")
try:
    from simu_utils.config_loader import load_config
    import os
    for robot in ("astribot_s1", "astribot_t1"):
        internal = robot.replace("astribot_", "")
        for backend in ("mujoco", "genesis"):
            data, _ = load_config(internal, backend=backend, verbose=False)
            scene_ok = os.path.exists(data["model_path"])
            mark = f"{GRN}OK  {NC}" if scene_ok else f"{RED}FAIL{NC}"
            if not scene_ok:
                fail += 1
            print(f"  {mark} {robot}/{backend}: scene={data.get('scene')} "
                  f"chassis={data.get('chassis_model')}")
except Exception as e:
    print(f"  {YLW}WARN config resolution failed: {e}{NC}")
    warn += 1

print("=" * 42)
if fail:
    print(f"{RED}FAILED: {fail} error(s), {warn} warning(s){NC}")
    sys.exit(2)
if warn:
    print(f"{YLW}PASSED WITH WARNINGS: {warn} warning(s){NC}")
    sys.exit(1)
print(f"{GRN}ALL CHECKS PASSED{NC}")
sys.exit(0)
PYEOF
PY_STATUS=$?

# -----------------------------------------------------------------------------
# Backend launch smoke tests.
#
# Each backend runs in its OWN process, on purpose: genesis.init() is a
# process-level singleton and Genesis EGL init fails if MuJoCo has already taken
# the GL context in the same process. Launching them separately is the only
# reliable ordering.
#
# The sim has no self-terminating mode, so each launch runs under `timeout` and
# an exit status of 124 (timed out while still running) is the success signal.
# -----------------------------------------------------------------------------
LAUNCH_WARN=0
if [ "$RUN_LAUNCH" == "1" ]; then
    echo "------------------------------------------"
    echo "Backend launch smoke tests (separate processes):"
    if [ -z "${ASTRIBOT_SIMU_ROOT:-}" ]; then
        echo -e "  ${YLW}WARN skipped: ASTRIBOT_SIMU_ROOT unset (source env.sh first)${NC}"
        LAUNCH_WARN=1
    else
        for backend in mujoco genesis; do
            LOG="$(mktemp)"
            # Genesis runs headless here: its viewer needs a real GL context, which
            # is often absent over SSH / in CI, and the failure looks like
            # "Attempt to retrieve context when no valid context".
            if [ "$backend" == "genesis" ]; then
                ASTRIBOT_GENESIS_HEADLESS=1 MUJOCO_GL="${MUJOCO_GL:-egl}" timeout 40 "$PY" \
                    "$ASTRIBOT_SIMU_ROOT/astribot_simulation.py" astribot_s1 \
                    --backend "$backend" >"$LOG" 2>&1
            else
                MUJOCO_GL="${MUJOCO_GL:-egl}" timeout 25 "$PY" \
                    "$ASTRIBOT_SIMU_ROOT/astribot_simulation.py" astribot_s1 \
                    --backend "$backend" >"$LOG" 2>&1
            fi
            rc=$?
            # 124 = still running when timeout fired, i.e. it started fine. The
            # banner confirms the backend actually reached its stepping loop.
            if [ "$rc" == "124" ] && grep -q "backend=$backend" "$LOG"; then
                banner=$(grep -o "backend=$backend[^\"]*" "$LOG" | head -1 | cut -c1-70)
                echo -e "  ${GRN}OK  ${NC} $backend launched and stepped"
                echo "        $banner"
            else
                echo -e "  ${YLW}WARN $backend launch smoke failed (exit=$rc)${NC}"
                grep -iE "error|traceback" "$LOG" | head -3 | sed 's/^/        /'
                LAUNCH_WARN=1
            fi
            rm -f "$LOG"
        done
    fi
else
    echo "  (backend launch smoke tests skipped: --no-launch)"
fi

echo "=========================================="
if [ "$PY_STATUS" -ge 2 ]; then
    echo -e "${RED}RESULT: FAILED — fix the errors above${NC}"
    exit 2
fi
if [ "$PY_STATUS" == "1" ] || [ "$LAUNCH_WARN" == "1" ] || [ "$ENV_WARN" == "1" ]; then
    echo -e "${YLW}RESULT: PASSED WITH WARNINGS${NC}"
    exit 1
fi
echo -e "${GRN}RESULT: ENVIRONMENT OK${NC}"
exit 0
