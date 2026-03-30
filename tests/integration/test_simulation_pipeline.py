"""Integration tests for simulation pipeline.

Tests the complete data flow:
  Factory -> create env -> reset -> step -> publish ROS topics

Each backend is tested independently.
Requires: ROS2, corresponding simulator installed.
"""
import pytest
import os
import time


# Skip all tests if ROS2 is not available
ros_version = os.getenv("ROS_VERSION")
pytestmark = pytest.mark.skipif(
    ros_version != "2",
    reason="ROS2 not available",
)


def wait_for_topic(node, topic_name, msg_type, timeout_sec=5.0):
    """Wait for a message on a ROS2 topic. Returns the message or None."""
    received = {"msg": None}

    def callback(msg):
        received["msg"] = msg

    sub = node.create_subscription(msg_type, topic_name, callback, 10)
    start = time.time()
    while received["msg"] is None and (time.time() - start) < timeout_sec:
        import rclpy

        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    return received["msg"]


@pytest.fixture(scope="module")
def mujoco_env():
    """Create MuJoCo environment for testing."""
    try:
        import mujoco  # noqa: F401
    except ImportError:
        pytest.skip("MuJoCo not installed")

    from src.astribot_envs.astribot_envs_factory import AstribotEnvsFactory

    yaml_file = "config/astribot_s1/simulation_mujoco_param.yaml"
    if not os.path.exists(yaml_file):
        pytest.skip("MuJoCo config not found")

    factory = AstribotEnvsFactory()
    config = AstribotEnvsFactory.load_yaml_file(yaml_file)
    # Use rgb_array mode to avoid opening a window
    config["mode"] = "rgb_array"
    _ = factory.create_simulation_env(config)

    yield factory

    # Cleanup
    factory.astribot_simu_env.close()


class TestMujocoPipeline:
    """Test MuJoCo simulation pipeline end-to-end."""

    def test_env_created(self, mujoco_env):
        """Verify environment is created successfully."""
        assert mujoco_env.astribot_simu_env is not None

    def test_joint_states_published(self, mujoco_env):
        """Verify joint states are published on ROS topics.

        Expected topics:
          /<robot_component>/joint_space_states
        """
        from astribot_msgs.msg import RobotJointState

        node = mujoco_env.astribot_simu_env.unwrapped.node
        msg = wait_for_topic(
            node, "/astribot_arm_left/joint_space_states", RobotJointState
        )
        assert msg is not None, "No joint_space_states received on /astribot_arm_left"

    def test_camera_image_published(self, mujoco_env):
        """Verify camera images are published on ROS topics.

        Expected topics (with camera config):
          /astribot_whole_body/camera/<name>/image_raw
        """
        env = mujoco_env.astribot_simu_env.unwrapped
        if not env.camera_names:
            pytest.skip("No cameras configured")

        from sensor_msgs.msg import Image

        node = env.node
        camera_name = env.camera_names[0]
        topic = f"/astribot_whole_body/camera/{camera_name}/image_raw"
        msg = wait_for_topic(node, topic, Image)
        assert msg is not None, f"No image received on {topic}"

    def test_reset_works(self, mujoco_env):
        """Verify reset() returns valid observation."""
        obs, info = mujoco_env.astribot_simu_env.reset()
        assert obs is not None
        assert info is not None

    def test_step_works(self, mujoco_env):
        """Verify step() returns valid 5-tuple."""
        action = mujoco_env.astribot_simu_env.action_space.sample()
        result = mujoco_env.astribot_simu_env.step(action)
        assert len(result) == 5  # obs, reward, terminated, truncated, info


class TestGenesisPipeline:
    """Test Genesis simulation pipeline end-to-end."""

    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            import genesis  # noqa: F401
        except ImportError:
            pytest.skip("Genesis not installed")

    def test_env_creation(self):
        """Verify Genesis environment can be created."""
        from src.astribot_envs.astribot_envs_factory import AstribotEnvsFactory

        yaml_file = "config/astribot_s1/simulation_genesis_param_chassis_fixed.yaml"
        if not os.path.exists(yaml_file):
            pytest.skip("Genesis config not found")

        factory = AstribotEnvsFactory()
        config = AstribotEnvsFactory.load_yaml_file(yaml_file)
        config["mode"] = "rgb_array"
        _ = factory.create_simulation_env(config)

        assert factory.astribot_simu_env is not None
        factory.astribot_simu_env.close()


class TestManiskillPipeline:
    """Test ManiSkill simulation pipeline end-to-end."""

    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            import mani_skill  # noqa: F401
        except ImportError:
            pytest.skip("ManiSkill not installed")

    def test_env_creation(self):
        """Verify ManiSkill environment can be created."""
        from src.astribot_envs.astribot_envs_factory import AstribotEnvsFactory

        yaml_file = "config/astribot_s1/simulation_maniskill_param_chassis_fixed.yaml"
        if not os.path.exists(yaml_file):
            pytest.skip("ManiSkill config not found")

        factory = AstribotEnvsFactory()
        config = AstribotEnvsFactory.load_yaml_file(yaml_file)
        config["mode"] = "rgb_array"
        _ = factory.create_simulation_env(config)

        assert factory.astribot_simu_env is not None
        factory.astribot_simu_env.close()
