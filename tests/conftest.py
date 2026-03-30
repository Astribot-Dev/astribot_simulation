"""Pytest configuration and shared fixtures."""
import pytest
import yaml
from pathlib import Path


@pytest.fixture
def mock_ros_env(monkeypatch):
    """Mock ROS environment variables."""
    monkeypatch.setenv('ROS_VERSION', '2')
    monkeypatch.setenv('ASTRIBOT_SIMU_ROOT', str(Path(__file__).parent.parent))


@pytest.fixture
def minimal_config():
    """Minimal test configuration."""
    return {
        'simulator_type': 'Mujoco',
        'robot_name': 'test_robot',
        'model_path': '/test/model.xml',
        'robot_list': ['test_arm'],
        'joint_names_list': [['joint_1', 'joint_2']],
        'gravity_compensation': True,
        'mode': 'rgb_array',
        'width': 640,
        'height': 480,
        'object_names': [],
        'camera_names': [],
        'sensor_names': [],
        'vel_compensation_list': [True],
    }


@pytest.fixture
def temp_yaml_config(tmp_path, minimal_config, mock_ros_env):
    """Create temporary YAML config file."""
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(minimal_config, f)
    return str(config_file)
