"""Unit tests for simulator imports to verify installations."""
import pytest


class TestSimulatorImports:
    """Verify that simulator backends can be imported correctly."""

    def test_mujoco_import(self):
        """Verify MuJoCo installation."""
        try:
            import mujoco

            assert hasattr(mujoco, "MjModel")
            assert hasattr(mujoco, "MjData")
            assert hasattr(mujoco, "mj_step")
        except ImportError:
            pytest.skip("MuJoCo not installed")

    def test_genesis_import(self):
        """Verify Genesis installation."""
        try:
            import genesis as gs

            assert hasattr(gs, "init")
            assert hasattr(gs, "Scene")
        except ImportError:
            pytest.skip("Genesis not installed")

    def test_maniskill_import(self):
        """Verify ManiSkill + SAPIEN installation."""
        try:
            import mani_skill
            import sapien

            assert hasattr(mani_skill, "envs")
            assert hasattr(sapien, "Scene")
        except ImportError:
            pytest.skip("ManiSkill not installed")

    def test_isaaclab_import(self):
        """Verify Isaac Lab installation."""
        try:
            import isaacsim

            assert hasattr(isaacsim, "SimulationApp")
        except ImportError:
            pytest.skip("Isaac Lab not installed")

    def test_gymnasium_import(self):
        """Verify Gymnasium installation (required)."""
        import gymnasium

        assert hasattr(gymnasium, "make")
        assert hasattr(gymnasium, "Env")

    def test_numpy_import(self):
        """Verify NumPy installation (required)."""
        import numpy as np

        assert hasattr(np, "array")
        assert hasattr(np, "ndarray")

    def test_opencv_import(self):
        """Verify OpenCV installation (required)."""
        import cv2

        assert hasattr(cv2, "imread")
        assert hasattr(cv2, "VideoCapture")

    def test_open3d_import(self):
        """Verify Open3D installation (required for point cloud)."""
        try:
            import open3d as o3d

            assert hasattr(o3d, "geometry")
        except ImportError:
            pytest.skip("Open3D not installed")
