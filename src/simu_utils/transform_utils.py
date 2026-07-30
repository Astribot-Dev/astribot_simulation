"""Coordinate-transform and quaternion utilities.

Used for camera-extrinsics frame conversions.
"""
import numpy as np


def quaternion_to_rotation_matrix(quat):
    """Convert a quaternion to a rotation matrix.

    Args:
        quat: (w, x, y, z) quaternion

    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ]
    )


def transform_position(local_pos, parent_pos, parent_quat):
    """Convert local coordinates to world coordinates.

    Args:
        local_pos: (x, y, z) local position
        parent_pos: (x, y, z) parent link world position
        parent_quat: (w, x, y, z) parent link world rotation

    Returns:
        (x, y, z) world position
    """
    # Parent link rotation matrix
    R = quaternion_to_rotation_matrix(parent_quat)

    # Local position vector
    local_vec = np.array(local_pos)

    # Rotate, then translate
    world_pos = R @ local_vec + np.array(parent_pos)

    return tuple(world_pos)


def multiply_quaternions(q1, q2):
    """Quaternion multiplication (q1 * q2).

    Args:
        q1: (w, x, y, z) first quaternion
        q2: (w, x, y, z) second quaternion

    Returns:
        (w, x, y, z) resulting quaternion
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,  # w
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,  # x
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,  # y
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,  # z
    )


def quaternion_to_lookat(cam_pos, cam_quat, distance=1.0):
    """Convert a camera quaternion to a lookat point.

    Computes the lookat point `distance` metres in front of the camera, given the
    camera's position and orientation.

    Args:
        cam_pos: (x, y, z) camera world position
        cam_quat: (w, x, y, z) camera world rotation
        distance: distance to the lookat point (metres)

    Returns:
        (x, y, z) world position of the lookat point
    """
    # Camera rotation matrix
    R = quaternion_to_rotation_matrix(cam_quat)

    # Camera forward vector (-Z axis, the OpenGL/MuJoCo convention)
    # OpenGL/MuJoCo: -Z = forward (optical axis), +Y = up, +X = right
    forward = -R @ np.array([0.0, 0.0, 1.0])

    # lookat point = camera position + forward * distance
    lookat = np.array(cam_pos) + forward * distance

    return tuple(lookat)
