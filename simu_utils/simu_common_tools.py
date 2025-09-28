#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025, Astribot Co., Ltd.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------
# Author: Tony Wang, tonywang@astribot.com
# -----------------------------------------------------------------------------

import os
import math
import numpy as np

import open3d as o3d

ros_version = os.getenv('ROS_VERSION')
if ros_version=='1':
    import rospy
elif ros_version=='2':
    import rclpy

def astribot_simu_log(message, command=None, level="INFO"):
    prefix = "ASTRIBOT SIMULATION"
    full_message = f"{prefix} - {message}"
    if command:
        full_message += f": {command}"

    color_map = {
        "DEBUG": "",
        "INFO": "\033[0;32m",
        "WARN": "\033[0;33m",
        "ERROR": "\033[0;31m",
        "FATAL": "\033[0;31m",
    }

    full_message_colored = f"{color_map.get(level, '')}{full_message}\033[0m"

    if ros_version == '1':
        level_func_map = {
            "DEBUG": rospy.logdebug,
            "INFO": rospy.loginfo,
            "WARN": rospy.logwarn,
            "ERROR": rospy.logerr,
            "FATAL": rospy.logfatal,
        }
        level_func_map.get(level, rospy.loginfo)(full_message_colored)
    elif ros_version == '2':
        logger = rclpy.logging.get_logger('astribot_logger')
        level_func_map = {
            "DEBUG": logger.debug,
            "INFO": logger.info,
            "WARN": logger.warn,
            "ERROR": logger.error,
            "FATAL": logger.fatal,
        }
        level_func_map.get(level, logger.info)(full_message_colored)

class SimuCommonTools(object):

    def __init__():
        pass
    
    @staticmethod
    def pose_to_matrix(pose):
        x, y, z, qw, qx, qy, qz = pose
        translation = np.array([x, y, z])
        rotation_matrix = np.array([
                [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)]
            ])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation_matrix
        matrix[:3, 3] = translation
        return matrix
    
    @staticmethod
    def matrix_to_pose(matrix):
        translation = matrix[:3, 3]
        rotation_matrix = matrix[:3, :3]

        trace = np.trace(rotation_matrix)

        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        elif (rotation_matrix[0, 0] > rotation_matrix[1, 1]) and (rotation_matrix[0, 0] > rotation_matrix[2, 2]):
            s = np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2  # s = 4 * qx
            qw = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            qx = 0.25 * s
            qy = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            qz = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
            s = np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2  # s = 4 * qy
            qw = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            qx = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            qy = 0.25 * s
            qz = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2  # s = 4 * qz
            qw = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
            qx = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            qy = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            qz = 0.25 * s

        return np.concatenate((translation, [qw, qx, qy, qz]))
    
    @staticmethod
    def pose_add(pose1, pose2, inv_1_flag = False, inv_2_flag = False):

        matrix1 = SimuCommonTools.pose_to_matrix(pose1)
        matrix2 = SimuCommonTools.pose_to_matrix(pose2)
        if inv_1_flag:
            matrix1 = np.linalg.inv(matrix1)
        if inv_2_flag:
            matrix2 = np.linalg.inv(matrix2)

        combined_matrix = np.dot(matrix1, matrix2)
        return SimuCommonTools.matrix_to_pose(combined_matrix)
    
    @staticmethod
    def trans_depth_image_to_point_cloud(depth_img, height, width, fovy, near, far, transform):
        real_depth = near / (1 - depth_img * (1 - near / far))

        f = height / (2 * math.tan(fovy / 2))
        od_depth = o3d.geometry.Image(real_depth)
        od_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, f, f, width / 2, height / 2)
        o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(od_depth, od_camera_intrinsic)
        transformed_cloud = o3d_cloud.transform(transform)
        return transformed_cloud
    
    @staticmethod
    def insert_values(original_list, index, values):
        original_list[index:index] = values
        return original_list
