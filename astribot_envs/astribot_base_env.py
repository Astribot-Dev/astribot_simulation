#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025, Astribot Co., Ltd.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------
# Author: Tony Wang, tonywang@astribot.com
# -----------------------------------------------------------------------------

"""
File: astribot_base_env.py
Brief: Base env for simulation
"""

import os
import copy
import time
import numpy as np
import math
from abc import ABC, abstractmethod

import gymnasium as gym
from gymnasium import spaces

from cv_bridge import CvBridge, CvBridgeError
import open3d as o3d

from simu_utils.simu_common_tools import SimuCommonTools, astribot_simu_log
from simu_utils.robot_ros_interface import MultiRobotRosInterface

from geometry_msgs.msg import WrenchStamped, Quaternion
from sensor_msgs.msg import Imu, PointCloud2, PointField
from std_msgs.msg import Header

ros_version = os.getenv('ROS_VERSION')
if ros_version=='1':
    import rospy
    from tf.transformations import quaternion_from_matrix
elif ros_version=='2':
    import rclpy
    from rclpy.node import Node
    from tf_transformations import quaternion_from_matrix
    from rclpy.executors import MultiThreadedExecutor

class AstribotBaseEnv(gym.Env, ABC):
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, param):

        astribot_simu_log("Setup param from yaml")
        self.robot_name = param.get('robot_name', '')
        self.model_path = param.get('model_path', '')
        self.robot_list = param.get('robot_list', [])
        self.joint_names_list = param.get('joint_names_list', [])
        self.gravity_compensation = param.get('gravity_compensation', True)
        self.render_mode = param.get('mode', '')
        self.width = param.get('width', 0)
        self.height = param.get('height', 0)
        self.object_names = param.get('object_names', [])
        self.camera_names = param.get('camera_names', [])
        self.sensor_names = param.get('sensor_names', [])
        self.vel_compensation_list = param.get('vel_compensation_list', [])
        self.update_trajectory_map = param.get('update_trajectory_map', {})

        self.setup_ros_node()

        self.reset_flag=False
        self.frame_skip = 3
        self.vel_compensation_map = dict()
        self.robot_joint_map = dict()
        self.cv_bridge = CvBridge()
        self.robot_dict = dict()

        for i in range(len(self.robot_list)):
            self.robot_joint_map[self.robot_list[i]] = copy.deepcopy(self.joint_names_list[i])
            self.vel_compensation_map[self.robot_list[i]] = self.vel_compensation_list[i]

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64)
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(7,), dtype=np.float64)

        self.setup_joint_interface()

        self.real_time_fps=50

    @abstractmethod
    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    @abstractmethod
    def step(self, action: np.ndarray) -> tuple:
        observation = self._get_obs()
        reward = 0
        terminated = False
        info = self._get_info()

        return observation, reward, terminated, False, info
    
    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        return np.zeros(7)
    
    @abstractmethod
    def _get_info(self) -> dict:
        return {}

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def close(self):
        pass
    
    @abstractmethod
    def get_joint_positions(self, names):
        pass

    @abstractmethod
    def get_joint_velocities(self, names):
        pass

    @abstractmethod
    def get_joint_accelerations(self, names):
        pass

    @abstractmethod
    def get_joint_torques(self, names):
        pass

    @abstractmethod
    def get_site_pose(self, site_name: str):
        pass

    @abstractmethod 
    def get_body_pose(self):
        pass

    @abstractmethod 
    def get_chassis_pose(self):
        pass

    def setup_ros_node(self):
        astribot_simu_log("Init ros node",self.robot_name)
        if ros_version=='1':
            rospy.init_node(self.robot_name)
            self.node=None
        elif ros_version=='2':
            rclpy.init()
            self.node = Node(self.robot_name)

    def setup_joint_interface(self):
        astribot_simu_log("Setup joint interface")
        self.multi_robot_ros_interface = MultiRobotRosInterface(self.node)
        self.robot_dict = self.multi_robot_ros_interface.setup_robot_joint_interface(self.robot_joint_map)
        self.multi_robot_ros_interface.setup_camera_interface(self.camera_names)
        self.multi_robot_ros_interface.setup_sensor_interface(self.sensor_names)
        self.multi_robot_ros_interface.setup_trajectory_and_com_psoe(self.update_trajectory_map)

    def update_reset_flag(self):
        press_status = self.get_reset_status()
        
        self.reset_flag = self.multi_robot_ros_interface.get_reset_flag() or press_status
        if self.reset_flag:
            self.multi_robot_ros_interface.reset_flag=False
            press_status=False
            for robot_name in self.robot_list:
                if self.robot_dict[robot_name].simu_running:
                    self.reset_time=time.time()
                    self.robot_dict[robot_name].joint_position_command.clear()
                    self.robot_dict[robot_name].joint_position_command = [0] * self.robot_dict[robot_name].dof
                    self.robot_dict[robot_name].joint_velocity_command.clear()
                    self.robot_dict[robot_name].joint_velocity_command = [0] * self.robot_dict[robot_name].dof
                    self.robot_dict[robot_name].joint_torque_command.clear()
                    self.robot_dict[robot_name].joint_torque_command = [0] * self.robot_dict[robot_name].dof
                else:
                    self.reset_flag=False
                    astribot_simu_log("Resetting is not supported while following the real robot")

    def update_object_states(self):
        for object_name in self.object_names:
            pose = self.get_body_pose(object_name)
            if pose is None:
                continue

            pose = self.pose_add(self.get_chassis_pose(), pose, inv_1_flag=True)
            
            self.multi_robot_ros_interface.publish_object_pose(pose, object_name)

        pose_tuple_list=self.multi_robot_ros_interface.get_object_pose_list()
        
        for pose_tuple in pose_tuple_list:
            if len(pose_tuple) > 0 and pose_tuple[2] in self.object_names:
                object_pose = self.pose_add(self.get_chassis_pose(), pose_tuple[0])
                if pose_tuple[3]:
                    self.set_body_pose(pose_tuple[2], pose=pose_tuple[0], twist=pose_tuple[1])
                    pose_list = list(pose_tuple) 
                    pose_list[3] = False  
                    pose_tuple = tuple(pose_list) 
        self.multi_robot_ros_interface.pose_tuple_list.clear()

        return pose_tuple_list

    def update_camera_data(self):
        camera_data_list=[]
        for camera_name in self.camera_names:
            camera_data = self.get_camera_image(camera_name=camera_name)
            if camera_data and 'rgb_img' in camera_data:

                rgb = camera_data['rgb_img']
                if isinstance(rgb, np.ndarray) and rgb.size > 0:
                    rgb_msg = self.cv_bridge.cv2_to_imgmsg(rgb, "rgb8")
                    rgb_msg.header.stamp = self.multi_robot_ros_interface.get_timestamp()
                    rgb_msg.header.frame_id = 'simulation'
                    self.multi_robot_ros_interface.camera_raw_ros_pub[camera_name].publish(rgb_msg)

                depth = camera_data['depth_img']
                if isinstance(depth, np.ndarray) and depth.size > 0:
                    depth_msg = self.cv_bridge.cv2_to_imgmsg(depth, "32FC1")
                    depth_msg.header.stamp = self.multi_robot_ros_interface.get_timestamp()
                    depth_msg.header.frame_id = 'simulation'
                    self.multi_robot_ros_interface.camera_depth_ros_pub[camera_name].publish(depth_msg)
                
                point_cloud = camera_data['point_cloud']
                if point_cloud is not None:

                    if ros_version=='1':
                        point_cloud_msg = PointCloud2()
                        point_cloud_msg.header.stamp = self.multi_robot_ros_interface.get_timestamp()
                        point_cloud_msg.header.frame_id = 'simulation'
                        point_cloud_msg.height = 1
                        point_cloud_msg.width = len(point_cloud.points)

                        point_cloud_msg.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                                            PointField('y', 4, PointField.FLOAT32, 1),
                                            PointField('z', 8, PointField.FLOAT32, 1),
                                            ]
                        point_cloud_msg.is_bigendian = False
                        point_cloud_msg.point_step = 12 
                        point_cloud_msg.row_step = point_cloud_msg.point_step * point_cloud_msg.width
                        point_cloud_msg.is_dense = False
                        point_cloud_msg.data = np.asarray(point_cloud.points).astype(np.float32).tostring()
                    elif ros_version=='2':

                        point_cloud_msg = PointCloud2()
                        point_cloud_msg.header = Header()
                        point_cloud_msg.header.stamp = self.multi_robot_ros_interface.get_timestamp()
                        point_cloud_msg.header.frame_id = 'simulation'
                        point_cloud_msg.height = 1
                        point_cloud_msg.width = len(point_cloud.points)
                        point_cloud_msg.fields = [
                            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                        ]
                        point_cloud_msg.is_bigendian = False
                        point_cloud_msg.point_step = 12  
                        point_cloud_msg.row_step = point_cloud_msg.point_step * point_cloud_msg.width
                        point_cloud_msg.is_dense = False
                        point_cloud_msg.data = np.asarray(point_cloud.points, dtype=np.float32).tobytes()
    
                    self.multi_robot_ros_interface.camera_point_cloud_ros_pub[camera_name] .publish(point_cloud_msg)

        return camera_data_list

    def update_joint_states(self):
        joint_position_command_all = list()
        joint_velocity_command_all = list()
        joint_torque_command_all = list()
        
        controller_mode_list = []
        joint_names_all = []

        for robot_name in self.robot_list:
            joint_names = self.robot_joint_map[robot_name]
            
            if 'gripper' in robot_name and len(joint_names) > 1: 
                joint_names_all += joint_names[0]
                del joint_names[1:]
            else:
                joint_names_all += joint_names

                pub_sensor_data = self.get_ft_sensor_data(robot_name)
                if pub_sensor_data is not None:
                    self.robot_dict[robot_name].publish_force_torque_sensor(pub_sensor_data)
                
            self.robot_dict[robot_name].publish_joint_states(self.get_joint_positions(joint_names),
                                                                self.get_joint_velocities(joint_names),
                                                                self.get_joint_accelerations(joint_names),
                                                                self.get_joint_torques(joint_names))
            if 'chassis' in robot_name:
                pose = self.get_site_pose('chassis')
                self.robot_dict[robot_name].publish_chassis_pose(pose)

            joint_position_command = self.robot_dict[robot_name].get_joint_position_command()
            joint_position_command_all += joint_position_command
            if self.vel_compensation_map[robot_name]:
                joint_velocity_command = self.robot_dict[robot_name].get_joint_velocity_command()
                joint_velocity_command_all += joint_velocity_command
            else:
                joint_velocity_command = [0.0 for i in range(len(joint_position_command))]
                joint_velocity_command_all += joint_velocity_command

            joint_torque_command = self.robot_dict[robot_name].get_joint_torque_command()
            joint_torque_command_all += joint_torque_command

            controller_mode_list += [self.robot_dict[robot_name].get_controller_mode()] * len(joint_position_command)

        return joint_names_all,controller_mode_list, joint_position_command_all, joint_velocity_command_all, joint_torque_command_all

    def update_trajectory_pose(self, trajectory=False):
        if self.update_trajectory_map != None:
            for robot_name, trajectory_pose_names in self.update_trajectory_map.items():

                if trajectory is False:
                    pose = self.robot_dict[robot_name].get_endpoint_desired_pose()
                    last_pose = self.multi_robot_ros_interface.trajectory_pose_dict[robot_name][-1]

                    err = np.linalg.norm(np.array(pose[:3]) - np.array(last_pose[:3]))
                    last_update_time = self.get_time() - self.last_pose_time
                    if err > 0.02 or last_update_time > 0.5:
                        self.multi_robot_ros_interface.trajectory_pose_dict[robot_name].pop(0)
                        self.multi_robot_ros_interface.trajectory_pose_dict[robot_name].append(pose)

                        self.last_pose_time = self.get_time()
                        for i in range(len(trajectory_pose_names)):
                            self.set_mocap_pose_with_id(i, self.multi_robot_ros_interface.trajectory_pose_dict[robot_name][i])
                else:
                    for i in range(len(trajectory_pose_names)):
                        self.set_mocap_pose_with_id(i, self.multi_robot_ros_interface.trajectory_pose_dict[robot_name][i])

    def update_com_pose(self):
        self.com_pos = copy.deepcopy(self.multi_robot_ros_interface.com_pos)
        self.com_pos = self.pose_add(self.get_chassis_pose(), self.com_pos + [1, 0, 0, 0])[0:3]

    def update_sensor_states(self):
        if len(self.sensor_names) > 0:
            for sensor_name in self.sensor_names:
                sensor_data=self.get_sensor_data(sensor_name)
                if 'force' in sensor_name and len(sensor_data) >= 3:
                    force_sensor_msg = WrenchStamped()
                    force_sensor_msg.header.stamp = self.multi_robot_ros_interface.get_timestamp()
                    force_sensor_msg.wrench.force.x = sensor_data[0]
                    force_sensor_msg.wrench.force.y = sensor_data[1]
                    force_sensor_msg.wrench.force.z = sensor_data[2]
                    self.multi_robot_ros_interface.sensor_ros_pub[sensor_name].publish(force_sensor_msg)
                elif 'imu' in sensor_name and len(sensor_data) >= 4:
                    mat_4x4 = np.eye(4)
                    mat_4x4[:3,:3] = sensor_data.reshape(3, 3)
                    imu_quat = quaternion_from_matrix(mat_4x4)
                    imu_msg = Imu()
                    imu_msg.header.stamp = self.multi_robot_ros_interface.get_timestamp()
                    quaternion_msg = Quaternion()
                    quaternion_msg.x = imu_quat[0]
                    quaternion_msg.y = imu_quat[1]
                    quaternion_msg.z = imu_quat[2]
                    quaternion_msg.w = imu_quat[3]
                    imu_msg.orientation = quaternion_msg
                    self.multi_robot_ros_interface.sensor_ros_pub[sensor_name].publish(imu_msg)

    def reindex_states_data(self):
        joint_names=self.joint_names
        controller_modes=self.reindex_string_data(self.controller_mode)
        joint_position_commands=self.reindex_command_data(self.joint_position_command_all)
        joint_velocity_commands=self.reindex_command_data(self.joint_velocity_command_all)
        joint_torque_commands=self.reindex_command_data(self.joint_torque_command_all)

        return joint_names, controller_modes, joint_position_commands, joint_velocity_commands, joint_torque_commands
    
    def vel_compensation_ctrl(self, ctrl_position, ctrl_velocity=None,idx = None):

        ctrl_position_out = []

        if self.real_time_fps < 50:
            ctrl_position_out = [ctrl_position[idx] + ctrl_velocity[idx] * (2.5/self.real_time_fps)]
        else:
            ctrl_position_out = [ctrl_position[idx] + ctrl_velocity[idx] * 0.1]
        
        return ctrl_position_out
    
    def from_matrix(self, matrix):
        return SimuCommonTools.from_matrix(matrix)

    def pose_to_matrix(self,pose):
        return SimuCommonTools.pose_to_matrix(pose)

    def insert_values(self,original_list, index, values):
        return SimuCommonTools.insert_values(original_list, index, values)
    
    def pose_add(self, pose1, pose2, inv_1_flag = False, inv_2_flag = False):
        return SimuCommonTools.pose_add(pose1, pose2, inv_1_flag, inv_2_flag)
    
    def trans_depth_image_to_point_cloud(self, depth_img, height, width, camera_name):

        near, far = self.get_near_and_far()
        fovy=self.get_camera_fovy(camera_name)
        transform=self.get_camera_transform(camera_name)

        return SimuCommonTools.trans_depth_image_to_point_cloud(depth_img, height, width, fovy, near, far, transform)



   
    
