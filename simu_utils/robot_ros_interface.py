#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025, Astribot Co., Ltd.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------
# Author: Tony Wang, tonywang@astribot.com
# -----------------------------------------------------------------------------

"""
File: robot_ros_interface.py
Brief: ros interface for simulation
"""

import os
import copy
import random
import json
import numpy as np
from typing import Any, Dict, List
from functools import partial
  
from astribot_msgs.srv import RawRequest 

from astribot_msgs.msg import RobotJointState, RobotJointController, DoubleArray, RobotVisualStates, RobotCartesianState, RobotCartesianStates
from geometry_msgs.msg import Pose, WrenchStamped, Quaternion, TransformStamped
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, Imu, JoyFeedback, PointCloud2

from simu_utils.simu_common_tools import astribot_simu_log

ros_version = os.getenv('ROS_VERSION')
if ros_version=='1':
    import rospy
    import tf
    from std_srvs.srv import Empty, EmptyResponse
elif ros_version=='2':
    import rclpy 
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
    from rclpy.duration import Duration   
    from tf2_ros import TransformBroadcaster, Buffer, TransformListener
    from std_srvs.srv import Empty

class MultiRobotRosInterface(object):
    def __init__(self, node=None):

        self.reset_flag = False
        self.pose_tuple = ()
        self.pose_tuple_list = []
        self.camera_raw_ros_pub = {}
        self.trajectory_pose_dict=dict()
        self.qos_list=[]
         
        if ros_version=='1':
            self.node=None
            self.tf_broadcaster = tf.TransformBroadcaster()
            rospy.Service('/simalation/reset', Empty, self.reset_ros_server)
            rospy.Service('/simulation/reset_object_pose', RawRequest, self.handle_object_pose_command)
        elif ros_version=='2':
            self.node=node
            self.tf_broadcaster = TransformBroadcaster(self.node)
            self.reset_service = self.node.create_service(Empty, '/simulation/reset', self.reset_ros_server)
            self.reset_object_pose_service = self.node.create_service(RawRequest, '/simulation/reset_object_pose', self.handle_object_pose_command)

            qos_profile_pub = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE, depth=10, deadline=Duration(seconds=0.1))
            qos_profile_sub = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, durability=QoSDurabilityPolicy.VOLATILE, depth=15)

            self.qos_list=[qos_profile_pub, qos_profile_sub]

    def setup_robot_joint_interface(self, robot_joint_map):
        robot_dict = dict()
        for robot_name, joint_names in robot_joint_map.items():
            if 'gripper' in robot_name and len(joint_names) > 1:
                del joint_names[1:]
                robot_dict[robot_name] = RobotRosInterface(robot_name, joint_names, self.node, self.qos_list)
            else:
                robot_dict[robot_name] = RobotRosInterface(robot_name, joint_names, self.node, self.qos_list)
        return robot_dict

    def setup_camera_interface(self, camera_names):
        self.camera_raw_ros_pub = dict()
        self.camera_depth_ros_pub = dict()
        self.camera_point_cloud_ros_pub = dict()
        if ros_version=='1':
            for camera_name in camera_names:
                self.camera_raw_ros_pub[camera_name] = rospy.Publisher('astribot_whole_body/camera/' + camera_name + '/image_raw', Image, queue_size=1)
                self.camera_depth_ros_pub[camera_name] = rospy.Publisher('astribot_whole_body/camera/' + camera_name + '/depth', Image, queue_size=1)
                self.camera_point_cloud_ros_pub[camera_name] = rospy.Publisher('astribot_whole_body/camera/' + camera_name + '/point_cloud', PointCloud2, queue_size=1)
        elif ros_version=='2':
            for camera_name in camera_names:
                self.camera_raw_ros_pub[camera_name] = self.node.create_publisher(Image, 'astribot_whole_body/camera/' + camera_name + '/image_raw', self.qos_list[0])
                self.camera_depth_ros_pub[camera_name] = self.node.create_publisher(Image, 'astribot_whole_body/camera/' + camera_name + '/depth', self.qos_list[0])
                self.camera_point_cloud_ros_pub[camera_name] = self.node.create_publisher(PointCloud2, 'astribot_whole_body/camera/' + camera_name + '/point_cloud', self.qos_list[0])
           
    def setup_sensor_interface(self, sensor_names):
        self.sensor_ros_pub = dict()
        if ros_version=='1':
            for sensor_name in sensor_names:
                if 'force' in sensor_name:
                    self.sensor_ros_pub[sensor_name] = rospy.Publisher(sensor_name + '/force_sensor', WrenchStamped, queue_size=1000)
                elif 'imu' in sensor_name:
                    self.sensor_ros_pub[sensor_name] = rospy.Publisher(sensor_name + '/imu_sensor', Imu, queue_size=1000)
        elif ros_version=='2':
            for sensor_name in sensor_names:
                if 'force' in sensor_name:
                    self.sensor_ros_pub[sensor_name] = self.node.create_publisher(WrenchStamped, sensor_name + '/force_sensor', self.qos_list[0])
                elif 'imu' in sensor_name:
                    self.sensor_ros_pub[sensor_name] = self.node.create_publisher(Imu, sensor_name + '/imu_sensor', self.qos_list[0])

    def setup_trajectory_and_com_psoe(self, update_trajectory_map):

        self.com_pos = [-0.2, 0, -0.0]
        for robot_name, trajectory_pose_names in update_trajectory_map:
            self.trajectory_pose_dict[robot_name] = list()
            for i in range(len(trajectory_pose_names)):
                self.trajectory_pose_dict[robot_name].append([0.0 for i in range(7)])

            if ros_version=='1':
                self.trajectory_pose_sub_dict[robot_name] = rospy.Subscriber(robot_name + '/trajectory_pose', RobotVisualStates, self.trajectory_pose_callback, robot_name, queue_size=1)
                if not hasattr(self, 'com_pos_sub'):
                    self.com_pos_sub = rospy.Subscriber('/astribot_whole_body/com_pos', DoubleArray, self.com_pos_callback,queue_size=1)
            elif ros_version=='2':
                self.trajectory_pose_sub_dict[robot_name] = self.node.create_subscription(RobotVisualStates, robot_name + '/trajectory_pose', self.trajectory_pose_callback, self.qos_list[1])
                if not hasattr(self, 'com_pos_sub'):
                    self.com_pos_sub = self.node.create_subscription(DoubleArray, '/astribot_whole_body/com_pos', self.com_pos_callback,self.qos_list[1])

    def get_object_pose_list(self):
        return self.pose_tuple_list
    
    def get_reset_flag(self):
        return self.reset_flag
    
    def reset_ros_server(self, request, response):
        self.reset_flag = True

        if ros_version=='1':
            return EmptyResponse()
        elif ros_version=='2':
            return response
    
    def handle_object_pose_command(self, req: Any):
        try:
            data: Dict[str, List[float]] = json.loads(req.request)

            for key, value in data.items():
                if not isinstance(value, list) or len(value) != 7:
                    raise ValueError(f"The value for '{key}' must be a list of exactly 7 floats.")
                
                twist: List[float] = [0, 0, 0, 0, 0, 0]
                self.pose_tuple = (value, twist, key, True)
                self.pose_tuple_list.append(self.pose_tuple)

            response_message = "Processed data: " + str(data)
        
        except json.JSONDecodeError:
            response_message = "Error: Invalid JSON format."
        except ValueError as ve:
            response_message = f"Error: {str(ve)}"
        except Exception as e:
            response_message = "Error: " + str(e)

        response.response = response_message
        return response
    
    def publish_object_pose(self, pose, object_name):
        if ros_version=='1':
            position = (pose[0], pose[1], pose[2])
            orientation = (pose[4], pose[5], pose[6], pose[3])

            self.tf_broadcaster.sendTransform(
                position,
                orientation,
                rospy.Time.now(),
                object_name, 
                "/simulation"  
            )
        elif ros_version=='2':

            position = (pose[0], pose[1], pose[2])
            orientation = (pose[4], pose[5], pose[6], pose[3])

            transform = TransformStamped()
            transform.header.stamp = self.node.get_clock().now().to_msg()
            transform.header.frame_id = "/simulation"
            transform.child_frame_id = object_name
            transform.transform.translation.x = position[0]
            transform.transform.translation.y = position[1]
            transform.transform.translation.z = position[2]
            transform.transform.rotation.x = orientation[0]
            transform.transform.rotation.y = orientation[1]
            transform.transform.rotation.z = orientation[2]
            transform.transform.rotation.w = orientation[3]

            self.tf_broadcaster.sendTransform(transform)

    def trajectory_pose_callback(self, msg, robot_name):
        len_every = int(len(msg.pose) / len(self.trajectory_pose_dict[robot_name]))
        for i in range(len(self.trajectory_pose_dict[robot_name])):
            len_ = len_every * i
            pose = [msg.pose[len_].position.x, msg.pose[len_].position.y, msg.pose[len_].position.z,
                    msg.pose[len_].orientation.w, msg.pose[len_].orientation.x, msg.pose[len_].orientation.y, msg.pose[len_].orientation.z]
            self.trajectory_pose_dict[robot_name][i] = pose

    def com_pos_callback(self, msg):
        self.com_pos = [msg.data[0]-0.2, msg.data[1], msg.data[2]]

    def get_timestamp(self):
        return RobotRosInterface.get_timestamp(self.node)
    
class RobotRosInterface(object):
    def __init__(self, robot_name, joint_names, node=None, qos_list=None, joint_position=None, joint_velocity=None, joint_torque=None):
        self.robot_name = robot_name
        self.joint_names = joint_names 
        self.dof = len(joint_names)
        self.joint_position = joint_position if joint_position is not None else [0.0] * len(joint_names)
        self.joint_velocity = joint_velocity if joint_velocity is not None else [0.0] * len(joint_names)
        self.joint_torque = joint_torque if joint_torque is not None else [0.0] * len(joint_names)
        self.joint_position_command = copy.deepcopy(self.joint_position)
        self.joint_velocity_command = [0.0 for i in range(self.dof)]
        self.joint_torque_command = [0.0 for i in range(self.dof)]
        self.simu_running = True 
        
        pub_name = self.robot_name + '/joint_space_states'
        sub_name = self.robot_name + '/joint_space_command'

        sub_endpoint_desired_name = self.robot_name + '/endpoint_desired_states'
        sub_endpoint_current_name = self.robot_name + '/endpoint_current_states'

        self.node = node
        if ros_version=='1':
            self.joint_states_pub = rospy.Publisher(pub_name, RobotJointState, queue_size=1)
            self.joint_command_sub = rospy.Subscriber(sub_name, RobotJointController, self.joint_command_callback, queue_size=1)
            self.joint_states_real_sub = rospy.Subscriber(self.robot_name + '/joint_space_states', RobotJointState, self.joint_states_callback, queue_size=1)

            self.endpoint_desired_states_sub = rospy.Subscriber(sub_endpoint_desired_name, RobotCartesianState, self.endpoint_desired_states_callback, queue_size=1)
            self.endpoint_current_states_sub = rospy.Subscriber(sub_endpoint_current_name, RobotCartesianState, self.endpoint_current_states_callback, queue_size=1)

            self.ft_sensor_pub = rospy.Publisher(self.robot_name + '/force_torque_sensor', WrenchStamped, queue_size=1000)
            if 'astribot_chassis' == self.robot_name:
                self.chassis_pose_ros_pub = rospy.Publisher(self.robot_name + '/simulation/chassis/pose', RobotVisualStates,queue_size=1)

        elif ros_version=='2':
            qos_profile_pub = qos_list[0]
            qos_profile_sub = qos_list[1]
            self.joint_states_pub = self.node.create_publisher(RobotJointState, pub_name, qos_profile_pub)
            self.joint_command_sub = self.node.create_subscription(RobotJointController, sub_name, self.joint_command_callback, qos_profile_sub)
            self.joint_states_real_sub = self.node.create_subscription(RobotJointState, pub_name, self.joint_states_callback, qos_profile_sub)

            self.endpoint_desired_states_sub = self.node.create_subscription(RobotCartesianState, sub_endpoint_desired_name, self.endpoint_desired_states_callback, qos_profile_sub)
            self.endpoint_current_states_sub = self.node.create_subscription(RobotCartesianState, sub_endpoint_current_name, self.endpoint_current_states_callback, qos_profile_sub)
            
            self.ft_sensor_pub = self.node.create_publisher(WrenchStamped, self.robot_name + '/force_torque_sensor', qos_profile_pub)
            if 'astribot_chassis' == self.robot_name:
                self.chassis_pose_ros_pub = self.node.create_publisher(RobotVisualStates, self.robot_name + '/simulation/chassis/pose', 1)

        self.endpoint_desired_pose = None
        self.endpoint_desired_twist = [0.0 for i in range(6)]
        self.endpoint_desired_wrench = [0.0 for i in range(6)]

        self.endpoint_current_pose = None
        self.endpoint_current_twist = [0.0 for i in range(6)]
        self.endpoint_current_wrench = [0.0 for i in range(6)]

        self.joint_states_msg = RobotJointState()
        self.joint_states_msg.name = self.joint_names
        self.joint_states_msg.position = self.joint_position
        self.joint_states_msg.velocity = self.joint_velocity
        self.joint_states_msg.acceleration = [0.0 for i in range(self.dof)]
        self.joint_states_msg.torque = [0.0 for i in range(self.dof)]

        self.controller_mode = 1

    def get_controller_mode(self):
        return self.controller_mode

    def joint_command_callback(self, msg):
        # print("msg mode: ", msg.mode)
        joint_position_command = [0 for i in range(self.dof)]
        joint_velocity_command = [0 for i in range(self.dof)]
        joint_torque_command = [0 for i in range(self.dof)]
        if msg.mode == 1:
            if len(msg.command) / self.dof == 1:
                joint_position_command = list(msg.command)
            elif len(msg.command) / self.dof == 2:
                joint_position_command = list(msg.command[:self.dof])
                joint_velocity_command = list(msg.command[self.dof:])
            elif len(msg.command) / self.dof == 3:
                joint_position_command = list(msg.command[:self.dof])
                joint_velocity_command = list(msg.command[self.dof:self.dof*2])
                self.joint_torque_command = list(msg.command[self.dof*2:])
            self.controller_mode = 1
        elif msg.mode == 2: 
            joint_velocity_command = list(msg.command)
            self.controller_mode = 2
        elif msg.mode == 3: 
            joint_torque_command = list(msg.command)
            self.controller_mode = 3

        self.joint_position_command = joint_position_command
        self.joint_velocity_command = joint_velocity_command
        self.joint_torque_command = joint_torque_command
            
    def endpoint_desired_states_callback(self, msg):
        self.endpoint_desired_pose = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                                      msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]
        self.endpoint_desired_twist = [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z,
                                       msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z]
        self.endpoint_desired_wrench = [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                                        msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z]

    def endpoint_current_states_callback(self, msg):
        self.endpoint_current_pose = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                                      msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]
        self.endpoint_current_twist = [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z,
                                       msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z]
        self.endpoint_current_wrench = [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                                        msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z]
        
    def publish_joint_states(self, position, velocity, acceleration, torque):
        if self.simu_running:

            self.joint_states_msg.header.stamp = RobotRosInterface.get_timestamp(self.node)
            self.joint_states_msg.header.frame_id = 'simulation'

            self.joint_states_msg.position = position
            self.joint_states_msg.velocity = velocity
            self.joint_states_msg.acceleration = acceleration
            self.joint_states_msg.torque = torque

            self.joint_states_pub.publish(self.joint_states_msg)
        else:
            pass

    def publish_chassis_pose(self, pose):
        if 'chassis' in self.robot_name:
            msg = RobotVisualStates()
            msg.id = ['astribot_chassis']
            pose_msg = Pose()
            
            pose_msg.position.x = pose[0]
            pose_msg.position.y = pose[1]
            pose_msg.position.z = pose[2]
            pose_msg.orientation.x = pose[4]
            pose_msg.orientation.y = pose[5]
            pose_msg.orientation.z = pose[6]
            pose_msg.orientation.w = pose[3]
            
            msg.pose.append(pose_msg)
            self.chassis_pose_ros_pub.publish(msg)

    def joint_states_callback(self, msg):
        if self.simu_running == True:
            if (msg.header.frame_id != 'simulation') & ('gripper' not in self.robot_name):
                self.simu_running = False
                self.joint_command_sub.unregister()
                print("real robot running, disable simulation running", self.robot_name)
        else:
            self.joint_position_command = msg.position
            self.joint_velocity_command = msg.velocity
            self.joint_torque_command = msg.torque

    def publish_force_torque_sensor(self, force_torque):

        if len(force_torque) != 6:
            return
        
        ft_sensor_msg = WrenchStamped()
        ft_sensor_msg.header.stamp = RobotRosInterface.get_timestamp(self.node)
        ft_sensor_msg.wrench.force.x = force_torque[0]
        ft_sensor_msg.wrench.force.y = force_torque[1]
        ft_sensor_msg.wrench.force.z = force_torque[2]
        ft_sensor_msg.wrench.torque.x = force_torque[3]
        ft_sensor_msg.wrench.torque.y = force_torque[4]
        ft_sensor_msg.wrench.torque.z = force_torque[5]
        self.ft_sensor_pub.publish(ft_sensor_msg)

    @staticmethod
    def get_timestamp(node):
        timestamp=None
        ros_version = os.getenv('ROS_VERSION')
        if ros_version=='1':
            timestamp = rospy.Time.now()
        elif ros_version=='2':
            timestamp = node.get_clock().now().to_msg()
        
        return timestamp
    
    def get_joint_position_command(self):
        return self.joint_position_command

    def get_joint_velocity_command(self):
        return self.joint_velocity_command

    def get_joint_torque_command(self):
        return self.joint_torque_command

    def get_endpoint_desired_pose(self):
        return copy.deepcopy(self.endpoint_desired_pose)

    def get_endpoint_desired_twist(self):
        return self.endpoint_desired_twist

    def get_endpoint_desired_wrench(self):
        return self.endpoint_desired_wrench

    def get_endpoint_current_pose(self):
        return self.endpoint_current_pose

    def get_endpoint_current_twist(self):
        return self.endpoint_current_twist

    def get_endpoint_current_wrench(self):
        return self.endpoint_current_wrench
