#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025, Astribot Co., Ltd.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------
# Author: Tony Wang, tonywang@astribot.com
# -----------------------------------------------------------------------------

"""
File: astribot_envs_factory.py
Brief: Factory for simulation env
"""

import os
import yaml
import importlib

import threading
import gymnasium

from simu_utils.simu_common_tools import astribot_simu_log

ros_version = os.getenv('ROS_VERSION')
if ros_version=='1':
    import rospy
elif ros_version=='2':
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    
class AstribotEnvsFactory:
    def __init__(self):
        self.astribot_simu_env=None

    def create_simulation_env(self,data):
        module_name = f"astribot_envs.astribot_{data['simulator_type'].lower()}_env"
        class_name = f"Astribot{data['simulator_type']}Env"

        try:
            module = importlib.import_module(module_name)
            gym_env_name=f"astribot_envs/{class_name}-v0"
            self.astribot_simu_env=gymnasium.make(gym_env_name,param=data)
            self.simu_thread = threading.Thread(target=self.simu_env_loop, args=())
            self.simu_thread.start()
            return self.simu_thread
        except ModuleNotFoundError:
            raise ValueError(f"Unknown simulator type: {data['simulator_type']}. Module: {module_name} not found.")
        except AttributeError:
            raise ValueError(f"Class {class_name} not found in {module_name}.")
    
    @staticmethod
    def load_yaml_file(yaml_file_path):
        with open(yaml_file_path, 'r') as yaml_file:
            astribot_simu_log("Load yaml from: ", yaml_file_path)
            data = yaml.safe_load(yaml_file)
            astribot_simu_root = os.getenv('ASTRIBOT_SIMU_ROOT')
            data['model_path'] = astribot_simu_root + data['model_path']
        return data

    def running(self):
        if ros_version=='1':
            if not hasattr(self, 'rate'):
                self.rate = rospy.Rate(50)
            return not rospy.is_shutdown()
        elif ros_version=='2':
            if not hasattr(self, 'rate'):
                self.rate = self.astribot_simu_env.unwrapped.node.create_rate(50)
            if not hasattr(self, 'spin_thread'):
                self.executor = rclpy.executors.SingleThreadedExecutor()
                self.executor.add_node(self.astribot_simu_env.unwrapped.node)
                self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
                self.spin_thread.start()
            return rclpy.ok()
        
    def simu_env_loop(self):
        observation, info = self.astribot_simu_env.reset()
        while self.running():
            action = self.astribot_simu_env.action_space.sample()
            observation, reward, terminated, truncated, info=self.astribot_simu_env.step(action)

            self.rate.sleep()
        self.astribot_simu_env.close()

