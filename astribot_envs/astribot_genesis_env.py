#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025, Astribot Co., Ltd.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------
# Author: Tony Wang, tonywang@astribot.com
# -----------------------------------------------------------------------------

"""
File: astribot_genesis_env.py
Brief: genesis simulation env
"""

import time
import numpy as np
import cv2

import torch

import genesis as gs

from simu_utils.simu_common_tools import astribot_simu_log
from astribot_envs.astribot_base_env import AstribotBaseEnv

class AstribotGenesisEnv(AstribotBaseEnv):
    def __init__(self, param):

        super().__init__(param)

        self.dof_index = []
        self.joint_name_to_index = {}
    
        astribot_simu_log("Init genesis viewer mode")
        self.show_viewer=False
        if self.render_mode == 'human':
            self.show_viewer=True
        elif self.render_mode == 'rgb_array':
            self.show_viewer=False

        astribot_simu_log("Init genesis scene")
        gs.init(backend=gs.gpu, logging_level='warning')
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
            res=(self.width, self.height),
            camera_pos=(2, 0.0, 2.0),
            camera_lookat=(0.0, 0.0, 1.0),
            camera_fov=100,
            max_FPS=60,),
            sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0, 0, -10.0),
            ),
            show_viewer=self.show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.setup_genesis_model()
        self.setup_genesis_camera()
        self.scene.build()

        astribot_simu_log("Init robot joint map")
        self.joint_names = [joint for sublist in self.joint_names_list for joint in sublist]
        self.setup_joint_index_mapping()
        self.joint_names_all,self.controller_mode, self.joint_position_command_all, self.joint_velocity_command_all, self.joint_torque_command_all = self.update_joint_states()
        
    def step(self, action: np.ndarray) -> tuple:
        self.update_reset_flag()
        if self.reset_flag == False:
            step_begin_time=time.time()
            self.joint_names_all,self.controller_mode, self.joint_position_command_all, self.joint_velocity_command_all, self.joint_torque_command_all = self.update_joint_states()
            self.joint_names_all,self.controller_mode, self.joint_position_command_all, self.joint_velocity_command_all, self.joint_torque_command_all = self.reindex_states_data()

            self.update_object_states()
            self.update_trajectory_pose()
            self.update_com_pose()
            self.update_sensor_states()

            pos_ctrl_data=[]
            pos_dof_index=[]
            force_ctrl_data=[]
            force_dof_index=[]
            for joint_id, mode in enumerate(self.controller_mode):
                joint_name=None
                if mode == 1:
                    pos_ctrl_data += self.vel_compensation_ctrl(self.joint_position_command_all, self.joint_velocity_command_all, idx=joint_id)
                    joint_name = self.joint_names_all[joint_id]
                    pos_dof_index += [self.joint_name_to_index[joint_name]]
                elif mode == 2:
                    now_position_status=self.reindex_command_data(self.get_joint_positions(self.joint_names_all))
                    pos_ctrl_data += self.vel_compensation_ctrl(now_position_status, self.joint_velocity_command_all, idx=joint_id)
                    joint_name = self.joint_names_all[joint_id]
                    pos_dof_index += [self.joint_name_to_index[joint_name]]
                elif mode == 3:
                    force_ctrl_data += [self.joint_torque_command_all[joint_id]]
                    joint_name = self.joint_names_all[joint_id]
                    force_dof_index += [self.joint_name_to_index[joint_name]]

            for _ in range(self.frame_skip):
                if pos_ctrl_data:
                    self.robot.set_dofs_position(pos_ctrl_data, pos_dof_index)
                if force_ctrl_data:
                    self.robot.control_dofs_force(force_ctrl_data, force_dof_index)

                self.scene.step()
            self.render()

            step_end_time=time.time()
            self.real_time_fps=1/(step_end_time-step_begin_time)
    
        else:
            self.reset()

            self.reset_flag = False
        
        observation = self._get_obs()
        reward = 0
        terminated = False
        info = self._get_info()

        return observation, reward, terminated, False, info
    
    def reset(self, seed=None, options=None):
        astribot_simu_log("Reset scene")
        super().reset(seed=seed)
        self.scene.reset()

        self.reset_flag = False

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _get_obs(self) -> np.ndarray:
        pose=list()
        if self.object_names:
            for object_name in self.object_names:
                pose = self.get_body_pose(object_name)
        else:
            pose=np.zeros(7)
        return pose
    
    def _get_info(self) -> dict:
        robot_info=dict()

        for robot_name in self.robot_list:
            joint_names = self.robot_joint_map[robot_name]
            
            if 'gripper' in robot_name:  
                del joint_names[1:]
                robot_info[robot_name] = self.get_joint_positions(joint_names)
            elif 'chassis' in robot_name:
                pose = self.get_site_pose('chassis')
                robot_info[robot_name] = pose
            else:
                robot_info[robot_name] = self.get_joint_positions(joint_names)

        return robot_info
    
    def render(self):
        # AstribotBaseEnv.astribot_simu_log("Render genesis env")
        self.scene.visualizer.update()
        self.update_camera_pose()
        return self.update_camera_data()

    def close(self):
        pass

    def setup_genesis_model(self):
        astribot_simu_log("Setup genesis model with mjcf")
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
            file  = self.model_path,
            pos =(0,0,0.05),
            euler = (0, 0, 0),),
            material=gs.materials.Rigid(gravity_compensation=self.gravity_compensation)
            )

    def setup_genesis_camera(self):
        astribot_simu_log("Setup genesis camera")
        for camera_name in self.camera_names:
            if camera_name=='astribot_head':
                self.head_camera = self.scene.add_camera(
                    res    = (640, 480),
                    pos    = (0.0861, 0.0800, 1.4077),
                    lookat = (1.0, 0.0, 1.4077),
                    fov    = 60,
                    GUI    = True,
                )
            elif camera_name=='astribot_arm_left_effector':
                self.left_camera = self.scene.add_camera(
                    res    = (640, 480),
                    pos    = (3.8, 0.0, 2.5),
                    lookat = (0, 0, 0.5),
                    fov    = 30,
                    GUI    = False,
                )
            elif camera_name=='astribot_arm_right_effector':
                self.right_camera = self.scene.add_camera(
                    res    = (640, 480),
                    pos    = (4.2, 0.0, 2.5),
                    lookat = (0, 0, 0.5),
                    fov    = 30,
                    GUI    = False,
                )
            elif camera_name=='astribot_global_camera':
                self.global_camera = self.scene.add_camera(
                    res    = (640, 480),
                    pos    = (0.5, 0.011, 2.0),
                    lookat = (0.5, 0.011, 0.0),
                    up = (-1, 0, 0),
                    fov    = 70,
                    GUI    = False,
                )

    def setup_joint_index_mapping(self):
        astribot_simu_log("Setup joint index mapping")

        self.dof_index = [self.robot.get_joint(name).dof_idx_local for name in self.joint_names]

        for name in self.joint_names:
            index = self.robot.get_joint(name).dof_idx_local
            self.joint_name_to_index[name] = index

    def get_ft_sensor_data(self, robot_name):
        pass

    def get_reset_status(self):
        return False
    
    def get_camera_image(self, camera_name='astribot_head'):

        data={}
        render_data=None
        rgb_img = np.zeros((640, 480, 3), dtype=np.uint8)

        if camera_name=='astribot_head':
            render_data=self.head_camera.render(rgb=True)
        elif camera_name=='astribot_arm_right_effector':
            render_data=self.right_camera.render(rgb=True)
        elif camera_name=='astribot_arm_left_effector':
            render_data=self.left_camera.render(rgb=True)
        elif camera_name=='astribot_global_camera':
            render_data=self.global_camera.render(rgb=True)

        rgb_img = cv2.cvtColor(render_data, cv2.COLOR_RGB2BGR)

        data['rgb_img'] = rgb_img
        data['depth_img'] = None
        data['point_cloud'] = None

        return data
    
    def update_camera_pose(self):
        for camera_name in self.camera_names:
            if camera_name=='astribot_head':
                link=self.robot.get_link(name='astribot_head_link_1')
                link_pose=link.get_pos()
                link_quat=link.get_quat()
                head_camera_pose = torch.cat((link_pose, link_quat)).cpu().numpy().tolist()
                head_camera_matrix=self.pose_to_matrix(head_camera_pose)
                print("head_camera: ",head_camera_matrix)
                self.head_camera.set_pose(transform=head_camera_matrix)

    def get_site_pose(self, name):
        joint = self.robot.get_joint(name)
        p = joint.get_pos()
        q = joint.get_quat()
        return np.hstack((p, q))

    def get_body_pose(self):
        pass

    def get_chassis_pose(self):
        return [0, 0, 0, 1, 0, 0, 0]
        
    def reindex_command_data(self, command):
        left_gripper_data = 0
        right_gripper_data = 0

        for joint_name in self.joint_names:
            if 'gripper' in joint_name:
                gripper_dof_index = self.joint_name_to_index[joint_name]
                gripper_index = next(i for i, value in enumerate(self.dof_index) if value == gripper_dof_index)

                if len(command) < 32:
                    insert_size = (32 - len(command)) // 2
                    insert_list = [0] * insert_size
                    command = self.insert_values(command, 14, insert_list)
                    command = self.insert_values(command, 27, insert_list)

                if joint_name in ['astribot_gripper_left_joint_L1', 'astribot_gripper_left_joint_L11', 
                                'astribot_gripper_left_joint_R1', 'astribot_gripper_left_joint_R2']:
                    if joint_name == 'astribot_gripper_left_joint_L1':
                        left_gripper_data = command[gripper_index]
                    command[gripper_index] = left_gripper_data / 100 * 0.93
                elif joint_name in ['astribot_gripper_left_joint_L2', 'astribot_gripper_left_joint_R11']:
                    command[gripper_index] = -left_gripper_data / 100 * 0.93

                elif joint_name in ['astribot_gripper_right_joint_L1', 'astribot_gripper_right_joint_L11', 
                                'astribot_gripper_right_joint_R1', 'astribot_gripper_right_joint_R2']:
                    if joint_name == 'astribot_gripper_right_joint_L1':
                        right_gripper_data = command[gripper_index]
                    command[gripper_index] = right_gripper_data / 100 * 0.93
                elif joint_name in ['astribot_gripper_right_joint_L2', 'astribot_gripper_right_joint_R11']:
                    command[gripper_index] = -right_gripper_data / 100 * 0.93
        
        return command
    
    def reindex_string_data(self, string):
        left_gripper_data = ""
        right_gripper_data = ""

        for joint_name in self.joint_names:
            if 'gripper' in joint_name:
                gripper_dof_index = self.joint_name_to_index[joint_name]
                gripper_index = next(i for i, value in enumerate(self.dof_index) if value == gripper_dof_index)

                if len(string) < 32:
                    insert_size = (32 - len(string)) // 2
                    insert_list = [""] * insert_size
                    string = self.insert_values(string, 14, insert_list)
                    string = self.insert_values(string, 27, insert_list)

                if joint_name in ['astribot_gripper_left_joint_L1', 'astribot_gripper_left_joint_L11', 
                                'astribot_gripper_left_joint_R1', 'astribot_gripper_left_joint_R2']:
                    if joint_name == 'astribot_gripper_left_joint_L1':
                        left_gripper_data = string[gripper_index]
                    string[gripper_index] = left_gripper_data 
                elif joint_name in ['astribot_gripper_left_joint_L2', 'astribot_gripper_left_joint_R11']:
                    string[gripper_index] = left_gripper_data 

                elif joint_name in ['astribot_gripper_right_joint_L1', 'astribot_gripper_right_joint_L11', 
                                'astribot_gripper_right_joint_R1', 'astribot_gripper_right_joint_R2']:
                    if joint_name == 'astribot_gripper_right_joint_L1':
                        right_gripper_data = string[gripper_index]
                    string[gripper_index] = right_gripper_data 
                elif joint_name in ['astribot_gripper_right_joint_L2', 'astribot_gripper_right_joint_R11']:
                    string[gripper_index] = right_gripper_data 
        
        return string
    
    def get_joint_position(self, joint_name):
        qpos = self.robot.get_qpos(self.dof_index).cpu().numpy().flatten()
        qpos_dof_index = self.joint_name_to_index[joint_name]
        qpos_index = [index for index, value in enumerate(self.dof_index) if value == qpos_dof_index]

        if self.robot.get_joint(joint_name).type == "free":
            return qpos[qpos_index:qpos_index + 3].item()
        elif 'gripper' in joint_name:
            return abs(qpos[qpos_index].item() / 0.93 * 100) 
        else:
            return qpos[qpos_index].item()

    def get_joint_positions(self, names):
        pos = []
        for i in names:
            pos.append(self.get_joint_position(i))
        return pos

    def get_joint_velocity(self, joint_name):
        qvel = self.robot.get_dofs_velocity(self.dof_index).cpu().numpy().flatten()
        qvel_dof_index = self.joint_name_to_index[joint_name]

        qvel_index = [index for index, value in enumerate(self.dof_index) if value == qvel_dof_index]

        if self.robot.get_joint(joint_name).type == "free":
            return qvel[qvel_index:qvel_index + 3].item()
        elif 'gripper' in joint_name:
            return qvel[qvel_index].item() / 0.93 * 100
        else:
            return qvel[qvel_index].item()

    def get_joint_velocities(self, names):
        vel = []
        for i in names:
            vel.append(self.get_joint_velocity(i))
        return vel
    
    def get_joint_accelerations(self, names):
        acc = []
        return acc

    def get_joint_torque(self, joint_name):
        qf = self.robot.get_dofs_force(self.dof_index).cpu().numpy().flatten()
        qf_dof_index = self.joint_name_to_index[joint_name]

        qf_index = [index for index, value in enumerate(self.dof_index) if value == qf_dof_index]
        if self.robot.get_joint(joint_name).type == "free":
            return qf[qf_index:qf_index + 3].item()
        else:
            return qf[qf_index].item()

    def get_joint_torques(self, names):
        tor = []
        for i in names:
            tor.append(self.get_joint_torque(i))
        return tor

