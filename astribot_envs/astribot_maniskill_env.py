#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025, Astribot Co., Ltd.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------
# Author: Tony Wang, tonywang@astribot.com
# -----------------------------------------------------------------------------

"""
File: astribot_maniskill_env.py
Brief: maniskill simulation env
"""

import os
import numpy as np
import torch

import sapien
import sapien.core as sapien
sapien.set_log_level("error")
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs import Pose
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env

from simu_utils.simu_common_tools import astribot_simu_log
from astribot_envs.astribot_base_env import AstribotBaseEnv

@register_agent()
class AstribotAgent(BaseAgent):
    uid = "Astribot_s1"
    urdf_path = os.getenv('ASTRIBOT_SIMU_ROOT') + "/astribot_descriptions/urdf/astribot_s1_urdf/astribot_whole_body_maniskill.urdf"
    urdf_config = dict()
    fix_root_link = True
    load_multiple_collisions = True

    keyframes = dict(
        standing=Keyframe(
            pose=sapien.Pose(q=[0,0,0,1],p=[0, 0, 0]),
            qpos=np.array([0.0] * (20)) * 1,
        )
    )

    body_joints = [
        "astribot_torso_joint_1",
        "astribot_torso_joint_2",
        "astribot_torso_joint_3",
        "astribot_torso_joint_4",
        "astribot_head_joint_1",
        "astribot_head_joint_2",
        "astribot_arm_left_joint_1",
        "astribot_arm_left_joint_2",
        "astribot_arm_left_joint_3",
        "astribot_arm_left_joint_4",
        "astribot_arm_left_joint_5",
        "astribot_arm_left_joint_6",
        "astribot_arm_left_joint_7",
        "astribot_arm_right_joint_1",
        "astribot_arm_right_joint_2",
        "astribot_arm_right_joint_3",
        "astribot_arm_right_joint_4",
        "astribot_arm_right_joint_5",
        "astribot_arm_right_joint_6",
        "astribot_arm_right_joint_7",
        'astribot_gripper_left_joint_L1',
        'astribot_gripper_left_joint_L2',
        'astribot_gripper_left_joint_L11',
        'astribot_gripper_left_joint_R1',
        'astribot_gripper_left_joint_R2',
        'astribot_gripper_left_joint_R11',
        'astribot_gripper_right_joint_L1',
        'astribot_gripper_right_joint_L2',
        'astribot_gripper_right_joint_L11',
        'astribot_gripper_right_joint_R1',
        'astribot_gripper_right_joint_R2',
        'astribot_gripper_right_joint_R11'
    ]

    body_stiffness = 40
    body_damping = 20
    body_force_limit = 30

    @property
    def _controller_configs(self):
        body_pd_joint_pos = PDJointPosControllerConfig(
            self.body_joints,
            lower=None,
            upper=None,
            stiffness=self.body_stiffness,
            damping=self.body_damping,
            force_limit=self.body_force_limit,
            normalize_action=False,
        )
        body_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.body_joints,
            lower=-0.01,
            upper=0.01,
            stiffness=self.body_stiffness,
            damping=self.body_damping,
            force_limit=self.body_force_limit,
            use_delta=True,
        )
        return dict(
            pd_joint_pos=dict(body=body_pd_joint_pos, balance_passive_force=False),
            pd_joint_delta_pos=dict(
                body=body_pd_joint_delta_pos, balance_passive_force=False
            ),
        )

    @property
    # sudo apt-get install xclip
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="left_wrist_rgbd",
                pose=sapien.Pose(p=[0, 0.04, 0.03], q=[0.5,-0.5,-0.5,-0.5]),
                width=640,
                height=360,
                fov=0.9948376736367679,
                near=0.028338245138894207,
                far=141.6912288615168,
                mount=self.robot.links_map["astribot_gripper_left_base"],
            ),
            CameraConfig(
                uid="right_wrist_rgbd",
                pose=sapien.Pose(p=[0, 0.04, 0.03], q=[0.5,-0.5,-0.5,-0.5]),
                width=640,
                height=360,
                fov=0.9948376736367679,
                near=0.028338245138894207,
                far=141.6912288615168,
                mount=self.robot.links_map["astribot_gripper_right_base"],
            ),
            CameraConfig(
                uid="head_rgbd",
                pose=sapien.Pose(p=[0.0603, -0.2139, -0.0277], q=[0.7071, 0.7071, 0.0, 0.0]),
                width=1280,
                height=720,
                fov=0.8974046681441843,
                near=0.028338245138894207,
                far=141.6912288615168,
                mount=self.robot.links_map["astribot_head_link_2"],
            )
        ]

@register_env("astribot_envs/AstribotManiskillEnv-v0", max_episode_steps=50)
class AstribotManiskillEnv(BaseEnv, AstribotBaseEnv):
    def __init__(self, *args, param, **kwargs):
        
        AstribotBaseEnv.__init__(self, param)
        BaseEnv.__init__(self, *args,render_mode=self.render_mode, robot_uids='Astribot_s1',**kwargs)

        astribot_simu_log("Setup joint qpos and qvel mapping")
        self.joint_name_to_index, self.joint_index_to_name=self.setup_joint_qpos_mapping()

        astribot_simu_log("Init robot joint map")
        self.joint_names = [joint for sublist in self.joint_names_list for joint in sublist]

        astribot_simu_log("Init ros joint interface")
        self.joint_names_all,self.controller_mode, self.joint_position_command_all, self.joint_velocity_command_all, self.joint_torque_command_all = self.update_joint_states()

    def setup_joint_qpos_mapping(self):
        active_joints_map = self.agent.robot.active_joints_map

        joint_name_to_index = {}
        joint_index_to_name = {}

        for index, (joint_name, _) in enumerate(active_joints_map.items()):
            joint_name_to_index[joint_name] = index
            joint_index_to_name[index] = joint_name

        return joint_name_to_index, joint_index_to_name

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(q=[0,0,0,1],p=[1.3, 0, -0.9]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=0.2
        )
        self.table_scene.build()

        self.obj = actors.build_cube(
            self.scene,
            half_size=0.025,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0.4, 0, 1.0]),
        )

    def get_ft_sensor_data(self, robot_name):
        pass

    def get_reset_status(self):
        press_status=False
        
        return press_status
    
    def get_gravity_torque(self):
        dof = int(self.agent.robot.get_dof())
        zero_vector = np.zeros(dof, dtype=np.float32)
        self.agent.robot.set_qvel(zero_vector)
        gravity_torque = self.agent.robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=False)
        return gravity_torque
    
    def get_joint_position(self, joint_name):
        qpos = self.agent.robot.get_qpos().flatten()
        qpos_index = self.joint_name_to_index[joint_name]
        if self.agent.robot.find_joint_by_name(joint_name).get_type() == "free":
            return qpos[qpos_index:qpos_index + 3]
        elif 'gripper' in joint_name:
            return abs(qpos[qpos_index] / 0.93 * 100)
        else:
            return qpos[qpos_index]

    def get_joint_positions(self, names):
        pos = []
        for i in names:
            pos.append(float(self.get_joint_position(i)))
        return pos
    
    def get_joint_velocity(self, joint_name):
        qvel = self.agent.robot.get_qvel().flatten()
        qvel_index = self.joint_name_to_index[joint_name]
        if self.agent.robot.find_joint_by_name(joint_name).get_type() == "free":
            return qvel[qvel_index:qvel_index + 3]
        elif 'gripper' in joint_name:
            return qvel[qvel_index] / 0.93 * 100
        else:
            return qvel[qvel_index]

    def get_joint_velocities(self, names):
        vel = []
        for i in names:
            vel.append(float(self.get_joint_velocity(i)))
        return vel
    
    def get_joint_torque(self, joint_name):
        qf = self.agent.robot.get_qf().flatten()
        qf_index = self.joint_name_to_index[joint_name]
        if self.agent.robot.find_joint_by_name(joint_name).get_type() == "free":
            return qf[qf_index:qf_index + 3]
        else:
            return qf[qf_index]

    def get_joint_torques(self, names):
        tor = []
        for i in names:
            tor.append(float(self.get_joint_torque(i)))
        return tor
    
    def get_joint_acceleration(self, joint_name):
        qacc = self.agent.robot.get_qacc().flatten()
        qacc_index = self.joint_name_to_index[joint_name]
        if self.agent.robot.find_joint_by_name(joint_name).get_type() == "free":
            return qacc[qacc_index:qacc_index + 3]
        else:
            return qacc[qacc_index]

    def get_joint_accelerations(self, names):
        acc = []
        # for i in names:
        #     acc.append(self.get_joint_acceleration(i))
        return acc
    
    def get_site_pose(self, name):
        p = []
        q = []
        joint_list = self.agent.robot.get_joints()
        for joint in joint_list:
            joint_name = joint.get_name()
            if name in joint_name:
                pose = joint.get_global_pose()
                p = pose.get_p()
                q = pose.get_q()
                return np.hstack((p, q))

    def get_camera_image(self, camera_name):
        data = {}

        sensor_images = self.get_sensor_images() 

        if not sensor_images:
            astribot_simu_log("No sensor images available", level="ERROR")
            return data
        if camera_name not in sensor_images:
            astribot_simu_log("Can not found in sensor images", level="ERROR")
            return data
        camera_images = sensor_images[camera_name]

        if 'rgb' not in camera_images:
            astribot_simu_log("No 'rgb' image data found for camera", level="ERROR")
            return data
        rgb_img = camera_images['rgb']

        if isinstance(rgb_img, torch.Tensor):
            rgb_img = rgb_img.cpu().numpy().squeeze()
        elif not isinstance(rgb_img, np.ndarray):
            astribot_simu_log("RGB data type not supported for camera", level="ERROR")
            return data
        
        if len(rgb_img.shape) != 3 or rgb_img.shape[2] != 3:
            astribot_simu_log("Invalid RGB image shape for camera", level="ERROR")
            return data
        data['rgb_img'] = rgb_img
        data['depth_img'] = None
        data['point_cloud'] = None

        return data
    
    def compute_normalized_dense_reward(self, obs, action, info):
        return 0

    def reindex_command_data(self, command):
        reindex_data=[0] * 32
        for joint_id, joint_name in enumerate(self.joint_names_all):
            reindex=self.joint_name_to_index[joint_name]

            if joint_name == 'astribot_gripper_left_joint_L1':
                left_gripper_qpos = command[joint_id]/ 100 * 0.93
                reindex_data[reindex]=-left_gripper_qpos

                for acc_joint_name in ['astribot_gripper_left_joint_R1', 'astribot_gripper_left_joint_R2', 'astribot_gripper_left_joint_L11']:
                    acc_index=self.joint_name_to_index[acc_joint_name]
                    reindex_data[acc_index] = left_gripper_qpos
                for acc_joint_name in ['astribot_gripper_left_joint_L2', 'astribot_gripper_left_joint_R11']:
                    acc_index=self.joint_name_to_index[acc_joint_name]
                    reindex_data[acc_index] = -left_gripper_qpos        
            elif joint_name == 'astribot_gripper_right_joint_L1':
                right_gripper_qpos = command[joint_id]/ 100 * 0.93
                reindex_data[reindex]=-right_gripper_qpos

                for acc_joint_name in ['astribot_gripper_right_joint_L11', 'astribot_gripper_right_joint_R1', 'astribot_gripper_right_joint_R2']:
                    acc_index=self.joint_name_to_index[acc_joint_name]
                    reindex_data[acc_index] = right_gripper_qpos
                for acc_joint_name in ['astribot_gripper_right_joint_L2', 'astribot_gripper_right_joint_R11']:
                    acc_index=self.joint_name_to_index[acc_joint_name]
                    reindex_data[acc_index] = -right_gripper_qpos
            elif 'gripper' not in joint_name:
                reindex_data[reindex]=command[joint_id]

        return reindex_data
    
    def reindex_string_data(self, command):
        reindex_data=[""] * 32
        for joint_id, joint_name in enumerate(self.joint_names_all):
            reindex=self.joint_name_to_index[joint_name]

            if joint_name == 'astribot_gripper_left_joint_L1':
                left_gripper_qpos = command[joint_id]
                reindex_data[reindex]=left_gripper_qpos

                for acc_joint_name in ['astribot_gripper_left_joint_R1', 'astribot_gripper_left_joint_R2', 'astribot_gripper_left_joint_L11']:
                    acc_index=self.joint_name_to_index[acc_joint_name]
                    reindex_data[acc_index] = left_gripper_qpos
                for acc_joint_name in ['astribot_gripper_left_joint_L2', 'astribot_gripper_left_joint_R11']:
                    acc_index=self.joint_name_to_index[acc_joint_name]
                    reindex_data[acc_index] = left_gripper_qpos
            elif joint_name == 'astribot_gripper_right_joint_L1':
                right_gripper_qpos = command[joint_id]
                reindex_data[reindex]=right_gripper_qpos

                for acc_joint_name in ['astribot_gripper_right_joint_L11', 'astribot_gripper_right_joint_R1', 'astribot_gripper_right_joint_R2']:
                    acc_index=self.joint_name_to_index[acc_joint_name]
                    reindex_data[acc_index] = right_gripper_qpos
                for acc_joint_name in ['astribot_gripper_right_joint_L2', 'astribot_gripper_right_joint_R11']:
                    acc_index=self.joint_name_to_index[acc_joint_name]
                    reindex_data[acc_index] = right_gripper_qpos
            elif 'gripper' not in joint_name:
                reindex_data[reindex]=command[joint_id]
        return reindex_data

    def step(self,action=None):
        self.update_reset_flag()
        if self.reset_flag == False:

            self.joint_names_all,self.controller_mode, self.joint_position_command_all, self.joint_velocity_command_all, self.joint_torque_command_all = self.update_joint_states()
            self.joint_names_all,self.controller_mode, self.joint_position_command_all, self.joint_velocity_command_all, self.joint_torque_command_all = self.reindex_states_data()
            
            self.update_object_states()
            self.update_trajectory_pose()
            self.update_com_pose()
            self.update_sensor_states()

            pos_ctrl_data=[]
            force_ctrl_data=[]
            now_position_status=[]
            for joint_id, mode in enumerate(self.controller_mode):
                if mode == 1:
                    if self.gravity_compensation:
                        gravity_torque = self.get_gravity_torque()
                        self.agent.robot.set_qf(gravity_torque)
                    pos_ctrl_data += self.vel_compensation_ctrl(self.joint_position_command_all, self.joint_velocity_command_all, idx=joint_id)
                elif mode == 2:
                    if self.gravity_compensation:
                        gravity_torque = self.get_gravity_torque()
                        self.agent.robot.set_qf(gravity_torque)
                    now_position_status=self.reindex_command_data(self.get_joint_positions(self.joint_names_all))
                    pos_ctrl_data += self.vel_compensation_ctrl(now_position_status, self.joint_velocity_command_all, idx=joint_id)
                elif mode ==3:
                    force_ctrl_data += [self.joint_torque_command_all[joint_id]]
            
            for _ in range(self.frame_skip):
                if pos_ctrl_data:
                    self.agent.robot.set_qpos(pos_ctrl_data)
                if force_ctrl_data:
                    self.agent.robot.set_qf(force_ctrl_data)

                super().step(action=None)
                
            self.render()
            self.update_camera_data()
            
        else:
            self.reset(seed=0)
            self.setup_joint_interface()

            self.reset_flag = False
        
        observation = self._get_obs()
        reward = 0
        terminated = False
        info = self._get_info()

        return observation, reward, terminated, False, info
    
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
    
    def _get_obs(self):
        return self.obj.pose

    def get_body_pose(self):
        pass

    def get_chassis_pose(self):
        return [0, 0, 0, 1, 0, 0, 0]
