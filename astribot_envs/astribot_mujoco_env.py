#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025, Astribot Co., Ltd.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------
# Author: Tony Wang, tonywang@astribot.com
# -----------------------------------------------------------------------------

"""
File: astribot_mujoco_env.py
Brief: mujoco simulation env
"""

import time
import copy
import glfw
import random
import math
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

import mujoco
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

from simu_utils.simu_common_tools import astribot_simu_log
from astribot_envs.astribot_base_env import AstribotBaseEnv

class AstribotMujocoEnv(AstribotBaseEnv):
    def __init__(self, param):

        super().__init__(param)

        astribot_simu_log("Setup astribot base env")
        self.setup_mujoco_model_and_data()

        astribot_simu_log("Setup mujoco renderer for human or camera")
        self.setup_mujoco_render_and_camera()

        astribot_simu_log("Setup joint qpos and qvel mapping")
        self.joint_name_to_qpos_index = self.setup_joint_qpos_mapping()
        self.joint_name_to_qvel_index = self.setup_joint_qvel_mapping()
        self.site_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(self.model.nsite)]

        astribot_simu_log("Backup mujoco actuator")
        self.actuator_dynprm_bak = self.model.actuator_dynprm.copy()
        self.actuator_gainprm_bak = self.model.actuator_gainprm.copy()
        self.actuator_biasprm_bak = self.model.actuator_biasprm.copy()
        self.actuator_ctrlrange_bak = self.model.actuator_ctrlrange.copy()

        self.joint_names_all,self.controller_mode, self.joint_position_command_all, self.joint_velocity_command_all, self.joint_torque_command_all = self.update_joint_states()
        
        self.reset_time = time.time()
        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()
     
    def step(self, action: np.ndarray) -> tuple:
        self.update_reset_flag()
        if self.reset_flag == False:
            step_begin_time=time.time()
            self.joint_names_all,self.controller_mode, self.joint_position_command_all, self.joint_velocity_command_all, self.joint_torque_command_all= self.update_joint_states()
            self.update_object_states()
            self.update_trajectory_pose()
            self.update_com_pose()
            self.update_sensor_states()

            gravity_torque = self.get_gravity_torque()
            self.data.qfrc_applied = np.zeros(self.model.nv)
            self.model.actuator_ctrlrange = self.actuator_ctrlrange_bak.copy()

            temp_ctrl_data=[]
            for joint_id, mode in enumerate(self.controller_mode):
                if mode == 1:
                    if self.gravity_compensation and 'gripper' not in self.joint_names_all[joint_id]:
                            gravity_torque_ids = self.get_joint_id(self.joint_names_all[joint_id])
                            gravity_torque_ids = self.model.jnt_dofadr[gravity_torque_ids]
                            self.set_joint_applied_torques([gravity_torque_ids], gravity_torque)
                    self.set_actuator_parameters(joint_id, gainprm=self.actuator_gainprm_bak[joint_id], biasprm=self.actuator_biasprm_bak[joint_id])

                    temp_ctrl_data += self.vel_compensation_ctrl(self.joint_position_command_all, self.joint_velocity_command_all, idx=joint_id)
                elif mode == 2:
                    if self.gravity_compensation and 'gripper' not in self.joint_names_all[joint_id]:
                            gravity_torque_ids = self.get_joint_id(self.joint_names_all[joint_id])
                            gravity_torque_ids = self.model.jnt_dofadr[gravity_torque_ids]
                            self.set_joint_applied_torques([gravity_torque_ids], gravity_torque)
                    self.set_actuator_parameters(joint_id, gainprm=self.actuator_gainprm_bak[joint_id], biasprm=self.actuator_biasprm_bak[joint_id])

                    now_position_status=self.get_joint_positions(self.joint_names_all)
                    temp_ctrl_data += self.vel_compensation_ctrl(now_position_status, self.joint_velocity_command_all, idx=joint_id)
                elif mode == 3:
                    self.model.actuator_ctrlrange[joint_id] = self.model.actuator_forcerange[joint_id].copy()
                    self.set_actuator_parameters(joint_id, gainprm=np.array([1, 0, 0]), biasprm=np.array([0, 0, 0]))
                    temp_ctrl_data += [self.joint_torque_command_all[joint_id]]

            self.data.ctrl = temp_ctrl_data

            for _ in range(self.frame_skip):
                mujoco.mj_step(self.model, self.data)
            
            self.render()

            step_end_time=time.time()
            self.real_time_fps=1/(step_end_time-step_begin_time)
        else:
            astribot_simu_log("Reset and setup joint interface")
            self.reset()

        observation = self._get_obs()
        reward = 0
        terminated = False
        info = self._get_info()

        return observation, reward, terminated, False, info
    
    def reset(self, seed=None, options=None):
        astribot_simu_log("Reset mujoco data and step")
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)

        self.reset_from_keyframe()
        self.reset_object_pose()
    
        self.reset_flag = False

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def check_joint_control_type(self):

        postion_control_ids = []
        torque_control_ids = []
        actuator_control_ids = []

        actuator_names = self.get_actuator_names()

        for actuator_name in actuator_names:
            if actuator_name:
                split = actuator_name.split("_")
                if len(split) >= 2:
                    if split[1] == "motor":
                        joint_name = actuator_name.replace("motor", "joint")
                        torque_control_ids.append(self.get_joint_id(joint_name))
                        actuator_control_ids.append(self.get_actuator_id(actuator_name))
                    else:
                        joint_name = actuator_name
                        postion_control_ids.append(self.get_joint_id(joint_name))
                        actuator_control_ids.append(self.get_actuator_id(actuator_name))
                else:
                    joint_name = actuator_name
                    postion_control_ids.append(self.get_joint_id(joint_name))

        trntype = self.model.actuator_trntype

        trnid = self.model.actuator_trnid

        ctrl_joints = []
        for i in range(trnid.shape[0]):
            if trntype[i] == 0 and trnid[i, 0] not in ctrl_joints:
                ctrl_joints.append(trnid[i, 0])

        self.controllable_joints = ctrl_joints

        astribot_simu_log("postion_control_ids: ", postion_control_ids)
        astribot_simu_log("torque_control_ids: ", torque_control_ids)
        astribot_simu_log("actuator_control_ids: ", actuator_control_ids)
        return postion_control_ids, torque_control_ids, actuator_control_ids
    
    def _identify_actuators(self, joint_ids):

        pos_actuator_ids = []
        torque_actuator_ids = []

        self._joint_to_actuator_map = {}
        for idx, jnt in enumerate(self.model.actuator_trnid[:, 0].tolist()):
            assert jnt == joint_ids[idx]
            actuator_name = self.get_actuator_name(idx)
            controller_type = actuator_name.split('_')[-1]
            if controller_type == 'torque' or controller_type == 'motor':
                torque_actuator_ids.append(idx)
                assert jnt not in self._joint_to_actuator_map, "Joint {} already has an actuator assigned!".format(self.model.joint_id2name(jnt))
                self._joint_to_actuator_map[jnt] = idx
            elif controller_type == 'position' or controller_type == 'joint':
                pos_actuator_ids.append(idx)
                assert jnt not in self._joint_to_actuator_map, "Joint {} already has an actuator assigned!".format(self.model.joint_id2name(jnt))
                self._joint_to_actuator_map[jnt] = idx
            else:
                pos_actuator_ids.append(idx)
                assert jnt not in self._joint_to_actuator_map, "Joint {} already has an actuator assigned!".format(self.model.joint_id2name(jnt))
                self._joint_to_actuator_map[jnt] = idx
                astribot_simu_log("Unknown actuator type Ignoring. This actuator will not be controllable via PandaArm api.")

        return pos_actuator_ids, torque_actuator_ids
    
    def impedance_control(self, qpos_des, kp=5000, kd=0, qvel_des=None):
        gravity_torque = self.get_gravity_torque()

        control_torque = self.data.ctrl.copy()

        if qvel_des is None:
            qvel_des = np.zeros(self.model.nv)

        for i in range(len(qpos_des)): 
            if i in self.torque_control_ids:
                index = self.torque_control_ids.index(i)
                joint_id = self.actuator_control_ids[index]
                control_torque[i] = kp * (qpos_des[i] - self.data.qpos[joint_id]) + kd * (qvel_des[i] - self.data.qvel[joint_id]) + \
                                    gravity_torque[joint_id]
        return control_torque
    
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
        if self.render_mode=="human":
            self.mujoco_renderer.render("human")
        elif self.render_mode=="rgb_array":
            return self.update_camera_data()

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()

    def reset_from_keyframe(self):
        try:
            self.data.qpos[:] = self.initial_qpos
            self.data.qvel[:] = self.initial_qvel
            astribot_simu_log("Reset from keyframe successfully")
        except:
            astribot_simu_log("reset_from_keyframe none","WARN")
        
        astribot_simu_log("Pressing the backspace key resets the robot simulator state, but can only be triggered once per second")

    def reset_object_pose(self, object_name='object'):
        delta_pos = [random.uniform(0.6, 0.8), random.uniform(-0.2, 0.2), 0.05]
        twist = [0, 0, 0, 0, 0, random.uniform(-50, 50)]
        self.set_body_pose(object_name, delta_pos=delta_pos, twist=twist)

    def setup_mujoco_render_and_camera(self):
        self.mujoco_renderer = MujocoRenderer(
            model=self.model,
            data=self.data,
            width=self.width,
            height=self.height,
            camera_name='human'
        )

        if self.render_mode=="rgb_array":
            self.camera_dict=dict()
            for camera_name in self.camera_names:
                camera_renderer=MujocoRenderer(
                model=self.model,
                data=self.data,
                width=640,
                height=480,
                camera_name=camera_name)
                
                # ERROR::exchange left and right
                if camera_name=='left_wrist_rgbd':
                    self.camera_dict['right_wrist_rgbd'] = camera_renderer
                elif camera_name=='right_wrist_rgbd':
                    self.camera_dict['left_wrist_rgbd'] = camera_renderer
                else:
                    self.camera_dict[camera_name]=camera_renderer

    def setup_mujoco_model_and_data(self):
        astribot_simu_log("Setup mujoco model and data")
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        self.model_for_cal = copy.deepcopy(self.model)
        for i in range(self.model_for_cal.ngeom):
            self.model_for_cal.geom_contype[i] = 0
            self.model_for_cal.geom_conaffinity[i] = 0
        self.data_for_cal = mujoco.MjData(self.model_for_cal)

    def setup_joint_qpos_mapping(self):
        joint_name_to_qpos_index = {}
        for joint_id in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            qpos_index = self.model.jnt_qposadr[joint_id]
            joint_name_to_qpos_index[joint_name] = qpos_index
        return joint_name_to_qpos_index

    def setup_joint_qvel_mapping(self):
        joint_name_to_qvel_index = {}
        for joint_id in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            qvel_index = self.model.jnt_dofadr[joint_id]
            joint_name_to_qvel_index[joint_name] = qvel_index
        return joint_name_to_qvel_index

    def get_contact_state(self):

        contact_num = self.data.ncon
        contact_geom_id_1 = list()
        contact_geom_id_2 = list()
        contact_geom_name_1 = list()
        contact_geom_name_2 = list()
        contact_pose = list()
        contact_force = list()
        contact_dist = list()

        contact_num_new = 0
        for i in range(contact_num):
            contact = self.data.contact[i]

            contact_geom1 = contact.geom1
            contact_geom2 = contact.geom2
            contact_geom1_name = str(self.get_geom_name_from_id(contact.geom1))
            contact_geom2_name = str(self.get_geom_name_from_id(contact.geom2))

            if (contact_geom1_name != 'None') & (contact_geom2_name != 'None'):
                contact_num_new += 1
                contact_dist.append(contact.dist)
                contact_geom_id_1.append(contact_geom1)
                contact_geom_id_2.append(contact_geom2)
                contact_geom_name_1.append(contact_geom1_name)
                contact_geom_name_2.append(contact_geom2_name)
                contact_frame = contact.frame.reshape(3, 3)
                contact_quat = self.from_matrix(contact_frame)

                contact_pose.append(contact.pos.tolist() + contact_quat.tolist())
                c_array = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, i, c_array)
                contact_force.append(c_array)

        return (contact_num_new, contact_geom_id_1, contact_geom_id_2, contact_geom_name_1, contact_geom_name_2,
                contact_pose, contact_force, contact_dist)

    def get_sensor_data(self, sensor_name):
        sensor_data=None
        if 'imu' in sensor_name:
            sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, sensor_name)
            sensor_data = self.data.site_xmat[sensor_id]
        else:
            sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            sensor_data = self.data.sensordata[self.model.sensor_adr[sensor_id]:self.model.sensor_adr[sensor_id]+self.model.sensor_dim[sensor_id]]

        return sensor_data

    def get_geom_names_by_group(self, group):
        names = []
        groups = self.model.geom_group
        for index in range(self.model.ngeom):
            if groups[index] == group:
                names.append(self.get_geom_name_from_id(index))
        return names

    def get_geom_name_from_id(self, id):
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, id)

    def get_ft_sensor_data(self, robot_name):
        force_name = robot_name + "_force_sensor"
        force_data = self.get_sensor_data(force_name)

        torque_name = robot_name + "_torque_sensor"
        torque_data = self.get_sensor_data(torque_name)

        if force_data is None and torque_data is None:
            return None
        else:
            return np.concatenate([force_data,torque_data])
        
    def get_reset_status(self):
        press_status=False

        if hasattr(self.mujoco_renderer.viewer, 'window'):
            if glfw.get_key(self.mujoco_renderer.viewer.window, glfw.KEY_BACKSPACE) == glfw.PRESS:
                press_time = time.time()
                if press_time - self.reset_time > 1:
                    press_status=True
        
        return press_status

    def get_mesh_vertices(self, name):
        mesh_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MESH, name)
        total_vertnum_before = sum(self.model.mesh_vertnum[:mesh_id])
        cur_vertnum = self.model.mesh_vertnum[mesh_id]
        return self.model.mesh_vert[total_vertnum_before:total_vertnum_before + cur_vertnum]
    
    def get_body_names(self):
        body_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(self.model.nbody)]
        return body_names

    def has_body(self, name):
        return name in self.get_body_names()
    
    def get_controllable_joints(self):
        trntype = self.model.actuator_trntype  
        trnid = self.model.actuator_trnid

        mvbl_jnts = []
        for i in range(trnid.shape[0]):
            if trntype[i] == 0 and trnid[i, 0] not in mvbl_jnts:
                mvbl_jnts.append(trnid[i, 0])

        return sorted(mvbl_jnts)

    def get_site_pose(self, name):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        pos = self.data.site_xpos[site_id]
        mat = self.data.site_xmat[site_id].reshape(3, 3)
        quat = self.from_matrix(mat)

        return np.hstack((pos, quat))

    def get_chassis_pose(self):
        if 'chassis' in self.site_names:
            chassis_pose = self.get_site_pose('chassis')
            return chassis_pose
        else:
            return [0, 0, 0, 1, 0, 0, 0]
        
    def get_joint_id(self, joint_name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

    def get_actuator_id(self, actuator_name):
        id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        return id
    
    def get_actuator_names(self):
        actuator_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in
                          range(self.model.njnt)]
        return actuator_names

    def get_joint_position(self, joint_name):
        qpos_index = self.joint_name_to_qpos_index[joint_name]
        joint_id = self.get_joint_id(joint_name)
        if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            return self.data.qpos[qpos_index:qpos_index+3]
        elif 'gripper' in joint_name:
            if self.data.qpos[qpos_index]<0:
                self.data.qpos[qpos_index]=0
            return self.data.qpos[qpos_index] / 0.93 * 100
        else:
            return self.data.qpos[qpos_index]

    def get_joint_positions(self, names):
        pos = []
        for i in names:
           pos.append(self.get_joint_position(i))
        return pos

    def get_joint_velocity(self, joint_name):
        qvel_index = self.joint_name_to_qvel_index[joint_name]
        joint_id = self.get_joint_id(joint_name)
        if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            return self.data.qvel[qvel_index:qvel_index + 3]
        elif 'gripper' in joint_name:
            return self.data.qvel[qvel_index] / 0.93 * 100 
        else:
            return self.data.qvel[qvel_index]

    def get_joint_velocities(self, names):
        vel = []
        for i in names:
           vel.append(self.get_joint_velocity(i))
        return vel

    def get_joint_acceleration(self, joint_name):
        qacc_index = self.joint_name_to_qpos_index[joint_name]
        joint_id = self.get_joint_id(joint_name)
        if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            return self.data.qacc_warmstart[qacc_index:qacc_index + 3]
        elif 'gripper' in joint_name:
            return self.data.qacc_warmstart[qacc_index] / 0.93 * 100   
        else:
            return self.data.qacc_warmstart[qacc_index]

    def get_joint_accelerations(self, names):
        acc = []
        for i in names:
           acc.append(self.get_joint_acceleration(i))
        return acc

    def get_joint_torque(self, joint_name):
        joint_id = self.get_joint_id(joint_name)
        return self.data.qfrc_actuator[self.model.jnt_dofadr[joint_id]]

    def get_joint_torques(self, names):
        tor = []
        for i in names:
            tor.append(self.get_joint_torque(i))
        return tor
    
    def get_joint_applied_torque(self, joint_name):
        qtor_index = self.joint_name_to_qpos_index[joint_name]
        joint_id = self.get_joint_id(joint_name)
        if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            # Return the position part of the free joint
            return self.data.qfrc_applied[qtor_index:qtor_index + 3]
        else:
            return self.data.qfrc_applied[qtor_index]
        
    def get_joint_applied_torques(self):
        return self.data.qfrc_applied

    def get_gravity_torque(self):
        self.data_for_cal.qpos[self.model.jnt_qposadr] = self.data.qpos[self.model.jnt_qposadr].copy()
        self.data_for_cal.qvel[self.get_controllable_joints()] = 0
        self.data_for_cal.qacc[self.get_controllable_joints()] = 0
        mujoco.mj_inverse(self.model_for_cal, self.data_for_cal)
        gravity_torque = self.data_for_cal.qfrc_inverse.copy()
        return gravity_torque

    def get_body_position(self, body_name):
        assert self.has_body(body_name), "Name mistaken"
        return self.data.body(body_name).xpos.copy()
    
    def get_body_pose(self, body_name):
        if not self.has_body(body_name):
            return np.zeros(7, dtype=np.float64)
        return np.hstack((self.get_body_position(body_name), self.get_body_orientation_quat(body_name)))
    
    def get_body_orientation_quat(self, body_name):
        assert self.has_body(body_name), "Name mistaken"
        return self.data.body(body_name).xquat.copy()

    def get_camera_image(self, camera_name='top'):
        
        rgb_img=self.camera_dict[camera_name].render("rgb_array")
        depth_img=self.camera_dict[camera_name].render("depth_array")

        width=640
        height=480
        target_size=(width, height)

        if camera_name=='head_rgbd':
            width=1280
            height=720
            target_size=(width, height)
        elif camera_name in ('left_wrist_rgbd', 'right_wrist_rgbd'):
            width=640
            height=360
            target_size=(width, height)

        rgb_img = cv2.resize(rgb_img, target_size, interpolation=cv2.INTER_LINEAR)
        depth_img = cv2.resize(depth_img, target_size, interpolation=cv2.INTER_LINEAR)

        rgb_img = cv2.flip(rgb_img, 0)
        depth_img=cv2.flip(depth_img, 0)
        point_cloud=self.trans_depth_image_to_point_cloud(depth_img, height, width, camera_name)

        data={}
        data['rgb_img'] = rgb_img
        data['depth_img'] = depth_img
        data['point_cloud'] = point_cloud

        return data
    
    def get_near_and_far(self):
        extent = self.model.stat.extent
        near = self.model.vis.map.znear * extent
        far = self.model.vis.map.zfar * extent

        return near, far
    
    def get_camera_fovy(self, camera_name):
        camid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        return math.radians(self.model.cam_fovy[camid])
    
    def get_camera_transform(self, camera_name):
        camid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

        cam_body_id = self.model.cam_bodyid[camid] 
        camera_position = self.model.body_pos[cam_body_id] 

        temp_trans = np.array([[ 1, 0, 0 ],
                  [ 0, -1, 0 ],
                  [ 0, 0, -1 ]])

        rotation_matrix = np.matmul(np.array(self.model.cam_mat0[camid]).reshape(3, 3), temp_trans)

        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = camera_position

        return transform 

    def set_joint_applied_torques(self, ids, values):
        for i in range(len(ids)):
            self.data.qfrc_applied[ids[i]] = values[ids[i]]

    def set_actuator_parameters(self, actuator_id, dynprm=None, gainprm=None, biasprm=None):
        if dynprm is not None:
            self.model.actuator_dynprm[actuator_id][:len(dynprm)] = dynprm
        if gainprm is not None:
            self.model.actuator_gainprm[actuator_id][:len(gainprm)] = gainprm
        if biasprm is not None:
            self.model.actuator_biasprm[actuator_id][:len(biasprm)] = biasprm

    def set_body_pose(self, name, pose=None, twist=None, delta_pos=None):
        id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

        qposadr = self.model.jnt_qposadr[self.model.body_jntadr[id]]
        qveladr = self.model.jnt_dofadr[self.model.body_jntadr[id]]

        try:
            if delta_pos is not None:
                self.data.qpos[qposadr:qposadr+3] += delta_pos

            if pose is not None:
                self.data.qpos[qposadr:qposadr+7] = pose
            if twist is not None:
                self.data.qvel[qveladr:qveladr+6] = twist
        except:
            pass

    def set_camera_pose(self, camera_name, pose):
        camid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self.model.cam_pos[camid] = pose[:3]
        self.model.cam_quat[camid] = pose[3:]

    def set_mocap_pose_with_id(self, id, pose):
        self.data.mocap_pos[id] = pose[:3]
        self.data.mocap_quat[id] = pose[3:]