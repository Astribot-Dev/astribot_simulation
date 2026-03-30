#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025, Astribot Co., Ltd.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------
# Author: Tony Wang, tonywang@astribot.com
# -----------------------------------------------------------------------------

"""
File: chassis_kinematics.py
Brief: chassis kinematics for astribot s0
"""

import numpy as np

class ChassisKinematics:
    def __init__(self, wheel_radius, wheel_distance):
        self.r = wheel_radius
        self.d = wheel_distance

    def motor_to_cartesian(self, motor_data):
        # Assuming motor_data is [omega_m1, omega_m2]
        T = [[self.r/2, self.r/2], [-self.r/self.d, self.r/self.d]]
        v = T[0][0]*motor_data[0] + T[0][1]*motor_data[1]
        omega = T[1][0]*motor_data[0] + T[1][1]*motor_data[1]
        return [v, omega]

    def cartesian_to_motor(self, cartesian_data):
        # Assuming cartesian_data is [v, omega]
        T_inv = [[1/self.r, -self.d/(2*self.r)], [1/self.r, self.d/(2*self.r)]]
        omega_m1 = T_inv[0][0]*cartesian_data[0] + T_inv[0][1]*cartesian_data[1]
        omega_m2 = T_inv[1][0]*cartesian_data[0] + T_inv[1][1]*cartesian_data[1]
        return [omega_m1, omega_m2]

    def torque_motor_to_cartesian(self, motor_torque):
        T = [[1/self.r, 1/self.r], [-self.d/(2*self.r), self.d/(2*self.r)]]
        F = T[0][0]*motor_torque[0] + T[0][1]*motor_torque[1]
        tau = T[1][0]*motor_torque[0] + T[1][1]*motor_torque[1]
        return [F, tau]

    def torque_cartesian_to_motor(self, cartesian_torque):
        T_inv = [[self.r, self.r], [-self.d/2, self.d/2]]
        tau_m1 = T_inv[0][0]*cartesian_torque[0] + T_inv[0][1]*cartesian_torque[1]
        tau_m2 = T_inv[1][0]*cartesian_torque[0] + T_inv[1][1]*cartesian_torque[1]
        return [tau_m1, tau_m2]

class OmniWheelKinematics:
    def __init__(self, wheel_radius, wheel_distance):
        self.r = wheel_radius
        self.d = wheel_distance

    def motor_to_cartesian(self, motor_data):

        T = np.array([
            [np.cos(np.pi / 4), np.sin(np.pi / 4), self.d / 2],
            [-np.cos(np.pi / 4), np.sin(np.pi / 4), self.d / 2],
            [-np.cos(np.pi / 4), -np.sin(np.pi / 4), self.d / 2],
            [np.cos(np.pi / 4), -np.sin(np.pi / 4), self.d / 2]
        ])

        cartesian_data = np.dot(T, motor_data)
        return cartesian_data

    def cartesian_to_motor(self, cartesian_data):

        T_inv = np.array([
            [np.cos(np.pi / 4), -np.cos(np.pi / 4), -np.cos(np.pi / 4), np.cos(np.pi / 4)],
            [np.sin(np.pi / 4), np.sin(np.pi / 4), -np.sin(np.pi / 4), -np.sin(np.pi / 4)],
            [self.d / 2, self.d / 2, self.d / 2, self.d / 2]
        ])

        motor_data = np.dot(np.linalg.pinv(T_inv), cartesian_data)
        return motor_data

    def torque_motor_to_cartesian(self, motor_torque):

        if len(motor_torque) != 4:
            raise ValueError("motor_torque must be a vector of length 4 representing the torques of the 4 wheels")

        T = np.array([
            [1 / self.r, 1 / self.r, 1 / self.r, 1 / self.r],
            [-self.d / (2 * self.r), self.d / (2 * self.r), -self.d / (2 * self.r), self.d / (2 * self.r)]
        ])

        cartesian_torque = np.dot(T, motor_torque)
        return cartesian_torque

    def torque_cartesian_to_motor(self, cartesian_torque):
        T_inv = np.array([
            [self.r, self.r, self.r, self.r],
            [-self.d / 2, self.d / 2, -self.d / 2, self.d / 2]
        ])

        motor_torque = np.dot(np.linalg.pinv(T_inv), cartesian_torque)
        return motor_torque

