#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024, Astribot Co., Ltd.
# All rights reserved.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------
# Author: Tony Wang, tonywang@astribot.com
# -----------------------------------------------------------------------------

from astribot_envs.astribot_envs_factory import AstribotEnvsFactory

def main():
    # Load param from yaml, create a simulation env using the Factory Pattern
    astribot_yaml_file='config/astribot_s1/simulation_mujoco_param.yaml'
    astribot_envs_factory = AstribotEnvsFactory()
    astribot_data=AstribotEnvsFactory.load_yaml_file(astribot_yaml_file)
    astribot_simulation_thread=astribot_envs_factory.create_simulation_env(astribot_data)

if __name__ == '__main__':
    main()

