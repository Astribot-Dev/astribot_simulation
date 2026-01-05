from gymnasium.envs.registration import register

register(
    id="astribot_envs/AstribotMujocoEnv-v0",
    entry_point="astribot_envs.astribot_mujoco_env:AstribotMujocoEnv",
    max_episode_steps=300,
    kwargs={'param': {}},
)

register(
    id="astribot_envs/AstribotGenesisEnv-v0",
    entry_point="astribot_envs.astribot_genesis_env:AstribotGenesisEnv",
    max_episode_steps=300,
    kwargs={'param': {}},
)

register(
    id="astribot_envs/AstribotIsaacLabEnv-v0",
    entry_point="astribot_envs.astribot_isaaclab_env:AstribotIsaacLabEnv",
    max_episode_steps=300,
    kwargs={'param': {}},
)