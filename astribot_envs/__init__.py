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
    id="astribot_envs/AstribotIsaacsimEnv-v0",
    entry_point="astribot_envs.astribot_isaacsim_env:AstribotIsaacsimEnv",
    max_episode_steps=300,
    kwargs={'param': {}},
)