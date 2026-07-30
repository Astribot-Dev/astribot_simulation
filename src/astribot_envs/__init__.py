from gymnasium.envs.registration import register

# Only the supported release backends are registered: MuJoCo and Genesis.
register(
    id="astribot_envs/AstribotMujocoEnv-v0",
    entry_point="astribot_envs.astribot_mujoco_env:AstribotMujocoEnv",
    max_episode_steps=300,
    kwargs={"param": {}},
)

register(
    id="astribot_envs/AstribotGenesisEnv-v0",
    entry_point="astribot_envs.astribot_genesis_env:AstribotGenesisEnv",
    max_episode_steps=300,
    kwargs={"param": {}},
)
