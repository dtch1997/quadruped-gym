import gym

gym.envs.register(
    id="A1BulletGymEnv-v0",
    entry_point="quadruped_gym.gym.env_builder:A1BulletGymEnv",
    max_episode_steps=1000,
    reward_threshold=1000.0,
)
