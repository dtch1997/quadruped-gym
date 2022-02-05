import gym

import quadruped_gym.gym  # noqa: F401


def test_reset():
    env = gym.make('A1BulletGymEnv-v0')
    observation = env.reset()
    assert observation in env.observation_space


def test_step():
    env = gym.make('A1BulletGymEnv-v0')
    env.reset()
    for _ in range(10):
        env.step(env.action_space.sample())
