import gym
import numpy as np

import quadruped_gym.gym  # noqa: F401

env = gym.make('A1BulletGymEnv-v0')
obs = env.reset()
for i in range(1000):
    action = np.zeros(12)
    obs, reward, done, info = env.step(action)
    print(done)
    # print(i, obs, reward, done, info)
env.close()
