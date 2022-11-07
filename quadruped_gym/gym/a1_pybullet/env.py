import numpy as np
import gym
from gym import spaces

from quadruped_gym.quadruped import a1_pybullet
from quadruped_gym.core import gym_utils
from quadruped_gym.core.types import RobotAction

class A1BulletGymEnv(gym_utils.GymEnvWrapper):
    
    def __init__(self):
        self.sim_params = a1_pybullet.A1PybulletSimulationParameters()
        self.simulator = a1_pybullet.A1PyBulletSimulator(self.sim_params)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(32,)) 
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(12,))

    def observation_fn(self, obs):
        return np.concatenate([
            obs.base_rpy, # 3
            obs.base_rpy_rate, # 3
            obs.base_velocity, # 2
            obs.motor_angles, # 12
            obs.motor_velocities, # 12
        ])

    def action_fn(self, action: np.ndarray):
        return RobotAction(
            desired_motor_angles=action,
            desired_motor_velocities=np.zeros_like(action),
            position_gain = 0.2 * np.ones_like(action), 
            velocity_gain=1.0 * np.ones_like(action),
            additional_torques=np.zeros_like(action)
        )

    def _render(self):
        return self.simulator.render()