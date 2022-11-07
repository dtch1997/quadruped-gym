import abc
import gym 
import numpy as np

from quadruped_gym.core.types import RobotAction, RobotObservation
from quadruped_gym.core.simulator import Simulator

class GymEnvWrapper(abc.ABC, gym.Env):
    """ Wrap a Simulator as a Gym envionment """
    
    def __init__(self, simulator: Simulator):
        self.simulator = simulator

    @abc.abstractmethod
    def action_fn(self, action: np.ndarray) -> RobotAction:
        pass

    @abc.abstractmethod
    def observation_fn(self, observation: RobotObservation) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _render(self):
        pass

    def step(self, action: np.ndarray):
        robot_action: RobotAction = self.action_fn(action)
        robot_obs: RobotObservation = self.simulator.step(robot_action)
        return self.observation_fn(robot_obs)

    def reset(self):
        robot_obs: RobotObservation = self.simulator.reset()
        return self.observation_fn(robot_obs)

    def render(self):
        return self._render()

class GymAgentWrapper:
    """ Wrap a Controller as a Gym agent """
    pass 
    # TODO

