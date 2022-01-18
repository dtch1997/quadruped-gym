from abc import ABC, abstractmethod

import numpy as np

from quadruped_gym.core.types import RobotAction, RobotObservation


class Simulator(ABC):
    @abstractmethod
    def reset(self, hard_reset: bool = False) -> RobotObservation:
        """Reset the simulation"""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: np.ndarray, n_repeats: int = 1) -> RobotObservation:
        """Performs one or more simulation steps"""
        raise NotImplementedError

    @abstractmethod
    def observe(self) -> RobotObservation:
        """Return the most recent robot observation"""
        raise NotImplementedError


class Controller(ABC):
    def __init__(self):
        self._current_time: int = None
        self._robot_obs: RobotObservation = None

    def update(self, current_time: int, robot_obs: RobotObservation):
        self._current_time = current_time
        self._robot_obs = robot_obs

    @abstractmethod
    def get_action(self) -> RobotAction:
        raise NotImplementedError
