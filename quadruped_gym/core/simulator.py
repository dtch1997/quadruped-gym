import numpy as np

from abc import ABC, abstractmethod
from quadruped_gym.core.types import RobotObservation


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
