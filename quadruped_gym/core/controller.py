from abc import ABC, abstractmethod

from quadruped_gym.core.types import RobotAction, RobotObservation


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
