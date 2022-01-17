"""State estimator."""

from typing import Any, Sequence

import numpy as np
import pybullet
from quadruped_gym.core.types import RobotObservation
from quadruped_gym.agents.whole_body_controller.moving_window_filter import MovingWindowFilter

_DEFAULT_WINDOW_SIZE = 20


class COMVelocityEstimator(object):
    """Estimate the CoM velocity using on board sensors.


    Requires knowledge about the base velocity in world frame, which for example
    can be obtained from a MoCap system. This estimator will filter out the high
    frequency noises in the velocity so the results can be used with controllers
    reliably.

    """

    def __init__(
        self,
        window_size: int = _DEFAULT_WINDOW_SIZE,
    ):
        self._window_size = window_size
        self.reset()

    @property
    def com_velocity_body_frame(self) -> Sequence[float]:
        """The base velocity projected in the body aligned inertial frame.

        The body aligned frame is a intertia frame that coincides with the body
        frame, but has a zero relative velocity/angular velocity to the world frame.

        Returns:
          The com velocity in body aligned frame.
        """
        return self._com_velocity_body_frame

    @property
    def com_velocity_world_frame(self) -> Sequence[float]:
        return self._com_velocity_world_frame

    def reset(self):
        # We use a moving window filter to reduce the noise in velocity estimation.
        self._velocity_filter_x = MovingWindowFilter(window_size=self._window_size)
        self._velocity_filter_y = MovingWindowFilter(window_size=self._window_size)
        self._velocity_filter_z = MovingWindowFilter(window_size=self._window_size)
        self._com_velocity_world_frame = np.array((0, 0, 0))
        self._com_velocity_body_frame = np.array((0, 0, 0))

    def update(self, robot_obs: RobotObservation):

        velocity = robot_obs.base_velocity

        vx = self._velocity_filter_x.calculate_average(velocity[0])
        vy = self._velocity_filter_y.calculate_average(velocity[1])
        vz = self._velocity_filter_z.calculate_average(velocity[2])
        self._com_velocity_world_frame = np.array((vx, vy, vz))

        base_orientation = robot_obs.base_orientation
        _, inverse_rotation = pybullet.invertTransform((0, 0, 0), base_orientation)

        (self._com_velocity_body_frame, _,) = pybullet.multiplyTransforms(
            (0, 0, 0), inverse_rotation, self._com_velocity_world_frame, (0, 0, 0, 1)
        )
