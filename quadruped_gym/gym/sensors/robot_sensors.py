# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple sensors related to the robot."""
import typing

import numpy as np

from quadruped_gym.core.types import RobotObservation
from quadruped_gym.gym.sensors import sensor

_ARRAY = typing.Iterable[float]  # pylint: disable=invalid-name
_FLOAT_OR_ARRAY = typing.Union[float, _ARRAY]  # pylint: disable=invalid-name


def _to_local_frame(dx, dy, yaw):
    # Transform the x and y direction distances to the robot's local frame
    dx_local = np.cos(yaw) * dx + np.sin(yaw) * dy
    dy_local = -np.sin(yaw) * dx + np.cos(yaw) * dy
    return dx_local, dy_local


class RobotSensor(sensor.BoxSpaceSensor):
    def __init__(
        self,
        name: str,
        lower_bound: _FLOAT_OR_ARRAY,
        upper_bound: _FLOAT_OR_ARRAY,
        shape: typing.Tuple[int, ...],
        dtype=np.float64,
    ):
        super(RobotSensor, self).__init__(
            name=name, lower_bound=lower_bound, upper_bound=upper_bound, shape=shape, dtype=dtype
        )
        self.robot_obs: RobotObservation = None

    def on_reset(self, env):
        self.on_step(env)

    def on_step(self, env):
        self.robot_obs = env.last_robot_obs


class MotorAngleSensor(RobotSensor):
    """A sensor that reads motor angles from the robot."""

    def __init__(
        self,
        num_motors: int,
    ) -> None:

        super(MotorAngleSensor, self).__init__(
            name="MotorAngle",
            shape=(num_motors,),
            lower_bound=-np.pi,
            upper_bound=np.pi,
        )

    def _get_observation(self) -> _ARRAY:
        return self.robot_obs.motor_angles


class MotorVelocitySensor(RobotSensor):
    """A sensor that reads motor velocities from the robot."""

    def __init__(
        self,
        num_motors: int,
    ) -> None:

        super(MotorVelocitySensor, self).__init__(
            name="MotorVelocity",
            shape=(num_motors,),
            lower_bound=-10,
            upper_bound=10,
        )

    def _get_observation(self) -> _ARRAY:
        return self.robot_obs.motor_velocities


class BaseVelocitySensor(RobotSensor):
    """A sensor that reads the robot's base velocity."""

    def __init__(
        self,
        convert_to_local_frame: bool = False,
        exclude_z: bool = False,
    ) -> None:

        size = 2 if exclude_z else 3
        super(BaseVelocitySensor, self).__init__(
            name="BaseVelocity",
            shape=(size,),
            lower_bound=-100,
            upper_bound=100,
        )

        self._convert_to_local_frame = convert_to_local_frame
        self._exclude_z = exclude_z

    def _get_observation(self) -> _ARRAY:
        vx, vy, vz = self.robot_obs.base_velocity
        current_yaw = self.robot_obs.base_rpy[2]

        if self._convert_to_local_frame:
            vx, vy = _to_local_frame(vx, vy, current_yaw)

        if self._exclude_z:
            return np.array([vx, vy])
        else:
            return np.array([vx, vy, vz])


class IMUSensor(RobotSensor):
    """An IMU sensor that reads orientations and angular velocities."""

    def __init__(self) -> None:
        super(IMUSensor, self).__init__(
            name="IMU",
            shape=(6,),
            lower_bound=[-2 * np.pi] * 3 + [-2000 * np.pi] * 3,
            upper_bound=[2 * np.pi] * 3 + [2000 * np.pi] * 3,
        )

    def _get_observation(self) -> _ARRAY:
        rpy = self.robot_obs.base_rpy
        drpy = self.robot_obs.base_rpy_rate
        return np.concatenate([rpy, drpy])


class TargetDisplacementSensor(RobotSensor):
    """A sensor that reports the target displacement in the robot frame."""

    def __init__(
        self,
        max_distance: float = 0.022,
    ) -> None:
        """Constructs ForwardTargetPositionSensor.
        Args:
            max_distance: The distance away from the robot's base to create the target displacement vector
        """
        self._env = None

        super(TargetDisplacementSensor, self).__init__(
            name="TargetDisplacement",
            shape=(2,),
            lower_bound=-1.0,
            upper_bound=1.0,
        )

        self._max_distance = max_distance

    def _get_observation(self) -> _ARRAY:
        current_base_pos = self.robot_obs.base_position
        current_yaw = self.robot_obs.base_rpy[2]

        # target y position is always zero
        dy_target = 0 - current_base_pos[1]
        # give some leeway for the robot to walk forward
        dy_target = max(min(dy_target, self._max_distance / 2), -self._max_distance / 2)
        # target x position is always forward
        dx_target = np.sqrt(pow(self._max_distance, 2) - pow(dy_target, 2))
        # Transform to local frame
        dx_target_local, dy_target_local = _to_local_frame(dx_target, dy_target, current_yaw)
        return [dx_target_local, dy_target_local]
