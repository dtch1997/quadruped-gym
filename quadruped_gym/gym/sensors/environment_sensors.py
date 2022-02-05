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

"""Simple sensors related to the environment."""
import typing

import numpy as np

from quadruped_gym.gym.quadruped_gym_env import QuadrupedGymEnv
from quadruped_gym.gym.sensors import sensor

_ARRAY = typing.Iterable[float]  # pylint:disable=invalid-name
_FLOAT_OR_ARRAY = typing.Union[float, _ARRAY]  # pylint:disable=invalid-name


class LastActionSensor(sensor.BoxSpaceSensor):
    """A sensor that reports the last action taken."""

    def __init__(
        self,
        num_actions: int,
        lower_bound: _FLOAT_OR_ARRAY = -np.pi,
        upper_bound: _FLOAT_OR_ARRAY = np.pi,
        name: str = "LastAction",
        dtype: 'typing.Type[typing.Any]' = np.float64,
    ) -> None:
        """Constructs LastActionSensor.

        Args:
          num_actions: the number of actions to read
          lower_bound: the lower bound of the actions
          upper_bound: the upper bound of the actions
          name: the name of the sensor
          dtype: data type of sensor value
        """
        self._num_actions = num_actions
        self._last_robot_action = np.zeros(self._num_actions)

        super(LastActionSensor, self).__init__(
            name=name,
            shape=(self._num_actions,),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype,
        )

    def on_reset(self, env: QuadrupedGymEnv) -> None:
        return self.on_step(env)

    def on_step(self, env: QuadrupedGymEnv) -> None:
        self._last_action = env.last_robot_action

    def _get_observation(self) -> _ARRAY:
        """Returns the last action of the environment."""
        return self._last_action.desired_motor_angles
