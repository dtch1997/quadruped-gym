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
"""Converts a list of sensors to gym space."""
import typing

import gym
import numpy as np
from gym import spaces

from quadruped_gym.gym.sensors import sensor


class UnsupportedConversionError(Exception):
    """An exception when the function cannot convert sensors to gym space."""


def convert_sensors_to_gym_space_dictionary(
    sensors: typing.List[sensor.Sensor],
) -> gym.Space:
    """Convert a list of sensors to the corresponding gym space dictionary.

    Args:
      sensors: a list of the current sensors

    Returns:
      space: the converted gym space dictionary

    Raises:
      UnsupportedConversionError: raises when the function cannot convert the
        given list of sensors.
    """
    gym_space_dict = {}
    for s in sensors:
        if isinstance(s, sensor.BoxSpaceSensor):
            gym_space_dict[s.get_name()] = spaces.Box(
                np.array(s.get_lower_bound()),
                np.array(s.get_upper_bound()),
                dtype=np.float64,
            )
        else:
            raise UnsupportedConversionError("sensors = " + str(sensors))
    return spaces.Dict(gym_space_dict)
