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
"""This file implements the locomotion gym env."""
import collections
from typing import Dict, List, Tuple

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from quadruped_gym.core.types import RobotActionConfig, RobotObservation
from quadruped_gym.gym.sensors import sensor, space_utils
from quadruped_gym.quadruped import a1_pybullet
from quadruped_gym.quadruped.a1_pybullet.robot import Robot
from quadruped_gym.quadruped.a1_pybullet.simulator import SimulationParameters


def _build_action_space(action_config: RobotActionConfig):
    """Builds action space"""
    return spaces.Box(
        action_config.motor_angle_lower_bounds,
        action_config.motor_angle_upper_bounds,
        dtype=np.float32,
    )


class QuadrupedGymEnv(gym.Env):
    """The gym environment for the locomotion task."""

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 100}
    _num_action_repeat = 30

    def __init__(
        self,
        sim_params: SimulationParameters = SimulationParameters(),
        robot_sensors: List[sensor.BoxSpaceSensor] = [],
        env_sensors: List[sensor.BoxSpaceSensor] = [],
        task=None,
    ):
        """Initializes the locomotion gym environment.

        Args:
          gym_config: An instance of LocomotionGymConfig.
          sensors: A list of environmental sensors for observation.
          task: A callable function/class to calculate the reward and termination
            condition. Takes the gym env as the argument when calling.

        Raises:
          ValueError: If the num_action_repeat is less than 1.

        """

        self.seed()
        self._robot_sensors = robot_sensors
        self._env_sensors = env_sensors
        self._task = task

        self._simulator = a1_pybullet.A1PyBulletSimulator(sim_params)

        # The action list contains the name of all actions.
        self.action_space = _build_action_space(self._simulator.robot.action_config)
        self.observation_space = space_utils.convert_sensors_to_gym_space_dictionary(self.all_sensors())

    @property
    def env_time_step(self):
        return self._num_action_repeat * self._simulator.sim_params.sim_time_step_s

    @property
    def last_robot_obs(self) -> RobotObservation:
        return self._last_robot_obs

    @property
    def last_env_obs(self) -> Dict[str, np.ndarray]:
        """Return dictionary of most recent sensor values"""
        return self._last_env_obs

    def seed(self, seed=None):
        self.np_random, self.np_random_seed = seeding.np_random(seed)
        return [self.np_random_seed]

    def all_sensors(self):
        """Returns all robot and environmental sensors."""
        return self._robot_sensors + self._env_sensors

    def sensor_by_name(self, name):
        """Returns the sensor with the given name, or None if not exist."""
        for sensor_ in self.all_sensors():
            if sensor_.get_name() == name:
                return sensor_
        return None

    def reset(self):
        """Resets the robot's position in the world or rebuild the sim world.

        Returns:
          A numpy array containing the initial observation after reset.
        """

        self._last_robot_obs: RobotObservation = self._simulator.reset()
        self._env_step_counter = 0
        self._last_robot_action = np.zeros(self.action_space.shape)

        for s in self.all_sensors():
            s.on_reset(self)
        # self._get_observation depends on sensors
        self._last_env_obs = self._get_observation()
        # task depends on self._last_env_obs
        if self._task:
            self._task.on_reset(self)

        return self._last_env_obs

    def step(self, action: np.ndarray):
        """Step forward the simulation, given the action.

        Args:
          action: A list of desired motor angles for all motors

        Returns:
          observations: The observation dictionary. The keys are the sensor names
            and the values are the sensor readings.
          reward: The reward for the current state-action pair.
          done: Whether the episode has ended.
          info: A dictionary that stores diagnostic information.
        """

        robot_action = RobotAction(
            
        )
        self._last_robot_obs: RobotObservation = self._simulator.step(action, n_repeats=self._num_action_repeat)
        self._env_step_counter += 1
        self._last_robot_action = action

        for s in self.all_sensors():
            s.on_step(self)
        self._last_env_obs = self._get_observation()
        if self._task:
            self._task.on_step(self)

        reward, reward_components = self._reward()
        done = self._termination()
        return self._last_env_obs, reward, done, {"reward_components": reward_components}

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            raise ValueError("Unsupported render mode:{}".format(mode))
        return self._simulator.render()

    def _termination(self) -> bool:
        """Computes whether the episode is over"""
        if self._task:
            return self._task.is_done()
        else:
            return False

    def _reward(self) -> Tuple[float, Dict[str, float]]:
        if self._task:
            return self._task.reward()
        else:
            return 0, {}

    def _get_observation(self):
        """Get observation of this environment from a list of sensors.

        Returns:
          observations: sensory observation in the numpy array format
        """
        sensors_dict = {}
        for s in self.all_sensors():
            sensors_dict[s.get_name()] = s.get_observation()

        observations = collections.OrderedDict(list(sensors_dict.items()))
        return observations
