import gym
import numpy as np

from quadruped_gym.gym.quadruped_gym_env import QuadrupedGymEnv
from quadruped_gym.gym.sensors import robot_sensors
from quadruped_gym.gym.tasks import forward_task_pos
from quadruped_gym.gym.wrappers import wrappers
from quadruped_gym.quadruped.a1_pybullet import simulator


def build_env(
    action_limit=np.array([0.5] * 12),
    render=False,
):

    sim_params = simulator.SimulationParameters(
        enable_rendering=render, n_action_repeat=30, enable_action_filter=True, enable_clip_motor_commands=True
    )

    robot_sensor_list = [
        robot_sensors.BaseVelocitySensor(),
        robot_sensors.IMUSensor(),
        robot_sensors.MotorAngleSensor(12),
        robot_sensors.MotorVelocitySensor(12),
        robot_sensors.TargetDisplacementSensor(),
    ]
    env_sensor_list = []
    task = forward_task_pos.ForwardTask()

    env = QuadrupedGymEnv(
        sim_params=sim_params, robot_sensors=robot_sensor_list, env_sensors=env_sensor_list, task=task
    )

    env = wrappers.ObservationDictionaryToArrayWrapper(env)
    env = wrappers.AddPoseOffsetWrapper(env)
    env = wrappers.ActionLimitWrapper(env, action_limit=action_limit)

    return env


class A1BulletGymEnv(gym.Env):
    """A1 environment that supports the gym interface."""

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, render=False):
        self._env = build_env(render=render)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def step(self, action):
        return self._env.step(action)

    def reset(self):
        return self._env.reset()

    def close(self):
        self._env.close()

    def render(self, mode):
        return self._env.render(mode)

    def __getattr__(self, attr):
        return getattr(self._env, attr)
