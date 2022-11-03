import numpy as np
import gym
import copy

import typing
from quadruped_gym.core.simulator import Controller, Simulator
from quadruped_gym.core.types import RobotObservation, RobotAction

from quadruped_gym.quadruped import a1_pybullet
from quadruped_gym.quadruped.a1_pybullet.simulator import SimulationParameters  # noqa: F401
from quadruped_gym.agents.linear_controller import LinearController

class Logger:

    def __init__(self):
        self.data = []

    def update(self, data: typing.Union[np.ndarray, RobotAction, RobotObservation]):
        self.data.append(data)

    def collate(self):
        if isinstance(self.data[0], np.ndarray):
            all_data = np.stack(self.data, axis=0)
        elif isinstance(self.data[0], RobotAction):
            all_data = RobotAction(
                desired_motor_angles = np.stack([act.desired_motor_angles for act in self.data]),
                desired_motor_velocities = np.stack([act.desired_motor_velocities for act in self.data]),
                position_gain = np.stack([act.position_gain for act in self.data]),
                velocity_gain = np.stack([act.velocity_gain for act in self.data]),
                additional_torques = np.stack([act.additional_torques for act in self.data])
            )
        else:
            all_data = RobotObservation(
                base_position = np.stack([obs.base_position for obs in self.data]),
                base_velocity = np.stack([obs.base_velocity for obs in self.data]),
                base_orientation = np.stack([obs.base_orientation for obs in self.data]),
                base_rpy = np.stack([obs.base_rpy for obs in self.data]),
                base_rpy_rate = np.stack([obs.base_rpy_rate for obs in self.data]),
                motor_angles = np.stack([obs.motor_angles for obs in self.data]),
                motor_velocities = np.stack([obs.motor_velocities for obs in self.data]),
                motor_torques = np.stack([obs.motor_torques for obs in self.data]),
                foot_contacts = np.stack([obs.foot_contacts for obs in self.data]),
                foot_contact_forces = np.stack([obs.foot_contact_forces for obs in self.data]),
                foot_positions = np.stack([obs.foot_positions for obs in self.data]),
            )
        return all_data   

def test_simulation_with_controller(simulator: Simulator, controller: Controller, n_time_steps = 100):
    obs_logger = Logger()
    act_logger = Logger()

    obs = simulator.reset()
    for i in range(n_time_steps):
        controller.update(i * simulator.sim_params.n_action_repeat, obs)
        action = controller.get_action()
        obs_logger.update(copy.deepcopy(obs))
        act_logger.update(copy.deepcopy(action))
        obs = simulator.step(action)

    observation_history = obs_logger.collate()
    action_history = act_logger.collate()
    return observation_history, action_history

if __name__ == "__main__":

    SIM_TIME_STEP = 0.001
    N_ACTION_REPEAT = 10
    N_TIME_STEPS = 1000

    simulator = a1_pybullet.simulator.Simulator(
        sim_params=a1_pybullet.simulator.SimulationParameters(
            sim_time_step_s = SIM_TIME_STEP,
            n_action_repeat = N_ACTION_REPEAT,
            enable_rendering=True,
            enable_clip_motor_commands=False,
            enable_action_filter=False)
    )
    controller = LinearController(
        init_pose = np.array([0, 0.67, -1.3] * 4),
        final_pose = np.array([0, 1.0, -2.0] * 4), 
        controller_kp = 0.2,  
        controller_kd = 1.0,
        interpolation_timesteps = 1000
    )

    obs_hist, act_hist = test_simulation_with_controller(
        simulator, controller, n_time_steps= N_TIME_STEPS)