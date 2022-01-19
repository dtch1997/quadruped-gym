"""A model based controller framework."""


from typing import Any, Dict, Tuple

import numpy as np

from quadruped_gym.agents.whole_body_controller.com_velocity_estimator import COMVelocityEstimator
from quadruped_gym.agents.whole_body_controller.openloop_gait_generator import OpenloopGaitGenerator
from quadruped_gym.agents.whole_body_controller.raibert_swing_leg_controller import RaibertSwingLegController
from quadruped_gym.agents.whole_body_controller.torque_stance_leg_controller import TorqueStanceLegController
from quadruped_gym.core.simulator import Controller, Simulator
from quadruped_gym.core.types import RobotAction, RobotObservation


class LocomotionController(Controller):
    """Generates the quadruped locomotion.

    The actual effect of this controller depends on the composition of each
    individual subcomponent.

    """

    def __init__(
        self,
        simulator: Simulator,
        gait_generator: OpenloopGaitGenerator,
        state_estimator: COMVelocityEstimator,
        swing_leg_controller: RaibertSwingLegController,
        stance_leg_controller: TorqueStanceLegController,
    ):
        """Initializes the class.

        Args:
          robot: A robot instance.
          gait_generator: Generates the leg swing/stance pattern.
          state_estimator: Estimates the state of the robot (e.g. center of mass
            position or velocity that may not be observable from sensors).
          swing_leg_controller: Generates motor actions for swing legs.
          stance_leg_controller: Generates motor actions for stance legs.
          clock: A real or fake clock source.
        """
        self._simulator = simulator
        self._gait_generator = gait_generator
        self._state_estimator = state_estimator
        self._swing_leg_controller = swing_leg_controller
        self._stance_leg_controller = stance_leg_controller

    @property
    def swing_leg_controller(self):
        return self._swing_leg_controller

    @property
    def stance_leg_controller(self):
        return self._stance_leg_controller

    @property
    def gait_generator(self):
        return self._gait_generator

    @property
    def state_estimator(self):
        return self._state_estimator

    def reset(self):
        self._gait_generator.reset()
        self._state_estimator.reset()
        self._swing_leg_controller.reset()
        self._stance_leg_controller.reset()

    def update(self, robot_obs: RobotObservation, current_time: float):
        self._gait_generator.update(robot_obs, current_time)
        self._state_estimator.update(robot_obs)
        self._swing_leg_controller.update(robot_obs)
        self._stance_leg_controller.update(robot_obs)

    def get_action(self) -> Tuple[RobotAction, Dict[str, Any]]:
        """Returns the control ouputs (e.g. positions/torques) for all motors."""
        swing_action = self._swing_leg_controller.get_action()
        stance_action, qp_sol = self._stance_leg_controller.get_action()

        action = RobotAction.zeros(self._simulator.robot_kinematics.NUM_MOTORS)
        for motor_idx in range(self._simulator.robot_kinematics.NUM_MOTORS):
            if motor_idx in swing_action:
                sw_act = swing_action[motor_idx]
                action.desired_motor_angles[motor_idx] = sw_act[0]
                action.position_gain[motor_idx] = sw_act[1]
                action.desired_motor_velocities[motor_idx] = sw_act[2]
                action.velocity_gain[motor_idx] = sw_act[3]
                action.additional_torques[motor_idx] = sw_act[4]
            else:
                assert motor_idx in stance_action
                st_act = stance_action[motor_idx]
                action.desired_motor_angles[motor_idx] = st_act[0]
                action.position_gain[motor_idx] = st_act[1]
                action.desired_motor_velocities[motor_idx] = st_act[2]
                action.velocity_gain[motor_idx] = st_act[3]
                action.additional_torques[motor_idx] = st_act[4]

        return action, dict(qp_sol=qp_sol)
