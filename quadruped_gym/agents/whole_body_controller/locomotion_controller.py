"""A model based controller framework."""


import numpy as np
from quadruped_gym.agents.whole_body_controller.com_velocity_estimator import COMVelocityEstimator
from quadruped_gym.agents.whole_body_controller.openloop_gait_generator import OpenloopGaitGenerator
from quadruped_gym.agents.whole_body_controller.raibert_swing_leg_controller import RaibertSwingLegController
from quadruped_gym.agents.whole_body_controller.torque_stance_leg_controller import TorqueStanceLegController
from quadruped_gym.core.simulator import Simulator
from quadruped_gym.core.types import RobotObservation

class LocomotionController(object):
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

    def reset(self, robot_obs: RobotObservation):
        self._gait_generator.reset()
        self._state_estimator.reset()
        self._swing_leg_controller.reset(robot_obs)
        self._stance_leg_controller.reset()

    def update(self, robot_obs: RobotObservation, current_time: float):
        self._gait_generator.update(robot_obs, current_time)
        self._state_estimator.update(robot_obs)
        self._swing_leg_controller.update(robot_obs)
        self._stance_leg_controller.update(robot_obs)

    def get_action(self):
        """Returns the control ouputs (e.g. positions/torques) for all motors."""
        swing_action = self._swing_leg_controller.get_action()
        stance_action, qp_sol = self._stance_leg_controller.get_action()
        action = []
        for joint_id in range(self._robot.num_motors):
            if joint_id in swing_action:
                action.extend(swing_action[joint_id])
            else:
                assert joint_id in stance_action
                action.extend(stance_action[joint_id])
        action = np.array(action, dtype=np.float32)

        return action, dict(qp_sol=qp_sol)
