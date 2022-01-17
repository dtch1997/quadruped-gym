# Lint as: python3
"""A torque based stance controller framework."""

from __future__ import absolute_import, division, print_function

from typing import Any, Sequence, Tuple

import numpy as np
from quadruped_gym.agents.whole_body_controller import gait_generator as gait_generator_lib
from quadruped_gym.agents.whole_body_controller import leg_controller, qp_torque_optimizer
from quadruped_gym.core.simulator import Simulator
from quadruped_gym.core.types import RobotObservation

# import time


_FORCE_DIMENSION = 3
KP = np.array((0.0, 0.0, 100.0, 100.0, 100.0, 0.0))
KD = np.array((40.0, 30.0, 10.0, 10.0, 10.0, 30.0))
MAX_DDQ = np.array((10.0, 10.0, 10.0, 20.0, 20.0, 20.0))
MIN_DDQ = -MAX_DDQ


class TorqueStanceLegController(leg_controller.LegController):
    """A torque based stance leg controller framework.

    Takes in high level parameters like walking speed and turning speed, and
    generates necessary the torques for stance legs.
    """

    def __init__(
        self,
        simulator: Simulator,
        gait_generator: Any,
        state_estimator: Any,
        desired_speed: Tuple[float, float] = (0, 0),
        desired_twisting_speed: float = 0,
        desired_body_height: float = 0.45,
        num_legs: int = 4,
        friction_coeffs: Sequence[float] = (0.45, 0.45, 0.45, 0.45),
    ):
        """Initializes the class.

        Tracks the desired position/velocity of the robot by computing proper joint
        torques using MPC module.

        Args:
          gait_generator: Used to query the locomotion phase and leg states.
          state_estimator: Estimate the robot states (e.g. CoM velocity).
          desired_speed: desired CoM speed in x-y plane.
          desired_twisting_speed: desired CoM rotating speed in z direction.
          desired_body_height: The standing height of the robot.
          body_mass: The total mass of the robot.
          body_inertia: The inertia matrix in the body principle frame. We assume
            the body principle coordinate frame has x-forward and z-up.
          num_legs: The number of legs used for force planning.
          friction_coeffs: The friction coeffs on the contact surfaces.
        """
        self._gait_generator = gait_generator
        self._state_estimator = state_estimator
        self.desired_speed = desired_speed
        self.desired_twisting_speed = desired_twisting_speed

        self._desired_body_height = desired_body_height
        self._num_legs = num_legs
        self._friction_coeffs = np.array(friction_coeffs)
        self._qp_torque_optimizer = qp_torque_optimizer.QPTorqueOptimizer(robot.MPC_BODY_MASS, robot.MPC_BODY_INERTIA)

    def reset(self):
        pass

    def update(self):
        pass

    def _estimate_robot_height(self, robot_obs: RobotObservation):
        base_orientation = robot_obs.base_orientation
        contacts = robot_obs.foot_contacts
        rot_mat = pybullet.getMatrixFromQuaternion(base_orientation)
        rot_mat = np.array(rot_mat).reshape((3, 3))
        foot_positions_world_frame = (rot_mat.dot(robot_obs.foot_positions.T)).T
        useful_heights = contacts * (-foot_positions_world_frame[:, 2])
        return np.sum(useful_heights) / np.sum(contacts)

    def get_action(self, robot_obs: RobotObservation):
        """Computes the torque for stance legs."""
        # Actual q and dq
        contacts = np.array(
            [
                (
                    leg_state
                    in (
                        gait_generator_lib.LegState.STANCE,
                        gait_generator_lib.LegState.EARLY_CONTACT,
                    )
                )
                for leg_state in self._gait_generator.desired_leg_state
            ],
            dtype=np.float64,
        )
        foot_positions = robot_obs.foot_positions_in_base_frame

        robot_com_height = self._estimate_robot_height(contacts, foot_positions)
        robot_com_velocity = self._state_estimator.com_velocity_body_frame
        robot_com_roll_pitch_yaw = robot_obs.base_rpy
        robot_com_roll_pitch_yaw[2] = 0.0  # To prevent yaw drifting
        robot_com_roll_pitch_yaw_rate = robot_obs.base_rpy_rate
        robot_q = np.hstack(([0.0, 0.0, robot_com_height], robot_com_roll_pitch_yaw))
        robot_dq = np.hstack((robot_com_velocity, robot_com_roll_pitch_yaw_rate))
        # Desired q and dq
        desired_com_position = np.array((0.0, 0.0, self._desired_body_height), dtype=np.float64)
        desired_com_velocity = np.array((self.desired_speed[0], self.desired_speed[1], 0.0), dtype=np.float64)
        desired_com_roll_pitch_yaw = np.array((0.0, 0.0, 0.0), dtype=np.float64)
        desired_com_angular_velocity = np.array((0.0, 0.0, self.desired_twisting_speed), dtype=np.float64)
        desired_q = np.hstack((desired_com_position, desired_com_roll_pitch_yaw))
        desired_dq = np.hstack((desired_com_velocity, desired_com_angular_velocity))
        # Desired ddq
        desired_ddq = KP * (desired_q - robot_q) + KD * (desired_dq - robot_dq)
        desired_ddq = np.clip(desired_ddq, MIN_DDQ, MAX_DDQ)
        contact_forces = self._qp_torque_optimizer.compute_contact_force(foot_positions, desired_ddq, contacts=contacts)

        action = {}
        for leg_id, force in enumerate(contact_forces):
            motor_torques = self.simulator.robot.map_contact_force_to_joint_torques(leg_id, force)
            for joint_id, torque in motor_torques.items():
                action[joint_id] = (0, 0, 0, 0, torque)
        return action, contact_forces
