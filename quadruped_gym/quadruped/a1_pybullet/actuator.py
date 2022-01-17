from dataclasses import dataclass

import numpy as np

from quadruped_gym.core.filter import ActionFilterButter
from quadruped_gym.core.types import RobotAction, RobotObservation
from quadruped_gym.quadruped.a1_pybullet.robot import Robot


class ActionFilter:
    pass


# TODO: Refactor Actuator to use this configuration dataclass
@dataclass
class ActuatorParams:
    enable_action_filter: bool
    action_filter_low_cut: float
    action_filter_high_cut: float
    action_filter_order: int

    enable_clip_motor_angles: bool
    max_motor_angle_change_rate: float  # radians / second


class Actuator:
    """Applies post-processing to action and performs action."""

    def __init__(
        self,
        enable_action_filter: bool,
        enable_clip_motor_commands: bool,
        sampling_rate: float,
    ):
        self._enable_action_filter = enable_action_filter
        self._enable_clip_motor_commands = enable_clip_motor_commands

        # Only used if boolean configured
        self._action_filter = ActionFilterButter(
            sampling_rate=sampling_rate, num_joints=12, lowcut=0.0, highcut=4.0, order=2
        )

    def reset(self, robot: Robot):
        self._action_filter.reset()
        self._action_filter.init_history(robot.INIT_MOTOR_ANGLES)

    def send_action(self, action: RobotAction, robot: Robot, robot_obs: RobotObservation):

        if self._enable_clip_motor_commands:
            max_angle_change = 0.2
            action.desired_motor_angles = np.clip(
                action.desired_motor_angles,
                action.desired_motor_angles - max_angle_change,
                action.desired_motor_angles + max_angle_change,
            )

        if self._enable_action_filter:
            action.desired_motor_angles = self._action_filter.filter(action.desired_motor_angles)

        robot.pybullet_client.setJointMotorControlArray(
            bodyIndex=robot.quadruped,
            jointIndices=robot.motor_id_list,
            controlMode=robot.pybullet_client.POSITION_CONTROL,
            targetPositions=action.desired_motor_angles,
        )

        """
        motor_torques = action.get_motor_torques(
            current_motor_angles = robot_obs.motor_angles,
            current_motor_velocities = robot_obs.motor_velocities
        )

        robot.pybullet_client.setJointMotorControlArray(
            bodyIndex=robot.quadruped,
            jointIndices=robot.motor_id_list,
            controlMode=robot.pybullet_client.TORQUE_CONTROL,
            forces = motor_torques
        )

        """
