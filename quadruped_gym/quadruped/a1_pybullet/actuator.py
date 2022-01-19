from enum import Enum

from quadruped_gym.core.types import RobotAction, RobotObservation
from quadruped_gym.quadruped.a1_pybullet.robot import Robot


class ControlMode(Enum):
    POSITION = 'position'
    HYBRID = 'hybrid'


class Actuator:
    """Performs action according to motion control mode"""

    def __init__(self, control_mode: ControlMode):
        self.control_mode = control_mode

    def reset(self, robot: Robot):
        pass

    def send_action(self, action: RobotAction, robot: Robot, robot_obs: RobotObservation):

        if self.control_mode == ControlMode.POSITION:

            robot.pybullet_client.setJointMotorControlArray(
                bodyIndex=robot.quadruped,
                jointIndices=robot.motor_id_list,
                controlMode=robot.pybullet_client.POSITION_CONTROL,
                targetPositions=action.desired_motor_angles,
            )

        elif self.control_mode == ControlMode.HYBRID:

            motor_torques = action.get_motor_torques(
                current_motor_angles=robot_obs.motor_angles, current_motor_velocities=robot_obs.motor_velocities
            )

            robot.pybullet_client.setJointMotorControlArray(
                bodyIndex=robot.quadruped,
                jointIndices=robot.motor_id_list,
                controlMode=robot.pybullet_client.TORQUE_CONTROL,
                forces=motor_torques,
            )

        else:

            raise ValueError(f"Unrecognized control mode {self.control_mode}")
