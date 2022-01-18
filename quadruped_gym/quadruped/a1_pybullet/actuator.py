from quadruped_gym.core.types import RobotAction, RobotObservation
from quadruped_gym.quadruped.a1_pybullet.robot import Robot


class Actuator:
    """Performs action according to motion control mode"""

    def reset(self, robot: Robot):
        pass

    def send_action(self, action: RobotAction, robot: Robot, robot_obs: RobotObservation):

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
