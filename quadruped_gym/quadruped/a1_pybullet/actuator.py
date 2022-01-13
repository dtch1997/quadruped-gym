import numpy as np

from quadruped_gym.quadruped.a1_pybullet.robot import Robot


class Actuator:
    def __init__(self):
        pass

    def send_action(self, action, robot: Robot):
        robot._pybullet_client.setJointMotorControlArray(
            bodyIndex=robot.quadruped,
            jointIndices=robot._motor_id_list,
            controlMode=robot._pybullet_client.POSITION_CONTROL,
            targetPositions=action,  # Target joint angles
            forces=100 * np.ones_like(action),  # Max force allowed
        )
