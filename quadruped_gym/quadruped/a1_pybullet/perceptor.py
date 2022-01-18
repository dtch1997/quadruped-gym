import numpy as np

from quadruped_gym.core.types import RobotObservation
from quadruped_gym.quadruped.a1_pybullet.robot import Robot


def _transform_angular_velocity_to_local_frame(angular_velocity, orientation, pybullet_client):
    _, orientation_inversed = pybullet_client.invertTransform([0, 0, 0], orientation)
    # Transform the angular_velocity at neutral orientation using a neutral
    # translation and reverse of the given orientation.
    relative_velocity, _ = pybullet_client.multiplyTransforms(
        [0, 0, 0],
        orientation_inversed,
        angular_velocity,
        pybullet_client.getQuaternionFromEuler([0, 0, 0]),
    )
    return relative_velocity


class Perceptor:
    """A class for re-computing the robot state from the PyBullet simulation"""

    def __init__(self):
        pass

    def reset(self, robot: Robot):
        _, self._init_orientation_inv = robot.pybullet_client.invertTransform(
            position=[0, 0, 0], orientation=robot.INIT_ORIENTATION
        )
        self._get_euler_from_quaternion = robot.pybullet_client.getEulerFromQuaternion

    def receive_observation(self, robot: Robot):
        """Compute the robot observations from the simulation"""
        # Update motor angles, velocities, and torques
        self._joint_states = robot.pybullet_client.getJointStates(robot.quadruped, robot.motor_id_list)
        # Update robot base position and orientation
        (
            self._base_position,
            orientation,
        ) = robot.pybullet_client.getBasePositionAndOrientation(robot.quadruped)
        _, self._base_orientation = robot.pybullet_client.multiplyTransforms(
            positionA=(0, 0, 0),
            orientationA=orientation,
            positionB=(0, 0, 0),
            orientationB=self._init_orientation_inv,
        )
        # Update base velocity and angular velocity
        angular_velocity_world, self._base_velocity = robot.pybullet_client.getBaseVelocity(robot.quadruped)
        self._angular_velocity_base = _transform_angular_velocity_to_local_frame(
            angular_velocity_world, self.get_base_orientation(), robot.pybullet_client
        )

        # Update foot contact state and contact force


    def get_observation(self) -> RobotObservation:
        """Return the robot observations without affecting the simulation"""
        return RobotObservation(
            base_position=self.get_base_position(),
            base_orientation=self.get_base_orientation(),
            base_rpy=self.get_base_roll_pitch_yaw(),
            base_velocity=self.get_base_velocity(),
            base_rpy_rate=self.get_base_roll_pitch_yaw_rate(),
            motor_angles=self.get_motor_angles(),
            motor_velocities=self.get_motor_velocities(),
            motor_torques=self.get_motor_torques(),
            foot_contacts=None,
            foot_positions=None
        )

    def get_base_position(self) -> np.ndarray:
        return np.array(self._base_position)

    def get_base_velocity(self) -> np.ndarray:
        return np.array(self._base_velocity)

    def get_base_orientation(self) -> np.ndarray:
        return np.array(self._base_orientation)

    def get_base_roll_pitch_yaw(self) -> np.ndarray:
        orientation = self.get_base_orientation()
        return np.array(self._get_euler_from_quaternion(orientation))

    def get_base_roll_pitch_yaw_rate(self) -> np.ndarray:
        return np.array(self._angular_velocity_base)

    def get_motor_angles(self) -> np.ndarray:
        return np.array([state[0] for state in self._joint_states])

    def get_motor_velocities(self) -> np.ndarray:
        return np.array([state[1] for state in self._joint_states])

    def get_motor_torques(self) -> np.ndarray:
        return np.array([state[3] for state in self._joint_states])

    def get_foot_contacts(self) -> np.ndarray:
        pass 

    def get_foot_positions(self) -> np.ndarray:
        pass
