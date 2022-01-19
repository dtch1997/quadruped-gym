from typing import List

import numba
import numpy as np
import pybullet

from quadruped_gym.core.types import RobotActionConfig, RobotObservation

class RobotKinematics:

    NUM_MOTORS = 12
    NUM_LEGS = 4
    NUM_MOTORS_PER_LEG = 3
    DEFAULT_HIP_POSITIONS = (
        (0.17, -0.135, 0),
        (0.17, 0.13, 0),
        (-0.195, -0.135, 0),
        (-0.195, 0.13, 0),
    )

    # TODO: Figure out what these mean
    COM_OFFSET = -np.array([0.012731, 0.002186, 0.000515])
    HIP_OFFSETS = (
        np.array(
            [
                [0.183, -0.047, 0.0],
                [0.183, 0.047, 0.0],
                [-0.183, -0.047, 0.0],
                [-0.183, 0.047, 0.0],
            ]
        )
        + COM_OFFSET
    )

    # MPC control values
    # At high replanning frequency, inaccurate values of BODY_MASS/INERTIA
    # doesn't seem to matter much. However, these values should be better tuned
    # when the replan frequency is low (e.g. using a less beefy CPU).
    MPC_BODY_MASS = 108 / 9.8
    MPC_BODY_INERTIA = np.array((0.017, 0, 0, 0, 0.057, 0, 0, 0, 0.064)) * 4.0
    MPC_BODY_HEIGHT = 0.24
    MPC_VELOCITY_MULTIPLIER = 0.5

    @classmethod
    def get_hip_positions(cls, obs: RobotObservation):
        # This is the default implementation in original code which just assumes constant hip positions
        del obs
        return cls.DEFAULT_HIP_POSITIONS

    @classmethod
    def compute_motor_angles_from_foot_position(cls, leg_id: int, foot_local_position: np.ndarray):
        joint_position_idxs = list(range(leg_id * cls.NUM_MOTORS_PER_LEG, (leg_id + 1) * cls.NUM_MOTORS_PER_LEG))

        joint_angles = cls.foot_position_in_hip_frame_to_joint_angle(
            foot_local_position - cls.HIP_OFFSETS[leg_id], l_hip_sign=(-1) ** (leg_id + 1)
        )

        # Return the joint index (the same as when calling GetMotorAngles) as well
        # as the angles.
        return joint_position_idxs, joint_angles.tolist()

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def foot_position_in_hip_frame_to_joint_angle(foot_position, l_hip_sign=1):
        l_up = 0.2
        l_low = 0.2
        l_hip = 0.08505 * l_hip_sign
        x, y, z = foot_position[0], foot_position[1], foot_position[2]
        theta_knee = -np.arccos((x ** 2 + y ** 2 + z ** 2 - l_hip ** 2 - l_low ** 2 - l_up ** 2) / (2 * l_low * l_up))
        l = np.sqrt(l_up ** 2 + l_low ** 2 + 2 * l_up * l_low * np.cos(theta_knee))
        theta_hip = np.arcsin(-x / l) - theta_knee / 2
        c1 = l_hip * y - l * np.cos(theta_hip + theta_knee / 2) * z
        s1 = l * np.cos(theta_hip + theta_knee / 2) * y + l_hip * z
        theta_ab = np.arctan2(s1, c1)
        return np.array([theta_ab, theta_hip, theta_knee])

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def foot_position_in_hip_frame(angles, l_hip_sign=1):
        theta_ab, theta_hip, theta_knee = angles[0], angles[1], angles[2]
        l_up = 0.2
        l_low = 0.2
        l_hip = 0.08505 * l_hip_sign
        leg_distance = np.sqrt(l_up ** 2 + l_low ** 2 + 2 * l_up * l_low * np.cos(theta_knee))
        eff_swing = theta_hip + theta_knee / 2

        off_x_hip = -leg_distance * np.sin(eff_swing)
        off_z_hip = -leg_distance * np.cos(eff_swing)
        off_y_hip = l_hip

        off_x = off_x_hip
        off_y = np.cos(theta_ab) * off_y_hip - np.sin(theta_ab) * off_z_hip
        off_z = np.sin(theta_ab) * off_y_hip + np.cos(theta_ab) * off_z_hip
        return np.array([off_x, off_y, off_z])

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def compute_analytical_leg_jacobian(leg_id: int, leg_angles: np.ndarray):
        """
        Computes the analytical Jacobian.
        Args:
          leg_angles: a list of 3 numbers for current abduction, hip and knee angle.
          l_hip_sign: whether it's a left (1) or right(-1) leg.
        """
        l_up = 0.2
        l_low = 0.2
        l_hip = 0.08505 * (-1) ** (leg_id + 1)

        t1, t2, t3 = leg_angles[0], leg_angles[1], leg_angles[2]
        l_eff = np.sqrt(l_up ** 2 + l_low ** 2 + 2 * l_up * l_low * np.cos(t3))
        t_eff = t2 + t3 / 2
        J = np.zeros((3, 3))
        J[0, 0] = 0
        J[0, 1] = -l_eff * np.cos(t_eff)
        J[0, 2] = l_low * l_up * np.sin(t3) * np.sin(t_eff) / l_eff - l_eff * np.cos(t_eff) / 2
        J[1, 0] = -l_hip * np.sin(t1) + l_eff * np.cos(t1) * np.cos(t_eff)
        J[1, 1] = -l_eff * np.sin(t1) * np.sin(t_eff)
        J[1, 2] = (
            -l_low * l_up * np.sin(t1) * np.sin(t3) * np.cos(t_eff) / l_eff - l_eff * np.sin(t1) * np.sin(t_eff) / 2
        )
        J[2, 0] = l_hip * np.cos(t1) + l_eff * np.sin(t1) * np.cos(t_eff)
        J[2, 1] = l_eff * np.sin(t_eff) * np.cos(t1)
        J[2, 2] = (
            l_low * l_up * np.sin(t3) * np.cos(t1) * np.cos(t_eff) / l_eff + l_eff * np.sin(t_eff) * np.cos(t1) / 2
        )
        return J

    @classmethod
    def map_contact_force_to_joint_torques(cls, leg_id: int, contact_force: np.ndarray, leg_angles: np.ndarray):
        """Maps the foot contact force to the leg joint torques."""
        jv = cls.compute_analytical_leg_jacobian(leg_id, leg_angles)
        motor_torques_list = np.matmul(contact_force, jv)
        motor_torques_dict = {}
        for torque_id, joint_id in enumerate(
            range(leg_id * cls.NUM_MOTORS_PER_LEG, (leg_id + 1) * cls.NUM_MOTORS_PER_LEG)
        ):
            motor_torques_dict[joint_id] = motor_torques_list[torque_id]
        return motor_torques_dict

    @classmethod
    def foot_positions_in_base_frame(cls, foot_angles: np.ndarray):
        foot_angles = foot_angles.reshape((cls.NUM_LEGS, cls.NUM_MOTORS_PER_LEG))

        foot_positions = np.zeros((cls.NUM_LEGS, 3))
        for i in range(4):
            foot_positions[i] = cls.foot_position_in_hip_frame(foot_angles[i], l_hip_sign=(-1) ** (i + 1))
        return foot_positions + cls.HIP_OFFSETS

class Robot:
    """Class to load A1 robot data in PyBullet and reset robot to initial state"""

    INIT_MOTOR_ANGLES = np.array([0, 0.8, -1.6] * 4)
    INIT_POSITION = [0, 0, 0.37]
    INIT_ORIENTATION = [0, 0, 0, 1]
    URDF_FILENAME = "a1/urdf/a1_black.urdf"

    kinematics = RobotKinematics

    ### Public API ###

    def __init__(self, pybullet_client):
        self._pybullet_client = pybullet_client

    @property
    def num_motors(self):
        return len(self._motor_id_list)

    @property
    def motor_id_list(self) -> List[int]:
        """Return a list of Pybullet bodyUniqueIds of the robot's motorized joints"""
        return self._motor_id_list

    @property
    def foot_link_id_list(self) -> List[int]:
        return self._foot_link_ids

    @property
    def action_config(self) -> RobotActionConfig:
        return self._action_config

    @property
    def pybullet_client(self):
        """Return the connection to the Pybullet client performing simulation"""
        return self._pybullet_client

    @property
    def quadruped(self) -> int:
        """Return the Pybullet bodyUniqueId of the robot's chassis"""
        return self._quadruped

    def reset(self, reload_urdf=False):
        if reload_urdf:
            self._build_from_urdf()
        else:
            self._pybullet_client.resetBasePositionAndOrientation(
                self.quadruped,
                self.INIT_POSITION,
                self.INIT_ORIENTATION,
            )
            self._pybullet_client.resetBaseVelocity(self.quadruped, [0, 0, 0], [0, 0, 0])
        self._ResetPose()

    ### Private functions ###

    def _build_from_urdf(self):
        self._LoadRobotURDF()
        self._BuildJointNameToIdDict()
        self._BuildUrdfIds()
        self._RemoveDefaultJointDamping()
        self._BuildMotorIdList()
        self._BuildActionConfig()

    def _LoadRobotURDF(self):
        """Loads the URDF file for the robot."""
        self._quadruped = self._pybullet_client.loadURDF(
            self.URDF_FILENAME,
            self.INIT_POSITION,
            self.INIT_ORIENTATION,
            flags=pybullet.URDF_USE_SELF_COLLISION,
            useFixedBase=False,
        )

    def _BuildJointNameToIdDict(self):
        num_joints = self._pybullet_client.getNumJoints(self.quadruped)
        self._joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
            self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

    def _BuildUrdfIds(self):
        """Build the link Ids from its name in the URDF file.
        Raises:
          ValueError: Unknown category of the joint name.
        """
        num_joints = self._pybullet_client.getNumJoints(self.quadruped)
        self._foot_link_ids = []

        self._bracket_link_ids = []
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
            joint_name = joint_info[1].decode("UTF-8")
            joint_id = self._joint_name_to_id[joint_name]
            if "foot" in joint_name:
                self._foot_link_ids.append(joint_id)

        self._foot_link_ids.sort()


    def _RemoveDefaultJointDamping(self):
        num_joints = self._pybullet_client.getNumJoints(self.quadruped)
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
            self._pybullet_client.changeDynamics(joint_info[0], -1, linearDamping=0, angularDamping=0)

    def _BuildMotorIdList(self):
        self._motor_id_list = []
        num_joints = self._pybullet_client.getNumJoints(self.quadruped)
        for i in range(num_joints):
            info = self._pybullet_client.getJointInfo(self.quadruped, i)
            jointType = info[2]
            if jointType == self._pybullet_client.JOINT_PRISMATIC or jointType == self._pybullet_client.JOINT_REVOLUTE:
                self._motor_id_list.append(i)

    def _BuildActionConfig(self):
        position_lbs = []
        position_ubs = []
        force_ubs = []
        velocity_ubs = []
        for motor_id in self.motor_id_list:
            joint_info = self._pybullet_client.getJointInfo(self.quadruped, motor_id)
            position_lb, position_ub, force_ub, velocity_ub = joint_info[8:12]
            position_lbs.append(position_lb)
            position_ubs.append(position_ub)
            force_ubs.append(force_ub)
            velocity_ubs.append(velocity_ub)

        self._action_config = RobotActionConfig(
            motor_angle_lower_bounds=np.array(position_lbs),
            motor_angle_upper_bounds=np.array(position_ubs),
            motor_force_upper_bounds=np.array(force_ubs),
            motor_velocity_upper_bounds=np.array(velocity_ubs),
        )

    def _ResetPose(self):
        for name in self._joint_name_to_id:
            joint_id = self._joint_name_to_id[name]
            self._pybullet_client.setJointMotorControl2(
                bodyIndex=self.quadruped,
                jointIndex=(joint_id),
                controlMode=self._pybullet_client.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0,
            )
        for i, motor_id in enumerate(self.motor_id_list):
            angle = self.INIT_MOTOR_ANGLES[i]
            self._pybullet_client.resetJointState(self.quadruped, motor_id, angle, targetVelocity=0)