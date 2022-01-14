from typing import List

import numpy as np
import pybullet

from quadruped_gym.core.types import RobotActionConfig


class Robot:
    """Class to load A1 robot data in PyBullet and reset robot to initial state"""

    INIT_MOTOR_ANGLES = np.array([0, 0.8, -1.6] * 4)
    INIT_POSITION = [0, 0, 0.37]
    INIT_ORIENTATION = [0, 0, 0, 1]
    URDF_FILENAME = "a1/urdf/a1_black.urdf"
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
