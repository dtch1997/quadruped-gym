import re
from typing import List

import numpy as np
import pybullet
import pybullet_data

from quadruped_gym.core.types import ScalarField


class Robot:
    """Class to load A1 robot data in PyBullet and reset robot to initial state"""

    NUM_MOTORS = 12
    NUM_LEGS = 4
    MOTOR_NAMES = [
        "FR_hip_joint",
        "FR_upper_joint",
        "FR_lower_joint",
        "FL_hip_joint",
        "FL_upper_joint",
        "FL_lower_joint",
        "RR_hip_joint",
        "RR_upper_joint",
        "RR_lower_joint",
        "RL_hip_joint",
        "RL_upper_joint",
        "RL_lower_joint",
    ]

    INIT_MOTOR_ANGLES = np.array([0, 0.67, -1.25] * NUM_LEGS)
    INIT_POSITION = [0, 0, 0.37]
    INIT_ORIENTATION = [0, 0, 0, 1]

    HIP_NAME_PATTERN = re.compile(r"\w+_hip_\w+")
    UPPER_NAME_PATTERN = re.compile(r"\w+_upper_\w+")
    LOWER_NAME_PATTERN = re.compile(r"\w+_lower_\w+")
    TOE_NAME_PATTERN = re.compile(r"\w+_toe\d*")
    IMU_NAME_PATTERN = re.compile(r"imu\d*")

    URDF_FILENAME = "a1/a1.urdf"

    ACTION_CONFIG = [
        ScalarField(name="FR_hip_motor", upper_bound=0.802851455917, lower_bound=-0.802851455917),
        ScalarField(name="FR_upper_joint", upper_bound=4.18879020479, lower_bound=-1.0471975512),
        ScalarField(
            name="FR_lower_joint",
            upper_bound=-0.916297857297,
            lower_bound=-2.69653369433,
        ),
        ScalarField(name="FL_hip_motor", upper_bound=0.802851455917, lower_bound=-0.802851455917),
        ScalarField(name="FL_upper_joint", upper_bound=4.18879020479, lower_bound=-1.0471975512),
        ScalarField(
            name="FL_lower_joint",
            upper_bound=-0.916297857297,
            lower_bound=-2.69653369433,
        ),
        ScalarField(name="RR_hip_motor", upper_bound=0.802851455917, lower_bound=-0.802851455917),
        ScalarField(name="RR_upper_joint", upper_bound=4.18879020479, lower_bound=-1.0471975512),
        ScalarField(
            name="RR_lower_joint",
            upper_bound=-0.916297857297,
            lower_bound=-2.69653369433,
        ),
        ScalarField(name="RL_hip_motor", upper_bound=0.802851455917, lower_bound=-0.802851455917),
        ScalarField(name="RL_upper_joint", upper_bound=4.18879020479, lower_bound=-1.0471975512),
        ScalarField(
            name="RL_lower_joint",
            upper_bound=-0.916297857297,
            lower_bound=-2.69653369433,
        ),
    ]

    ### Public API ###

    def __init__(self, pybullet_client):
        self._pybullet_client = pybullet_client
        self._pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())

    @property
    def num_motors(self):
        return len(self._motor_id_list)

    @property
    def motor_id_list(self) -> List[int]:
        """Return a list of Pybullet bodyUniqueIds of the robot's motorized joints"""
        return self._motor_id_list

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
        self._RecordMassInfoFromURDF()
        self._RecordInertiaInfoFromURDF()

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
        self._chassis_link_ids = [-1]
        self._hip_link_ids = [-1]
        self._leg_link_ids = []
        self._motor_link_ids = []
        self._lower_link_ids = []
        self._foot_link_ids = []
        self._imu_link_ids = []

        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
            joint_name = joint_info[1].decode("UTF-8")
            joint_id = self._joint_name_to_id[joint_name]
            if self.HIP_NAME_PATTERN.match(joint_name):
                self._hip_link_ids.append(joint_id)
            elif self.UPPER_NAME_PATTERN.match(joint_name):
                self._motor_link_ids.append(joint_id)
            # We either treat the lower leg or the toe as the foot link, depending on
            # the urdf version used.
            elif self.LOWER_NAME_PATTERN.match(joint_name):
                self._lower_link_ids.append(joint_id)
            elif self.TOE_NAME_PATTERN.match(joint_name):
                # assert self._urdf_filename == URDF_WITH_TOES
                self._foot_link_ids.append(joint_id)
            elif self.IMU_NAME_PATTERN.match(joint_name):
                self._imu_link_ids.append(joint_id)
            else:
                raise ValueError("Unknown category of joint %s" % joint_name)

        self._leg_link_ids.extend(self._lower_link_ids)
        self._leg_link_ids.extend(self._foot_link_ids)

        # assert len(self._foot_link_ids) == NUM_LEGS
        self._hip_link_ids.sort()
        self._motor_link_ids.sort()
        self._lower_link_ids.sort()
        self._foot_link_ids.sort()
        self._leg_link_ids.sort()

    def _RemoveDefaultJointDamping(self):
        num_joints = self._pybullet_client.getNumJoints(self.quadruped)
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
            self._pybullet_client.changeDynamics(joint_info[0], -1, linearDamping=0, angularDamping=0)

    def _BuildMotorIdList(self):
        self._motor_id_list = [self._joint_name_to_id[motor_name] for motor_name in self.MOTOR_NAMES]

    def _RecordMassInfoFromURDF(self):
        """Records the mass information from the URDF file."""
        self._base_mass_urdf = []
        for chassis_id in self._chassis_link_ids:
            self._base_mass_urdf.append(self._pybullet_client.getDynamicsInfo(self.quadruped, chassis_id)[0])
        self._leg_masses_urdf = []
        for leg_id in self._leg_link_ids:
            self._leg_masses_urdf.append(self._pybullet_client.getDynamicsInfo(self.quadruped, leg_id)[0])
        for motor_id in self._motor_link_ids:
            self._leg_masses_urdf.append(self._pybullet_client.getDynamicsInfo(self.quadruped, motor_id)[0])

    def _RecordInertiaInfoFromURDF(self):
        """Record the inertia of each body from URDF file."""
        self._link_urdf = []
        num_bodies = self._pybullet_client.getNumJoints(self.quadruped)
        for body_id in range(-1, num_bodies):  # -1 is for the base link.
            inertia = self._pybullet_client.getDynamicsInfo(self.quadruped, body_id)[2]
            self._link_urdf.append(inertia)
        # We need to use id+1 to index self._link_urdf because it has the base
        # (index = -1) at the first element.
        self._base_inertia_urdf = [self._link_urdf[chassis_id + 1] for chassis_id in self._chassis_link_ids]
        self._leg_inertia_urdf = [self._link_urdf[leg_id + 1] for leg_id in self._leg_link_ids]
        self._leg_inertia_urdf.extend([self._link_urdf[motor_id + 1] for motor_id in self._motor_link_ids])

        self._motor_inertia = [self._link_urdf[motor_id + 1] for motor_id in self._motor_id_list]

    def _ResetPose(self):
        for i, name in enumerate(self.MOTOR_NAMES):
            if not ("hip_joint" in name or "upper_joint" in name or "lower_joint" in name):
                raise ValueError("The name %s is not recognized as a motor joint." % name)
            angle = self.INIT_MOTOR_ANGLES[i]
            self._pybullet_client.resetJointState(self.quadruped, self._joint_name_to_id[name], angle, targetVelocity=0)
