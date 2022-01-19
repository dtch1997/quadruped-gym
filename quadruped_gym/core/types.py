from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np


@dataclass
class ScalarField:
    """A dataclass representing a named, bounded variable"""

    name: str
    upper_bound: float
    lower_bound: float


@dataclass
class RobotObservation:
    # Base position state
    base_position: np.ndarray
    base_orientation: np.ndarray
    base_rpy: np.ndarray  # In the world frame
    # Base velocity state
    base_velocity: np.ndarray  # In the world frame
    base_rpy_rate: np.ndarray
    # Motor state
    motor_angles: np.ndarray
    motor_velocities: np.ndarray
    motor_torques: np.ndarray
    # Foot state
    foot_contacts: np.ndarray  # In the base frame
    foot_positions: np.ndarray  # In the base frame


@dataclass
class RobotActionConfig:
    motor_angle_lower_bounds: np.ndarray
    motor_angle_upper_bounds: np.ndarray
    motor_force_upper_bounds: np.ndarray
    motor_velocity_upper_bounds: np.ndarray


@dataclass
class RobotAction:
    desired_motor_angles: np.ndarray
    desired_motor_velocities: np.ndarray
    position_gain: np.ndarray
    velocity_gain: np.ndarray
    additional_torques: np.ndarray

    @classmethod
    def zeros(shape: Tuple[int, ...]) -> 'RobotAction':
        return RobotAction(
            desired_motor_angles=np.zeros(shape),
            desired_motor_velocities=np.zeros(shape),
            position_gain=np.zeros(shape),
            velocity_gain=np.zeros(shape),
            additional_torques=np.zeros(shape),
        )

    def get_motor_torques(self, current_motor_angles: np.ndarray, current_motor_velocities: np.ndarray) -> np.ndarray:
        motor_torques = (
            self.position_gain * (self.desired_motor_angles - current_motor_angles)
            + self.velocity_gain * (self.desired_motor_velocities - current_motor_velocities)
            + self.additional_torques
        )
        return motor_torques

    def set_value_at_idx(self, idx, value: 'RobotAction'):
        self.desired_motor_angles[idx] = value.desired_motor_angles[idx]
        self.desired_motor_velocities[idx] = value.desired_motor_velocities[idx]
        self.position_gain[idx] = value.position_gain[idx]
        self.velocity_gain[idx] = value.velocity_gain[idx]
        self.additional_torques[idx] = value.additional_torques[idx]


class RobotControlMode(Enum):
    POSITION = 0
    HYBRID = 1
