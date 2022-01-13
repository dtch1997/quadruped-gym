from dataclasses import dataclass

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
    base_rpy: np.ndarray
    # Base velocity state
    base_velocity: np.ndarray
    base_rpy_rate: np.ndarray
    # Motor state
    motor_angles: np.ndarray
    motor_velocities: np.ndarray
    motor_torques: np.ndarray
