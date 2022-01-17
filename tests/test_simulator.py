import numpy as np

from quadruped_gym.core.types import RobotAction
from quadruped_gym.quadruped.a1_pybullet import A1PyBulletSimulator
from quadruped_gym.quadruped.a1_pybullet.simulator import SimulationParameters as A1PyBulletSimulationParameters


def test_simulator():
    simulator = A1PyBulletSimulator(A1PyBulletSimulationParameters(enable_rendering=False))
    action = RobotAction(
        desired_motor_angles=simulator._robot.INIT_MOTOR_ANGLES,
        desired_motor_velocities=0,
        position_gain=5,
        velocity_gain=1,
        additional_torques=np.zeros(12),
    )
    for i in range(300):
        simulator.step(action)
