import numpy as np

from quadruped_gym.quadruped.a1_pybullet import A1PyBulletSimulator
from quadruped_gym.quadruped.a1_pybullet.simulator import SimulationParameters as A1PyBulletSimulationParameters


def test_simulator():
    simulator = A1PyBulletSimulator(A1PyBulletSimulationParameters(enable_rendering=False))
    action = np.array(simulator._robot.INIT_MOTOR_ANGLES)
    for i in range(300):
        simulator.step(action)
