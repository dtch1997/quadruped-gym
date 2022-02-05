import numpy as np

from quadruped_gym.core.types import RobotAction
from quadruped_gym.quadruped.a1_pybullet import A1PyBulletSimulator
from quadruped_gym.quadruped.a1_pybullet.simulator import SimulationParameters as A1PyBulletSimulationParameters


def test_simulator():
    simulator = A1PyBulletSimulator(A1PyBulletSimulationParameters(enable_rendering=False, n_action_repeat=1))
    action = RobotAction(
        desired_motor_angles=simulator._robot.INIT_MOTOR_ANGLES,
        desired_motor_velocities=np.zeros(12),
        position_gain=np.full(12, 100.0),
        velocity_gain=np.full(12, 1.0),
        additional_torques=np.zeros(12),
    )
    for i in range(300):
        obs = simulator.step(action)
        # Check stability
        assert np.all(np.abs(obs.base_rpy) < 0.01), obs.base_rpy
