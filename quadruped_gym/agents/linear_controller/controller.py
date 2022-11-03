import numpy as np

from quadruped_gym.core.controller import Controller
from quadruped_gym.core.types import RobotAction


class LinearController(Controller):
    def __init__(
        self,
        init_pose: np.ndarray,
        final_pose: np.ndarray,
        controller_kp: float,
        controller_kd: float,
        interpolation_timesteps=100,
    ) -> None:

        super(LinearController, self).__init__()
        self._interpolation_timesteps = interpolation_timesteps
        self._init_pose = init_pose
        self._final_pose = final_pose
        self._controller_kp = controller_kp
        self._controller_kd = controller_kd

    def get_action(self) -> RobotAction:

        time_step = self._current_time % (2 * self._interpolation_timesteps)
        if time_step < self._interpolation_timesteps:
            action = (time_step / self._interpolation_timesteps) * self._final_pose + (
                1 - time_step / self._interpolation_timesteps
            ) * self._init_pose
        else:
            time_step = time_step - self._interpolation_timesteps
            action = (time_step / self._interpolation_timesteps) * self._init_pose + (
                1 - time_step / self._interpolation_timesteps
            ) * self._final_pose

        return RobotAction(
            desired_motor_angles=action,
            desired_motor_velocities=np.zeros_like(action),
            position_gain=np.full(action.shape, self._controller_kp),
            velocity_gain=np.full(action.shape, self._controller_kd),
            additional_torques=np.zeros_like(action),
        )
