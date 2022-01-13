import numpy as np
import pybullet

from quadruped_gym.gym.quadruped_gym_env import QuadrupedGymEnv


def _to_local_frame(dx, dy, yaw):
    # Transform the x and y direction distances to the robot's local frame
    dx_local = np.cos(yaw) * dx + np.sin(yaw) * dy
    dy_local = -np.sin(yaw) * dx + np.cos(yaw) * dy
    return dx_local, dy_local


class ForwardTask(object):
    def on_reset(self, env: QuadrupedGymEnv):
        """Resets the internal state of the task."""
        self._current_base_pos = np.zeros(3)
        self._env_time_step = env.env_time_step
        self.on_step(env)

    def on_step(self, env: QuadrupedGymEnv):
        """Updates the internal state of the task."""
        self._last_base_pos = self._current_base_pos
        self._current_base_pos = env.last_robot_obs.base_position
        self._robot_obs = env.last_robot_obs
        self._target_displacement = env.last_env_obs['TargetDisplacement']

    def is_done(self):
        """Checks if the episode is over.
        If the robot base becomes unstable (based on orientation), the episode
        terminates early.
        """
        return False

    def reward(self):
        """Get the reward without side effects.

        Also return a dict of reward components"""

        # Reward distance travelled in target direction.
        dx, dy, _ = np.array(self._current_base_pos) - np.array(self._last_base_pos)
        dx_local, dy_local = _to_local_frame(dx, dy, self._robot_obs.base_rpy[2])
        dxy_local = np.array([dx_local, dy_local])
        distance_target = np.linalg.norm(self._target_displacement)
        distance_towards = np.dot(dxy_local, self._target_displacement) / distance_target
        distance_reward = min(distance_towards / distance_target, 1)

        # Penalize sideways rotation of the body.
        orientation = self._robot_obs.base_orientation
        rot_matrix = pybullet.getMatrixFromQuaternion(orientation)
        local_up_vec = rot_matrix[6:]
        shake_reward = -abs(np.dot(np.asarray([1, 1, 0]), np.asarray(local_up_vec)))

        # Penalize energy usage.
        energy_reward = (
            -np.abs(np.dot(self._robot_obs.motor_torques, self._robot_obs.motor_velocities)) * self._env_time_step
        )

        # Dictionary of:
        # - {name: reward * weight}
        # for all reward components
        weighted_objectives = {
            "distance": distance_reward * 0.03,
            "shake": shake_reward * 0.001,
            "energy": energy_reward * 0.0005,
        }

        reward = sum([o for o in weighted_objectives.values()])
        return reward, weighted_objectives
