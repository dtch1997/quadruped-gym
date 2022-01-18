from dataclasses import dataclass
from time import sleep
from timeit import default_timer as timer

import numpy as np
import pybullet
import pybullet_utils.bullet_client as bullet_client

from quadruped_gym.core.filter import ActionFilterButter
from quadruped_gym.core.simulator import Simulator as BaseSimulator
from quadruped_gym.core.types import RobotAction, RobotObservation
from quadruped_gym.quadruped import data
from quadruped_gym.quadruped.a1_pybullet import A1PyBulletActuator, A1PyBulletPerceptor, A1PyBulletRobot


@dataclass
class SimulationParameters(object):
    """Parameters specific for the pyBullet simulation."""

    # Basic simulation parameters
    sim_time_step_s: float = 0.001
    n_action_repeat: int = 1
    num_bullet_solver_iterations: int = 10
    enable_rendering: bool = False
    # Parameters concerning camera rendering
    camera_distance: float = 1.0
    camera_yaw: float = 0
    camera_pitch: float = -30
    render_width: int = 480
    render_height: int = 360
    # Flags concerning action post-processing
    # TODO: Investigate how these are implemented
    enable_action_filter: bool = False
    enable_clip_motor_commands: bool = False
    clip_max_angle_change: float = 0.2


class Simulator(BaseSimulator):
    def __init__(self, sim_params: SimulationParameters):
        self.sim_params = sim_params
        self._pybullet_client = self._init_pybullet_client()
        self._last_frame_time = 0.0
        # Only used if boolean configured
        self._action_filter = ActionFilterButter(
            sampling_rate=1 / (sim_params.sim_time_step_s * sim_params.n_action_repeat),
            num_joints=12,
            lowcut=0.0,
            highcut=4.0,
            order=2,
        )

        self._robot = A1PyBulletRobot(self._pybullet_client)
        self._robot_perceptor = A1PyBulletPerceptor()
        self._robot_actuator = A1PyBulletActuator()
        self.reset(hard_reset=True)

    @property
    def robot(self):
        return self._robot

    @property
    def robot_kinematics(self):
        return self._robot_kinematics

    def reset(self, hard_reset=False) -> RobotObservation:
        """Reset the simulation"""

        if self.sim_params.enable_rendering:
            self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 0)

        if hard_reset:
            self._pybullet_client.resetSimulation()
            self._pybullet_client.setPhysicsEngineParameter(
                numSolverIterations=self.sim_params.num_bullet_solver_iterations
            )
            self._pybullet_client.setTimeStep(self.sim_params.sim_time_step_s)
            self._pybullet_client.setGravity(0, 0, -9.81)
            self._world_dict = {"ground": self._pybullet_client.loadURDF("plane_implicit.urdf")}

        self._robot.reset(reload_urdf=hard_reset)
        self._robot_perceptor.reset(self._robot)
        self._robot_perceptor.receive_observation(self._robot)
        robot_obs = self._robot_perceptor.get_observation()

        self._robot_actuator.reset(self._robot)
        self._action_filter.reset()
        self._action_filter.init_history(robot_obs.motor_angles)

        self._pybullet_client.setPhysicsEngineParameter(enableConeFriction=0)

        if self.sim_params.enable_rendering:
            self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 1)

        return robot_obs

    def step(self, action: RobotAction) -> RobotObservation:
        """Performs one or more simulation steps"""

        if self.sim_params.enable_rendering:
            # Sleep, otherwise the computation takes less time than real time,
            # which will make the visualization like a fast-forward video.
            current_time = timer()
            time_spent = current_time - self._last_frame_time
            self._last_frame_time = current_time
            time_to_sleep = self.sim_params.sim_time_step_s * self.sim_params.n_action_repeat - time_spent
            if time_to_sleep > 0:
                sleep(time_to_sleep)

            # Also keep the previous orientation of the camera set by the user.
            [yaw, pitch, dist] = self._pybullet_client.getDebugVisualizerCamera()[8:11]
            base_pos = self._robot_perceptor.get_base_position()
            self._pybullet_client.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)
            self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

        prev_obs = self._robot_perceptor.get_observation()

        # Action filtering
        if self.sim_params.enable_action_filter:
            action.desired_motor_angles = self._action_filter.filter(action.desired_motor_angles)

        for _ in range(self.sim_params.n_action_repeat):
            self._robot_actuator.send_action(action, self._robot, prev_obs)
            self._robot.pybullet_client.stepSimulation()
        self._robot_perceptor.receive_observation(self._robot)
        return self._robot_perceptor.get_observation()

    def observe(self) -> RobotObservation:
        """Return the most recent robot observation"""
        return self._robot_perceptor.get_observation()

    def render(self):
        base_pos = self._robot_perceptor.get_base_position()
        view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(base_pos[0], base_pos[1], 0),
            distance=self.sim_params.camera_distance,
            yaw=self.sim_params.camera_yaw,
            pitch=self.sim_params.camera_pitch,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self.sim_params.render_width) / self.sim_params.render_height,
            nearVal=0.1,
            farVal=100.0,
        )
        (w, h, px, _, _) = self._pybullet_client.getCameraImage(
            width=self.sim_params.render_width,
            height=self.sim_params.render_height,
            renderer=self._pybullet_client.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
        )

        rgb_array = np.array(px)
        rgb_array = rgb_array.reshape(h, w, 4)
        rgb_array = rgb_array.astype(np.uint8)
        return rgb_array

    def _init_pybullet_client(self):
        if self.sim_params.enable_rendering:
            pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, enable=True)
        else:
            pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)

        pybullet_client.setAdditionalSearchPath(data.get_data_path())
        return pybullet_client
