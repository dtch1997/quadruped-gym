import numpy as np
import typing

from quadruped_gym.core.types import RobotAction
from quadruped_gym.core.controller import Controller
from quadruped_gym.quadruped import a1_pybullet 

def parse_mocap_data(mocap_path: str) -> typing.Tuple[np.array, np.array]:
    times = [] 
    joint_poses = []
    with open(mocap_path, 'r') as filestream:
        for line in filestream:
            currentline = line.split(",")
            t = [currentline[1]]
            times.append(t)
            joints=currentline[2:14]
            joint_poses.append(joints)
    times = np.array(times, dtype=np.int32)
    times = times - times[0]
    joint_poses = np.array(joint_poses, dtype=np.float64)
    return times, joint_poses

class MocapController(Controller):

    def __init__(self, 
        mocap_path: str,
        controller_kp: float,
        controller_kd: float,
    ):
        super(MocapController, self).__init__()
        self._times, self._joint_poses = parse_mocap_data(mocap_path)
        self._controller_kp = controller_kp
        self._controller_kd = controller_kd

    def get_action(self) -> RobotAction:
        # TODO: Interpolate between frames to find interpolated mocap pose
        # For now: Find the last time step before current time step and use that action
        curr_time_step = self._current_time % (self._times[-1] + 1)
        
        last_pose_index = None
        for i, t in enumerate(self._times):
            last_pose_index = i
            if self._times[i+1] > curr_time_step >= t: break
        
        action = self._joint_poses[last_pose_index]
        return RobotAction(
            desired_motor_angles = action,
            desired_motor_velocities=np.zeros_like(action),
            position_gain=np.full(action.shape, self._controller_kp),
            velocity_gain=np.full(action.shape, self._controller_kd),
            additional_torques=np.zeros_like(action),
        )

if __name__ == "__main__":

    SIM_TIME_STEP = 0.001
    N_ACTION_REPEAT = 10
    N_TIME_STEPS = 1000

    from pathlib import Path
    mocap_path = Path(__file__).absolute().parent / 'mocap.txt'

    simulator = a1_pybullet.simulator.Simulator(
        sim_params=a1_pybullet.simulator.SimulationParameters(
            sim_time_step_s = SIM_TIME_STEP,
            n_action_repeat = N_ACTION_REPEAT,
            enable_rendering=True,
            enable_clip_motor_commands=False,
            enable_action_filter=False)
    )

    controller = MocapController(mocap_path, 0.2, 1.0)

    obs = simulator.reset()
    for i in range(N_TIME_STEPS):
        controller.update(i * simulator.sim_params.n_action_repeat, obs)
        action = controller.get_action()
        obs = simulator.step(action)