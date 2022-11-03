import numpy as np
from quadruped_gym.quadruped import a1_pybullet
from quadruped_gym.agents.linear_controller import LinearController

if __name__ == "__main__":

    SIM_TIME_STEP = 0.001
    N_ACTION_REPEAT = 10
    N_TIME_STEPS = 1000

    simulator = a1_pybullet.simulator.Simulator(
        sim_params=a1_pybullet.simulator.SimulationParameters(
            sim_time_step_s = SIM_TIME_STEP,
            n_action_repeat = N_ACTION_REPEAT,
            enable_rendering=True,
            enable_clip_motor_commands=False,
            enable_action_filter=False)
    )
    controller = LinearController(
        init_pose = np.array([0, 0.67, -1.3] * 4),
        final_pose = np.array([0, 1.0, -2.0] * 4), 
        controller_kp = 0.2,  
        controller_kd = 1.0,
        interpolation_timesteps = 1000
    )

    obs = simulator.reset()
    for i in range(N_TIME_STEPS):
        controller.update(i * simulator.sim_params.n_action_repeat, obs)
        action = controller.get_action()
        obs = simulator.step(action)