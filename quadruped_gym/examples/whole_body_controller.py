from quadruped_gym.agents.whole_body_controller import com_velocity_estimator
from quadruped_gym.agents.whole_body_controller import gait_generator as gait_generator_lib
from quadruped_gym.agents.whole_body_controller import (
    locomotion_controller,
    openloop_gait_generator,
    raibert_swing_leg_controller,
    torque_stance_leg_controller,
)
from quadruped_gym.core.simulator import Simulator
from quadruped_gym.quadruped import a1_pybullet
from quadruped_gym.quadruped.a1_pybullet.simulator import SimulationParameters  # noqa: F401

_STANCE_DURATION_SECONDS = [0.13] * 4  # For faster trotting (v > 1.5 ms reduce this to 0.13s).

# Standing
# _DUTY_FACTOR = [1.] * 4
# _INIT_PHASE_FULL_CYCLE = [0., 0., 0., 0.]

# _INIT_LEG_STATE = (
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
# )

# Tripod
# _DUTY_FACTOR = [.8] * 4
# _INIT_PHASE_FULL_CYCLE = [0., 0.25, 0.5, 0.]

# _INIT_LEG_STATE = (
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.SWING,
# )

# Trotting
_DUTY_FACTOR = [0.6] * 4
_INIT_PHASE_FULL_CYCLE = [0.9, 0, 0, 0.9]

_INIT_LEG_STATE = (
    gait_generator_lib.LegState.SWING,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.SWING,
)


def setup_controller(simulator: Simulator):
    """Demonstrates how to create a locomotion controller."""
    desired_speed = (0, 0)
    desired_twisting_speed = 0

    gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
        simulator=simulator,
        stance_duration=_STANCE_DURATION_SECONDS,
        duty_factor=_DUTY_FACTOR,
        initial_leg_phase=_INIT_PHASE_FULL_CYCLE,
        initial_leg_state=_INIT_LEG_STATE,
    )
    window_size = 60
    state_estimator = com_velocity_estimator.COMVelocityEstimator(window_size=window_size)
    sw_controller = raibert_swing_leg_controller.RaibertSwingLegController(
        gait_generator=gait_generator,
        state_estimator=state_estimator,
        desired_speed=desired_speed,
        desired_twisting_speed=desired_twisting_speed,
        desired_height=simulator.robot_kinematics.MPC_BODY_HEIGHT,
        foot_clearance=0.01,
    )

    st_controller = torque_stance_leg_controller.TorqueStanceLegController(
        simulator=simulator,
        gait_generator=gait_generator,
        state_estimator=state_estimator,
        desired_speed=desired_speed,
        desired_twisting_speed=desired_twisting_speed,
        desired_body_height=simulator.robot_kinematics.MPC_BODY_HEIGHT,
    )

    controller = locomotion_controller.LocomotionController(
        simulator=simulator,
        gait_generator=gait_generator,
        state_estimator=state_estimator,
        swing_leg_controller=sw_controller,
        stance_leg_controller=st_controller,
    )
    return controller


if __name__ == "__main__":
    simulator = a1_pybullet.A1PyBulletSimulator(
        sim_params=SimulationParameters(
            sim_time_step_s=0.001,
            n_action_repeat=1,
            enable_rendering=False,
            enable_clip_motor_commands=False,
            enable_action_filter=False,
        )
    )
    controller = setup_controller(simulator)
    controller.reset()

    print('Successfully set up controller')

    start_time = simulator.time_since_reset
    current_time = start_time
