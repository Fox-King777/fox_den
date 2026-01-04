"""This module contains functions to calculate the control matrix and apply the state feedback controller to the inverted pendulum system."""

import gymnasium as gym
import numpy as np
from scipy import linalg


def calculate_control_matrix(env: gym.Env) -> np.ndarray:
    """This function calculates the control matrix for the inverted pendulum system."""
    # state matrix
    a = env.unwrapped.gravity / (
        env.unwrapped.length
        * (4.0 / 3 - env.unwrapped.polemass_length / (env.unwrapped.total_mass))
    )
    state_matrix = np.array([[0, 1, 0, 0], [0, 0, a, 0], [0, 0, 0, 1], [0, 0, a, 0]])

    # input matrix
    b = -1 / (
        env.unwrapped.length
        * (4.0 / 3 - env.unwrapped.polemass_length / (env.unwrapped.total_mass))
    )
    input_matrix = np.array([[0], [1 / env.unwrapped.total_mass], [0], [b]])

    # Define the performance and actuator weight matrices
    performance_weight_matrix = 5 * np.eye(state_matrix.shape[0])
    actuator_weight_matrix = np.eye(input_matrix.shape[1])

    # Solve Riccati equation
    riccati_solution = linalg.solve_continuous_are(
        state_matrix, input_matrix, performance_weight_matrix, actuator_weight_matrix
    )

    # Calculate the control matrix
    return np.dot(np.linalg.inv(actuator_weight_matrix), np.dot(input_matrix.T, riccati_solution))


def apply_state_controller(control_matrix: np.ndarray, state: np.ndarray) -> tuple:
    """This function applies the state feedback controller to the inverted pendulum system."""
    # feedback controller
    force = -np.dot(control_matrix, state)  # u = -Kx
    if force > 0:
        return 1, force  # if force_dem > 0 -> move cart right

    return 0, force  # if force_dem <= 0 -> move cart left


def run_linear_quadratic_regulator(env: gym.Env, time_steps=400) -> np.ndarray:
    """This function runs the inverted pendulum system with the linear quadratic regulator controller.

    Returns:
        state_history: Array of shape (time_steps, n_states) containing the state trajectory
    """
    state = env.unwrapped.state
    control_matrix = calculate_control_matrix(env)
    state_history = np.zeros((time_steps, env.observation_space.shape[0]))
    for time_step in range(time_steps):
        action, force = apply_state_controller(control_matrix, env.unwrapped.state)
        force = abs(np.clip(force, -10, 10))
        env.unwrapped.force_mag = force
        state, _, terminated, _, _ = env.step(action)
        state_history[time_step] = state

        if terminated:
            print(f'Terminated at time step {time_step}')
            break

    env.close()
    return state_history
