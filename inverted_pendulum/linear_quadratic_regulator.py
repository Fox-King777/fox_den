"""This module contains functions to calculate the control matrix and apply the state feedback controller to the inverted pendulum system."""

import gym
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg


def calculate_control_matrix(env: gym.Env) -> np.ndarray:
    """This function calculates the control matrix for the inverted pendulum system."""
    # state matrix
    a = env.gravity / (env.length * (4.0 / 3 - env.polemass_length / (env.total_mass)))
    state_matrix = np.array([[0, 1, 0, 0], [0, 0, a, 0], [0, 0, 0, 1], [0, 0, a, 0]])

    # input matrix
    b = -1 / (env.length * (4.0 / 3 - env.polemass_length / (env.total_mass)))
    input_matrix = np.array([[0], [1 / env.total_mass], [0], [b]])

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


def run_linear_quadratic_regulator(env: gym.Env, time_steps=400) -> None:
    """This function runs the inverted pendulum system with the linear quadratic regulator controller."""
    state = env.state
    control_matrix = calculate_control_matrix(env)
    state_history = np.zeros((time_steps, env.observation_space.shape[0]))
    for time_step in range(time_steps):
        action, force = apply_state_controller(control_matrix, env.state)
        force = abs(np.clip(force, -10, 10))
        env.force_mag = force
        state, _, terminated, _, _ = env.step(action)
        state_history[time_step] = state

        if terminated:
            print(f'Terminated at time step {time_step}')
            break

    env.close()
    plot_states_over_time(state_history, np.arange(time_steps))


def plot_states_over_time(state_history: np.ndarray, time_steps: np.ndarray) -> None:
    """This function plots the states of the inverted pendulum system over time."""
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, state_history[:, 0], label='Cart Position')
    plt.plot(time_steps, state_history[:, 1], label='Cart Velocity')
    plt.plot(time_steps, state_history[:, 2], label='Pole Angle')
    plt.plot(time_steps, state_history[:, 3], label='Pole Angular Velocity')
    plt.xlabel('Time Steps')
    plt.ylabel('State Values')
    plt.title('Inverted Pendulum System States Over Time')
    plt.legend()
    plt.show()
