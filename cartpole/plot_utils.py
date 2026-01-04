"""Utility functions for plotting training progress in CartPole experiments."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import display


def setup_matplotlib():
    """Set up matplotlib for interactive plotting in Jupyter notebooks."""
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        pass
    plt.ion()
    return is_ipython


def plot_durations(episode_durations, show_result=False):
    """Plot episode durations during training with a moving average.

    Args:
        episode_durations: List of episode durations/rewards
        show_result: If True, display final result; otherwise show training progress
    """
    is_ipython = 'inline' in matplotlib.get_backend()

    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def plot_states_over_time(
    state_history: np.ndarray,
    time_steps: np.ndarray = None,
    labels: list[str] | None = None,
    title: str = 'CartPole States Over Time',
    figsize: tuple = (10, 5),
) -> None:
    """Plot the states of the CartPole system over time.

    Args:
        state_history: Array of shape (n_steps, n_states) containing state values
        time_steps: Array of time step values (optional, defaults to range(n_steps))
        labels: List of state labels (optional, defaults to CartPole state names)
        title: Plot title
        figsize: Figure size tuple
    """
    if time_steps is None:
        time_steps = np.arange(state_history.shape[0])

    if labels is None:
        labels = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity']

    plt.figure(figsize=figsize)
    for i, label in enumerate(labels):
        if i < state_history.shape[1]:
            plt.plot(time_steps, state_history[:, i], label=label)

    plt.xlabel('Time Steps')
    plt.ylabel('State Values')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
