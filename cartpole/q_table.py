"""Q-table agent for discrete state and action spaces."""

import gym
import numpy as np


class QTable:
    """Q-table agent for discrete state and action spaces."""

    def __init__(
        self,
        env: gym.Env,
        buckets: list[np.ndarray],
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        exploration_rate: float = 1.0,
        exploration_decay_rate: float = 0.999,
    ):
        self.env = env
        self.buckets = buckets
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate

        bucket_sizes = [len(bucket) + 1 for bucket in buckets]
        self.q_table = np.zeros(bucket_sizes + [n_actions])

    def discretize(self, state: np.ndarray) -> tuple:
        """Given a continuous state, return the discretized state.

        Args:
            state (np.ndarray): continuous state

        Returns:
            tuple: discretized state
        """
        discrete_state = [int(np.digitize(state[i], self.buckets[i])) for i in range(len(state))]
        return tuple(discrete_state)

    def choose_action(self, state: np.ndarray, explore=True) -> int:
        """Choose an action based on the current state.

        Args:
            state (np.ndarray): current state
            explore (bool, optional): whether to explore or not

        Returns:
            int: action
        """
        state = self.discretize(state)
        # Exploration
        if np.random.random() < self.exploration_rate and explore:
            return self.env.action_space.sample()
        # Exploitation
        return np.argmax(self.q_table[state])

    def update_q_table(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray
    ) -> None:
        """Update the Q-table based on the current state, action, reward, and next state.

        Args:
            state (np.ndarray): current state
            action (int): action
            reward (float): reward
            next_state (np.ndarray): next state
        """
        state = self.discretize(state)
        next_state = self.discretize(next_state)
        self.q_table[state][action] += self.learning_rate * (
            reward
            + self.discount_factor * np.max(self.q_table[next_state])
            - self.q_table[state][action]
        )

    def update_exploration_rate(self) -> None:
        """Update the exploration rate."""
        self.exploration_rate *= self.exploration_decay_rate
        self.exploration_rate = max(0.05, self.exploration_rate)
