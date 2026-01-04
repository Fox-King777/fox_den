"""Q-table agent for discrete state and action spaces."""

import gymnasium as gym
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


class TileCoding:
    """Tile coding agent for continuous state spaces with discrete actions.

    Uses multiple overlapping grids (tilings) to represent continuous states,
    providing smoother generalization than simple discretization.
    """

    def __init__(
        self,
        env: gym.Env,
        state_bounds: list[tuple[float, float]],
        n_tilings: int = 8,
        n_tiles_per_dim: int = 8,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        exploration_rate: float = 1.0,
        exploration_decay_rate: float = 0.9995,
    ):
        """Initialize tile coding agent.

        Args:
            env: Gym environment
            state_bounds: List of (min, max) tuples for each state dimension
            n_tilings: Number of overlapping tilings
            n_tiles_per_dim: Number of tiles per dimension in each tiling
            learning_rate: Learning rate for Q-learning updates
            discount_factor: Discount factor for future rewards
            exploration_rate: Initial exploration rate (epsilon)
            exploration_decay_rate: Decay rate for exploration
        """
        self.env = env
        self.n_tilings = n_tilings
        self.n_tiles_per_dim = n_tiles_per_dim
        self.state_bounds = state_bounds
        self.n_state_dims = len(state_bounds)
        self.n_actions = env.action_space.n
        self.learning_rate = learning_rate / n_tilings  # Divide by n_tilings for stability
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate

        # Calculate tile width for each dimension
        self.tile_widths = np.array(
            [(bounds[1] - bounds[0]) / n_tiles_per_dim for bounds in state_bounds]
        )

        # Create offsets for each tiling (uniformly distributed)
        self.tiling_offsets = np.zeros((n_tilings, self.n_state_dims))
        for i in range(n_tilings):
            self.tiling_offsets[i] = (self.tile_widths / n_tilings) * i

        # Initialize weights for each tile-action pair
        # Each tiling has (n_tiles_per_dim)^n_state_dims tiles
        tiles_per_tiling = n_tiles_per_dim**self.n_state_dims
        self.weights = np.zeros((n_tilings, tiles_per_tiling, self.n_actions))

    def get_tile_indices(self, state: np.ndarray) -> list[int]:
        """Get active tile indices for a given state across all tilings.

        Args:
            state: Current state

        Returns:
            List of active tile indices, one per tiling
        """
        tile_indices = []

        for tiling_idx in range(self.n_tilings):
            # Apply offset for this tiling
            offset_state = state - self.tiling_offsets[tiling_idx]

            # Calculate tile coordinates for each dimension
            tile_coords = []
            for dim in range(self.n_state_dims):
                # Normalize to [0, state_bound_range]
                normalized = offset_state[dim] - self.state_bounds[dim][0]
                # Calculate tile index
                tile_idx = int(normalized / self.tile_widths[dim])
                # Clip to valid range
                tile_idx = np.clip(tile_idx, 0, self.n_tiles_per_dim - 1)
                tile_coords.append(tile_idx)

            # Convert multi-dimensional tile coordinates to single index
            tile_index = 0
            for dim, coord in enumerate(tile_coords):
                tile_index += coord * (self.n_tiles_per_dim**dim)

            tile_indices.append(tile_index)

        return tile_indices

    def get_q_value(self, state: np.ndarray, action: int) -> float:
        """Get Q-value for a state-action pair (sum across all active tiles).

        Args:
            state: Current state
            action: Action

        Returns:
            Q-value
        """
        tile_indices = self.get_tile_indices(state)
        return sum(
            self.weights[tiling_idx, tile_idx, action]
            for tiling_idx, tile_idx in enumerate(tile_indices)
        )

    def choose_action(self, state: np.ndarray, explore: bool = True) -> int:
        """Choose an action using epsilon-greedy policy.

        Args:
            state: Current state
            explore: Whether to use exploration

        Returns:
            Selected action
        """
        # Exploration
        if explore and np.random.random() < self.exploration_rate:
            return self.env.action_space.sample()

        # Exploitation - choose action with highest Q-value
        q_values = [self.get_q_value(state, action) for action in range(self.n_actions)]
        return int(np.argmax(q_values))

    def update_q_table(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray
    ) -> None:
        """Update weights using Q-learning rule. Alias for compatibility with QTable interface.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        # Get current Q-value
        current_q = self.get_q_value(state, action)

        # Get max Q-value for next state
        next_q_values = [self.get_q_value(next_state, a) for a in range(self.n_actions)]
        max_next_q = np.max(next_q_values)

        # Calculate TD error
        td_error = reward + self.discount_factor * max_next_q - current_q

        # Update weights for all active tiles
        tile_indices = self.get_tile_indices(state)
        for tiling_idx, tile_idx in enumerate(tile_indices):
            self.weights[tiling_idx, tile_idx, action] += self.learning_rate * td_error

    def update_exploration_rate(self) -> None:
        """Decay exploration rate."""
        self.exploration_rate *= self.exploration_decay_rate
        self.exploration_rate = max(0.01, self.exploration_rate)
