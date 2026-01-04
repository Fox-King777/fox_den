"""Module for the policy gradient algorithm."""

import gymnasium as gym
import numpy as np
import torch
from torch import nn

DEVICE = (
    'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
)


class PolicyNetwork(nn.Module):
    """Policy network for the inverted pendulum environment."""

    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass of the policy network.

        Args:
            state (torch.Tensor): state of the environment

        Returns:
            torch.Tensor: action probabilities
        """
        return self.layers(state)


def choose_action(state: np.ndarray, policy_network: PolicyNetwork) -> tuple[int, torch.Tensor]:
    """Choose an action based on the state.

    Args:
        state (np.ndarray): state of the environment
        policy_network (PolicyNetwork): policy network

    Returns:
        tuple[int, torch.Tensor]: action and action probabilities
    """
    state = torch.from_numpy(state).float().to(DEVICE)
    action_prob = policy_network(state)
    action = torch.multinomial(action_prob, 1)
    return action.item(), action_prob


def calculate_discounted_rewards(
    rewards: list[int], discount_factor: float, normalize=True
) -> torch.Tensor:
    """Calculate the discounted rewards.

    Args:
        rewards (list[int]): list of rewards
        discount_factor (float): discount factor
        normalize (bool, optional): normalize the discounted rewards. Defaults to True.

    Returns:
        torch.Tensor: discounted rewards
    """
    discounted_rewards = []
    discounted_reward = 0
    for reward in reversed(rewards):
        discounted_reward = reward + discount_factor * discounted_reward
        discounted_rewards.insert(0, discounted_reward)

    discounted_rewards = torch.tensor(discounted_rewards).to(DEVICE)
    if normalize:
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + 1e-8
        )

    return discounted_rewards


def update_policy(
    discounted_rewards: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    log_prob_actions: list[torch.Tensor],
) -> float:
    """Update the policy network based on one episode.

    Args:
        discounted_rewards (torch.Tensor): discounted rewards
        optimizer (torch.optim.Optimizer): optimizer
        log_prob_actions (list[torch.Tensor]): log probabilities of the actions

    Returns:
        float: loss
    """
    discounted_rewards = discounted_rewards.detach()
    loss = -(discounted_rewards * log_prob_actions).sum()
    optimizer.zero_grad()
    # calculated the gradient of the loss w.r.t. the policy network parameters.
    # stored in .grad attributes of the parameters
    loss.backward()
    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=1.0)
    optimizer.step()

    return loss.item()


def train_one_episode(
    env: gym.Env,
    policy: PolicyNetwork,
    optimizer: torch.optim.Optimizer,
    discount_factor: float = 0.99,
) -> tuple[int, float]:
    """Train the policy network for one episode.

    Args:
        env (gym.Env): gym environment
        policy (PolicyNetwork): policy network
        optimizer (torch.optim.Optimizer): optimizer
        discount_factor (float, optional): discount_factor. Defaults to 0.99.

    Returns:
        tuple[int, float]: episode reward and loss
    """
    policy.train()
    log_prob_actions = []
    rewards = []
    episode_reward = 0
    state = env.reset()[0]
    terminated = False
    truncated = False
    while not terminated and not truncated:
        action, action_prob = choose_action(state, policy)
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state

        log_prob_actions.append(torch.log(action_prob[action]))
        rewards.append(reward)
        episode_reward += reward

    log_prob_actions = torch.stack(log_prob_actions)

    discounted_rewards = calculate_discounted_rewards(rewards, discount_factor)

    loss = update_policy(discounted_rewards, optimizer, log_prob_actions)

    return episode_reward, loss


def evaluate_policy(env: gym.Env, policy: PolicyNetwork) -> float:
    """Evaluate the policy network.

    Args:
        env (gym.Envs): gym environment
        policy (PolicyNetwork): policy network

    Returns:
        float: episode reward
    """
    state = env.reset()[0]
    episode_reward = 0
    terminated = False
    while not terminated:
        with torch.no_grad():
            action, _ = choose_action(state, policy)
            state, reward, terminated, _, _ = env.step(action)

        episode_reward += reward

    return episode_reward
