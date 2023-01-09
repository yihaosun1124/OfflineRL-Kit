import numpy as np
import torch
import torch.nn as nn
import gym

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from collections import defaultdict
from offlinerlkit.policy import SACPolicy
from offlinerlkit.dynamics import BaseDynamics


class MOPEPolicy(SACPolicy):
    """
    Model-based Offline with Policy Embedding
    """

    def __init__(
        self,
        dynamics: BaseDynamics,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2
    ) -> None:
        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha
        )

        self.dynamics = dynamics

    def rollout(
        self,
        init_obss: np.ndarray,
        rollout_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict]:

        self.dynamics.model.eval()
        
        num_transitions = 0
        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)
        batch_size, obs_dim = init_obss.shape

        # rollout
        observations = init_obss
        obs_history = init_obss.reshape(len(init_obss), 1, -1)
        for _ in range(rollout_length):
            actions = self.select_action(obs_history.reshape(-1, obs_dim))
            act_history = actions.reshape(len(obs_history), -1, actions.shape[-1])
            next_observations, rewards, terminals, info = self.dynamics.step(obs_history, act_history)

            rollout_transitions["obss"].append(observations)
            rollout_transitions["next_obss"].append(next_observations)
            rollout_transitions["actions"].append(act_history[:, -1])
            rollout_transitions["rewards"].append(rewards)
            rollout_transitions["terminals"].append(terminals)

            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.flatten())

            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break
            
            obs_history = np.concatenate([obs_history, next_observations.reshape(len(next_observations), 1, -1)], axis=1)
            obs_history = obs_history[nonterm_mask].copy()
            observations = next_observations[nonterm_mask].copy()
        
        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.concatenate(v, axis=0)

        return rollout_transitions, \
            {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean()}