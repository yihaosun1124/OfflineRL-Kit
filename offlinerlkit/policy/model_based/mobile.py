import numpy as np
import torch
import torch.nn as nn
import gym

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from copy import deepcopy
from collections import defaultdict
from offlinerlkit.policy import BasePolicy
from offlinerlkit.dynamics import BaseDynamics


class MOBILEPolicy(BasePolicy):
    """
    Model-Bellman Inconsistancy Penalized Offline Reinforcement Learning
    """

    def __init__(
        self,
        dynamics: BaseDynamics,
        actor: nn.Module,
        critics: nn.ModuleList,
        actor_optim: torch.optim.Optimizer,
        critics_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        penalty_coef: float = 1.0,
        num_samples: int = 10,
        deterministic_backup: bool = False
    ) -> None:

        super().__init__()
        self.dynamics = dynamics
        self.actor = actor
        self.critics = critics
        self.critics_old = deepcopy(critics)
        self.critics_old.eval()

        self.actor_optim = actor_optim
        self.critics_optim = critics_optim

        self._tau = tau
        self._gamma = gamma

        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self.alpha_optim = alpha
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha

        self._penalty_coef = penalty_coef
        self._num_samples = num_samples
        self._deteterministic_backup = deterministic_backup

    def train(self) -> None:
        self.actor.train()
        self.critics.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critics.eval()

    def _sync_weight(self) -> None:
        for o, n in zip(self.critics_old.parameters(), self.critics.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
    
    def actforward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.actor(obs)
        if deterministic:
            squashed_action, raw_action = dist.mode()
        else:
            squashed_action, raw_action = dist.rsample()
        log_prob = dist.log_prob(squashed_action, raw_action)
        return squashed_action, log_prob

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        with torch.no_grad():
            action, _ = self.actforward(obs, deterministic)
        return action.cpu().numpy()
    
    def rollout(
        self,
        init_obss: np.ndarray,
        rollout_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict]:

        num_transitions = 0
        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)

        # rollout
        observations = init_obss
        for _ in range(rollout_length):
            actions = self.select_action(observations)
            next_observations, rewards, terminals, info = self.dynamics.step(observations, actions)

            rollout_transitions["obss"].append(observations)
            rollout_transitions["next_obss"].append(next_observations)
            rollout_transitions["actions"].append(actions)
            rollout_transitions["rewards"].append(rewards)
            rollout_transitions["terminals"].append(terminals)

            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.flatten())

            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations[nonterm_mask]
        
        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.concatenate(v, axis=0)

        return rollout_transitions, \
            {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean()}
    
    @ torch.no_grad()
    def compute_lcb(self, obss: torch.Tensor, actions: torch.Tensor):
        # compute next q std
        pred_next_obss = self.dynamics.sample_next_obss(obss, actions, self._num_samples)
        num_samples, num_ensembles, batch_size, obs_dim = pred_next_obss.shape
        pred_next_obss = pred_next_obss.reshape(-1, obs_dim)
        pred_next_actions, _ = self.actforward(pred_next_obss)
        
        pred_next_qs =  torch.cat([critic_old(pred_next_obss, pred_next_actions) for critic_old in self.critics_old], 1)
        pred_next_qs = torch.min(pred_next_qs, 1)[0].reshape(num_samples, num_ensembles, batch_size, 1)
        penalty = pred_next_qs.mean(0).std(0)

        return penalty

    def learn(self, batch: Dict) -> Dict[str, float]:
        real_batch, fake_batch = batch["real"], batch["fake"]
        mix_batch = {k: torch.cat([real_batch[k], fake_batch[k]], 0) for k in real_batch.keys()}
        
        obss, actions, next_obss, rewards, terminals = mix_batch["observations"], mix_batch["actions"], mix_batch["next_observations"], mix_batch["rewards"], mix_batch["terminals"]

        # update critic
        qs = torch.stack([critic(obss, actions) for critic in self.critics], 0)
        with torch.no_grad():
            penalty = self.compute_lcb(obss, actions)
            penalty[:len(real_batch["rewards"])] = 0.0

            next_actions, next_log_probs = self.actforward(next_obss)
            next_qs = torch.cat([critic_old(next_obss, next_actions) for critic_old in self.critics_old], 1)
            next_q = torch.min(next_qs, 1)[0].reshape(-1, 1)
            if not self._deteterministic_backup:
                next_q -= self._alpha * next_log_probs
            target_q = (rewards - self._penalty_coef * penalty) + self._gamma * (1 - terminals) * next_q
            target_q = torch.clamp(target_q, 0, None)

        critic_loss = ((qs - target_q) ** 2).mean()
        self.critics_optim.zero_grad()
        critic_loss.backward()
        self.critics_optim.step()

        # update actor
        a, log_probs = self.actforward(obss)
        qas = torch.cat([critic(obss, a) for critic in self.critics], 1)
        actor_loss = -torch.min(qas, 1)[0].mean() + self._alpha * log_probs.mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = torch.clamp(self._log_alpha.detach().exp(), 0.0, 1.0)

        self._sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item()
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()

        return result