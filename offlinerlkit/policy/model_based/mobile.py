import numpy as np
import torch
import torch.nn as nn
import gym

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from collections import defaultdict
from offlinerlkit.policy import SACPolicy
from offlinerlkit.dynamics import BaseDynamics


class MOBILEPolicy(SACPolicy):
    """
    Model-Bellman Inconsistancy Penalized Offline Reinforcement Learning
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
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        penalty_coef: float = 1.0,
        num_samples: int = 10,
        deterministic_backup=False
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
        self._penalty_coef = penalty_coef
        self._num_samples = num_samples
        self._deterministic_backup = deterministic_backup

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
        pred_next_obss = self.dynamics.predict_next_obs(obss, actions, self._num_samples)
        num_samples, num_ensembles, batch_size, obs_dim = pred_next_obss.shape
        pred_next_obss = pred_next_obss.reshape(-1, obs_dim)
        pred_next_actions, _ = self.actforward(pred_next_obss)

        pred_next_qs = torch.min(self.critic1_old(pred_next_obss, pred_next_actions), self.critic2_old(pred_next_obss, pred_next_actions))\
            .reshape(num_samples, num_ensembles, batch_size, 1)
        penalty = pred_next_qs.mean(0).std(0)

        return penalty
    
    @ torch.no_grad()
    def compute_bellman_error(self, real_next_obss: torch.Tensor, pred_next_obss: torch.Tensor):
        real_next_actions, _ = self.actforward(real_next_obss)
        pred_next_actions, _ = self.actforward(pred_next_obss)
        real_next_qs = torch.min(self.critic1_old(real_next_obss, real_next_actions), self.critic2_old(real_next_obss, real_next_actions))
        pred_next_qs = torch.min(self.critic1_old(pred_next_obss, pred_next_actions), self.critic2_old(pred_next_obss, pred_next_actions))
        return torch.abs(real_next_qs - pred_next_qs)

    @ torch.no_grad()
    def compute_q_value(self, obss: torch.Tensor, actions: torch.Tensor):
        q_values = torch.min(self.critic1_old(obss, actions), self.critic2_old(obss, actions))
        return q_values.mean().cpu().numpy()

    def learn(self, batch: Dict) -> Dict[str, float]:
        real_batch, fake_batch = batch["real"], batch["fake"]
        mix_batch = {k: torch.cat([real_batch[k], fake_batch[k]], 0) for k in real_batch.keys()}
        
        obss, actions, next_obss, rewards, terminals = mix_batch["observations"], mix_batch["actions"], mix_batch["next_observations"], mix_batch["rewards"], mix_batch["terminals"]

        # update critic
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        with torch.no_grad():
            penalty = self.compute_lcb(obss, actions)
            penalty[:len(real_batch["rewards"])] = 0.0

            next_actions, next_log_probs = self.actforward(next_obss)
            next_q = torch.min(
                self.critic1_old(next_obss, next_actions),
                self.critic2_old(next_obss, next_actions)
            )
            if not self._deterministic_backup:
                next_q -= self._alpha * next_log_probs
            target_q = (rewards - self._penalty_coef * penalty) + self._gamma * (1 - terminals) * next_q
            target_q = torch.clamp(target_q, 0, None)

        critic1_loss = ((q1 - target_q).pow(2)).mean()
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        critic2_loss = ((q2 - target_q).pow(2)).mean()
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # update actor
        a, log_probs = self.actforward(obss)
        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)

        actor_loss = - torch.min(q1a, q2a).mean() + self._alpha * log_probs.mean()
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
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()

        return result