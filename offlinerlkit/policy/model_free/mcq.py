import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from offlinerlkit.policy import SACPolicy


class MCQPolicy(SACPolicy):
    """
    Mildly Conservative Q-Learning <Ref: https://arxiv.org/abs/2206.04745>
    """

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        behavior_policy: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        behavior_policy_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        lmbda: float = 0.7,
        num_sampled_actions: int = 10
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

        self.behavior_policy = behavior_policy
        self.behavior_policy_optim = behavior_policy_optim
        self._lmbda = lmbda
        self._num_sampled_actions = num_sampled_actions

    def learn(self, batch: Dict) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        
        # update behavior policy
        recon, mean, std = self.behavior_policy(obss, actions)
        recon_loss = F.mse_loss(recon, actions)
        KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + KL_loss

        self.behavior_policy_optim.zero_grad()
        vae_loss.backward()
        self.behavior_policy_optim.step()

        # update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actforward(next_obss)
            next_q = torch.min(
                self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions)
            ) - self._alpha * next_log_probs
            target_q_for_in_actions = rewards + self._gamma * (1 - terminals) * next_q
        q1_in, q2_in = self.critic1(obss, actions), self.critic2(obss, actions)
        critic1_loss_for_in_actions = ((q1_in - target_q_for_in_actions).pow(2)).mean()
        critic2_loss_for_in_actions = ((q2_in - target_q_for_in_actions).pow(2)).mean()

        s_in = torch.cat([obss, next_obss], dim=0)
        with torch.no_grad():
            s_in_repeat = torch.repeat_interleave(s_in, self._num_sampled_actions, 0)
            sampled_actions = self.behavior_policy.decode(s_in_repeat)
            target_q1_for_ood_actions = self.critic1_old(s_in_repeat, sampled_actions).reshape(s_in.shape[0], -1).max(1)[0].reshape(-1, 1)
            target_q2_for_ood_actions = self.critic2_old(s_in_repeat, sampled_actions).reshape(s_in.shape[0], -1).max(1)[0].reshape(-1, 1)
            target_q_for_ood_actions = torch.min(target_q1_for_ood_actions, target_q2_for_ood_actions)
            ood_actions, _ = self.actforward(s_in)
        
        q1_ood, q2_ood = self.critic1(s_in, ood_actions), self.critic2(s_in, ood_actions)
        critic1_loss_for_ood_actions = ((q1_ood - target_q_for_ood_actions).pow(2)).mean()
        critic2_loss_for_ood_actions = ((q2_ood - target_q_for_ood_actions).pow(2)).mean()

        critic1_loss = self._lmbda * critic1_loss_for_in_actions + (1 - self._lmbda) * critic1_loss_for_ood_actions
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        critic2_loss = self._lmbda * critic2_loss_for_in_actions + (1 - self._lmbda) * critic2_loss_for_ood_actions
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
            "loss/behavior_policy": vae_loss.item()
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()

        return result

