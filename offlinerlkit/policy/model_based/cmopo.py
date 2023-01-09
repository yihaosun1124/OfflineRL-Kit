from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import gym

from torch.nn import functional as F
from typing import Dict, List, Union, Tuple
from collections import defaultdict
from offlinerlkit.policy import BasePolicy
from offlinerlkit.dynamics import BaseDynamics


class CMOPOPolicy(BasePolicy):
    """
    Model-based Offline Policy Optimization <Ref: https://arxiv.org/abs/2005.13239>
    """

    def __init__(
        self,
        dynamics: BaseDynamics,
        actor: nn.Module,
        critics: List[nn.Module],
        actor_optim: torch.optim.Optimizer,
        critic_optims: List[torch.optim.Optimizer],
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2
    ) -> None:
        
        super().__init__()
        self.dynamics = dynamics
        self.actor = actor
        self.critics = critics
        self.critics_old = [deepcopy(critic) for critic in critics]
        for critic_old in self.critics_old: critic_old.eval()

        self.actor_optim = actor_optim
        self.critic_optims = critic_optims

        self._tau = tau
        self._gamma = gamma

        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self.alpha_optim = alpha
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha

    def train(self) -> None:
        self.actor.train()
        for critic in self.critics:
            critic.train()

    def eval(self) -> None:
        self.actor.eval()
        for critic in self.critics:
            critic.eval()

    def _sync_weight(self) -> None:
        for i in range(len(self.critics)):
            for o, n in zip(self.critics_old[i].parameters(), self.critics[i].parameters()):
                o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

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
    
    def learn(self, batch: Dict) -> Dict[str, float]:
        real_batch, fake_batch = batch["real"], batch["fake"]
        real_obss, real_actions, real_next_obss, real_rewards, real_terminals = real_batch["observations"], real_batch["actions"], \
            real_batch["next_observations"], real_batch["rewards"], real_batch["terminals"]
        fake_obss, fake_actions, fake_next_obss, fake_rewards, fake_terminals = fake_batch["observations"], fake_batch["actions"], \
            fake_batch["next_observations"], fake_batch["rewards"], fake_batch["terminals"]
        mix_obss = torch.cat([real_obss, fake_obss], 0)

        # update critic
        real_qs = torch.stack([critic(real_obss, real_actions) for critic in self.critics], 0)
        fake_qs = torch.stack([critic(fake_obss, fake_actions) for critic in self.critics], 0)
        with torch.no_grad():
            real_next_actions, real_next_log_probs = self.actforward(real_next_obss)
            real_next_qs = torch.cat([critic_old(real_next_obss, real_next_actions) for critic_old in self.critics_old], 1)
            real_next_q = torch.max(real_next_qs, 1)[0].reshape(-1, 1) - self._alpha * real_next_log_probs
            # real_next_q = torch.mean(real_next_qs, 1).reshape(-1, 1) - self._alpha * real_next_log_probs
            real_target_q = real_rewards + self._gamma * (1 - real_terminals) * real_next_q

            fake_next_actions, fake_next_log_probs = self.actforward(fake_next_obss)
            fake_next_qs = torch.cat([critic_old(fake_next_obss, fake_next_actions) for critic_old in self.critics_old], 1)
            fake_next_q = torch.min(fake_next_qs, 1)[0].reshape(-1, 1) - self._alpha * fake_next_log_probs
            fake_target_q = fake_rewards + self._gamma * (1 - fake_terminals) * fake_next_q

        real_critic_loss = ((real_qs - real_target_q) ** 2).mean()
        fake_critic_loss = ((fake_qs - fake_target_q) ** 2).mean()
        critic_loss = (real_critic_loss + fake_critic_loss) / 2
        for critic_optim in self.critic_optims:
            critic_optim.zero_grad()
        critic_loss.backward()
        for critic_optim in self.critic_optims:
            critic_optim.step()

        # update actor
        a, log_probs = self.actforward(real_obss)
        qas = torch.cat([critic(real_obss, a) for critic in self.critics], 1)
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