import numpy as np
import torch
import torch.nn as nn
import gym

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from collections import defaultdict
from offlinerlkit.policy import CQLPolicy
from offlinerlkit.dynamics import BaseDynamics


class COMBOPolicy(CQLPolicy):
    """
    Conservative Offline Model-Based Policy Optimization <Ref: https://arxiv.org/abs/2102.08363>
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
        action_space: gym.spaces.Space,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        cql_weight: float = 1.0,
        temperature: float = 1.0,
        max_q_backup: bool = False,
        deterministic_backup: bool = True,
        with_lagrange: bool = True,
        lagrange_threshold: float = 10.0,
        cql_alpha_lr: float = 1e-4,
        num_repeart_actions:int = 10,
        uniform_rollout: bool = False,
        rho_s: str = "mix"
    ) -> None:
        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            action_space,
            tau=tau,
            gamma=gamma,
            alpha=alpha,
            cql_weight=cql_weight,
            temperature=temperature,
            max_q_backup=max_q_backup,
            deterministic_backup=deterministic_backup,
            with_lagrange=with_lagrange,
            lagrange_threshold=lagrange_threshold,
            cql_alpha_lr=cql_alpha_lr,
            num_repeart_actions=num_repeart_actions
        )

        self.dynamics = dynamics
        self._uniform_rollout = uniform_rollout
        self._rho_s = rho_s

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
            if self._uniform_rollout:
                actions = np.random.uniform(
                    self.action_space.low[0],
                    self.action_space.high[0],
                    size=(len(observations), self.action_space.shape[0])
                )
            else:
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
    
    def learn(self, batch: Dict) -> Dict[str, float]:
        real_batch, fake_batch = batch["real"], batch["fake"]
        mix_batch = {k: torch.cat([real_batch[k], fake_batch[k]], 0) for k in real_batch.keys()}

        obss, actions, next_obss, rewards, terminals = mix_batch["observations"], mix_batch["actions"], \
            mix_batch["next_observations"], mix_batch["rewards"], mix_batch["terminals"]
        batch_size = obss.shape[0]
        
        # update actor
        a, log_probs = self.actforward(obss)
        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)
        actor_loss = (self._alpha * log_probs - torch.min(q1a, q2a)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()
        
        # compute td error
        if self._max_q_backup:
            with torch.no_grad():
                tmp_next_obss = next_obss.unsqueeze(1) \
                    .repeat(1, self._num_repeat_actions, 1) \
                    .view(batch_size * self._num_repeat_actions, next_obss.shape[-1])
                tmp_next_actions, _ = self.actforward(tmp_next_obss)
                tmp_next_q1 = self.critic1_old(tmp_next_obss, tmp_next_actions) \
                    .view(batch_size, self._num_repeat_actions, 1) \
                    .max(1)[0].view(-1, 1)
                tmp_next_q2 = self.critic2_old(tmp_next_obss, tmp_next_actions) \
                    .view(batch_size, self._num_repeat_actions, 1) \
                    .max(1)[0].view(-1, 1)
                next_q = torch.min(tmp_next_q1, tmp_next_q2)
        else:
            with torch.no_grad():
                next_actions, next_log_probs = self.actforward(next_obss)
                next_q = torch.min(
                    self.critic1_old(next_obss, next_actions),
                    self.critic2_old(next_obss, next_actions)
                )
                if not self._deterministic_backup:
                    next_q -= self._alpha * next_log_probs

        target_q = rewards + self._gamma * (1 - terminals) * next_q
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        critic2_loss = ((q2 - target_q).pow(2)).mean()

        # compute conservative loss
        if self._rho_s == "model":
            obss, actions, next_obss = fake_batch["observations"], \
                fake_batch["actions"], fake_batch["next_observations"]
            
        batch_size = len(obss)
        random_actions = torch.FloatTensor(
            batch_size * self._num_repeat_actions, actions.shape[-1]
        ).uniform_(self.action_space.low[0], self.action_space.high[0]).to(self.actor.device)
        # tmp_obss & tmp_next_obss: (batch_size * num_repeat, obs_dim)
        tmp_obss = obss.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, obss.shape[-1])
        tmp_next_obss = next_obss.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, obss.shape[-1])
        
        obs_pi_value1, obs_pi_value2 = self.calc_pi_values(tmp_obss, tmp_obss)
        next_obs_pi_value1, next_obs_pi_value2 = self.calc_pi_values(tmp_next_obss, tmp_obss)
        random_value1, random_value2 = self.calc_random_values(tmp_obss, random_actions)

        for value in [
            obs_pi_value1, obs_pi_value2, next_obs_pi_value1, next_obs_pi_value2,
            random_value1, random_value2
        ]:
            value.reshape(batch_size, self._num_repeat_actions, 1)
        
        # cat_q shape: (batch_size, 3 * num_repeat, 1)
        cat_q1 = torch.cat([obs_pi_value1, next_obs_pi_value1, random_value1], 1)
        cat_q2 = torch.cat([obs_pi_value2, next_obs_pi_value2, random_value2], 1)

        conservative_loss1 = \
            torch.logsumexp(cat_q1 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
            q1.mean() * self._cql_weight
        conservative_loss2 = \
            torch.logsumexp(cat_q2 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
            q2.mean() * self._cql_weight
        
        if self._with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), 0.0, 1e6)
            conservative_loss1 = cql_alpha * (conservative_loss1 - self._lagrange_threshold)
            conservative_loss2 = cql_alpha * (conservative_loss2 - self._lagrange_threshold)

            self.cql_alpha_optim.zero_grad()
            cql_alpha_loss = -(conservative_loss1 + conservative_loss2) * 0.5
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optim.step()
        
        critic1_loss = critic1_loss + conservative_loss1
        critic2_loss = critic2_loss + conservative_loss2

        # update critic
        self.critic1_optim.zero_grad()
        critic1_loss.backward(retain_graph=True)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        self._sync_weight()

        result =  {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item()
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()
        if self._with_lagrange:
            result["loss/cql_alpha"] = cql_alpha_loss.item()
            result["cql_alpha"] = cql_alpha.item()
        
        return result