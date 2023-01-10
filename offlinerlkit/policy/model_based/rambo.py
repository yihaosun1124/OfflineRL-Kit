import numpy as np
import torch
import torch.nn as nn
import gym
import os

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from collections import defaultdict
from operator import itemgetter
from offlinerlkit.policy import MOPOPolicy
from offlinerlkit.dynamics import BaseDynamics


class RAMBOPolicy(MOPOPolicy):
    """
    RAMBO-RL: Robust Adversarial Model-Based Offline Reinforcement Learning <Ref: https://arxiv.org/abs/2204.12581>
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
        dynamics_adv_optim: torch.optim.Optimizer, 
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2, 
        adv_weight: float=0, 
        adv_rollout_batch_size: int=256, 
        adv_rollout_length: int=5, 
        include_ent_in_adv: bool=False,   # CHECK 这里是不是False
        device="cpu"
    ) -> None:
        super().__init__(
            dynamics, 
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

        self._dynmics_adv_optim = dynamics_adv_optim
        self._adv_weight = adv_weight
        self._adv_rollout_batch_size = adv_rollout_batch_size
        self._adv_rollout_length = adv_rollout_length
        self._include_ent_in_adv = include_ent_in_adv
        self.device = device
        
    def load(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"))

    def pretrain(self, data: Dict, n_epoch, batch_size, lr, logger) -> None:
        self._bc_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        observations = data["observations"]
        actions = data["actions"]
        sample_num = observations.shape[0]
        idxs = np.arange(sample_num)

        logger.log("Pretraining policy")
        self.actor.train()
        for i_epoch in range(n_epoch):
            np.random.shuffle(idxs)
            sum_loss = 0
            for i_batch in range(sample_num // batch_size):
                batch_obs = observations[i_batch * batch_size: (i_batch + 1) * batch_size]
                batch_act = actions[i_batch * batch_size: (i_batch + 1) * batch_size]
                batch_obs = torch.from_numpy(batch_obs).to(self.device)
                batch_act = torch.from_numpy(batch_act).to(self.device)
                dist = self.actor(batch_obs)
                log_prob = dist.log_prob(batch_act)
                bc_loss = - log_prob.mean()

                self._bc_optim.zero_grad()
                bc_loss.backward()
                self._bc_optim.step()
                sum_loss += bc_loss.cpu().item()
            print(f"Epoch {i_epoch}, mean bc loss {sum_loss/i_batch}")
        torch.save(self.state_dict(), os.path.join(logger.model_dir, "rambo_pretrain.pt"))


    def update_dynamics(
        self, 
        real_buffer, 
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        all_loss_info = {
            "all_loss": 0, 
            "sl_loss": 0, 
            "adv_loss": 0
        }
        steps = 0
        while steps < 1000:
            init_obss = real_buffer.sample(self._adv_batch_size)["observations"].cpu().numpy()
            observations = init_obss
            for t in range(self._adv_rollout_length):
                actions = self.select_action(observations)
                sl_observations, sl_actions, sl_next_observations, sl_rewards = \
                    itemgetter("observations", "actions", "next_observations", "rewards")(real_buffer.sample(self._adv_batch_size))
                next_observations, terminals, loss_info = self.dynamics_step_and_forward(observations, actions, sl_observations, sl_actions, sl_next_observations, sl_rewards)
                all_loss_info["all_loss"] += loss_info["all_loss"]
                all_loss_info["adv_loss"] += loss_info["adv_loss"]
                all_loss_info["sl_loss"] += loss_info["sl_loss"]

                nonterm_mask = (~terminals).flatten()
                steps += 1
                observations = next_observations[nonterm_mask]
                if nonterm_mask.sum() == 0:
                    break
                if steps == 1000:
                    break
        return {_key: _value/steps for _key, _value in all_loss_info.items()}


    def dynamics_step_and_forward(
        self, 
        observations,  
        actions, 
        sl_observations, 
        sl_actions, 
        sl_next_observations, 
        sl_rewards, 
    ):
        obs_act = np.concatenate([observations, actions], axis=-1)
        obs_act = self.dynamics.scaler.transform(obs_act)
        with torch.no_grad():
            mean, logvar = self.dynamics.model(obs_act)
        # mean = mean.cpu().numpy()
        # logvar = logvar.cpu().numpy()
        observations = torch.from_numpy(observations).to(mean.device)
        mean[..., :-1] += observations
        std = torch.sqrt(torch.exp(logvar))
        _noise_generator = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(mean))
        noise = _noise_generator.sample()

        # select the next observations
        sample_size = mean.shape[1]
        selected_indexes = np.random.randint(0, noise.shape[0], size=sample_size)
        noise = noise[selected_indexes, np.arange(sample_size)]
        sample = mean + noise * std
        next_observations = sample[..., :-1][selected_indexes, np.arange(sample_size)]
        rewards = sample[..., -1][selected_indexes, np.arange(sample_size)]
        terminals = np.squeeze(self.dynamics.terminal_fn(observations.detach().cpu().numpy(), actions, next_observations.detach().cpu().numpy()))
        # terminals = torch.from_numpy(terminals).to(mean.device)
        # evaluate the noises
        log_prob = _noise_generator.log_prob(noise)
        log_prob = log_prob.exp().sum(dim=0).log().sum(-1)

        # compute the advantage
        with torch.no_grad():
            next_actions, next_policy_log_prob = self.actforward(next_observations, deterministic=True)
            next_q = torch.minimum(
                self.critic1(next_observations, next_actions), 
                self.critic2(next_observations, next_actions)
            )
            if self._include_ent_in_adv:
                next_q = next_q - self._alpha * next_policy_log_prob
            value = rewards.unsqueeze(1) + (1-torch.from_numpy(terminals).to(mean.device).float().unsqueeze(1)) * self._gamma * next_q

            q = torch.minimum(
                self.critic1(observations, actions), 
                self.critic2(observations, actions)
            )
            advantage = q - value
        adv_loss = (log_prob * advantage).mean()

        # compute the supervised loss
        sl_input = torch.cat([sl_observations, sl_actions], dim=-1).cpu().numpy()
        sl_target = torch.cat([sl_next_observations-sl_observations, sl_rewards], dim=-1)
        sl_input = self.dynamics.transform(sl_input)
        sl_mean, sl_logvar = self.dynamics.model(sl_input)
        sl_inv_var = torch.exp(-sl_logvar)
        sl_mse_loss_inv = (torch.pow(sl_mean - sl_target, 2) * sl_inv_var).mean(dim=(1, 2))
        sl_var_loss = sl_logvar.mean(dim=(1, 2))
        sl_loss = sl_mse_loss_inv.sum() + sl_var_loss.sum()
        sl_loss = sl_loss + self.dynamics.model.get_decay_loss()
        sl_loss = sl_loss + 0.01 * self.dynamics.model.max_logvar.sum() - 0.01 * self.dynamics.model.min_logvar.sum()

        all_loss = self._adv_weight * adv_loss + sl_loss
        self._dynmics_adv_optim.zero_grad()
        all_loss.backward()
        self._dynmics_adv_optim.step()

        return next_observations.cpu().numpy(), terminals, {
            "all_loss": all_loss.cpu().item(), 
            "sl_loss": sl_loss.cpu().item(), 
            "adv_loss": adv_loss.cpu().item()
        }