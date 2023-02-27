import numpy as np
import torch
import torch.nn as nn

from typing import Dict, Union, Tuple
from copy import deepcopy
from offlinerlkit.policy import BasePolicy


class EDACPolicy(BasePolicy):
    """
    Ensemble-Diversified Actor Critic <Ref: https://arxiv.org/abs/2110.01548>
    """

    def __init__(
        self,
        actor: nn.Module,
        critics: nn.ModuleList,
        actor_optim: torch.optim.Optimizer,
        critics_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        max_q_backup: bool = False,
        deterministic_backup: bool = True,
        eta: float = 1.0
    ) -> None:

        super().__init__()
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

        self._max_q_backup = max_q_backup
        self._deterministic_backup = deterministic_backup
        self._eta = eta
        self._num_critics = self.critics._num_ensemble

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

    def learn(self, batch: Dict) -> Dict:
        obss, actions, next_obss, rewards, terminals = \
            batch["observations"], batch["actions"], batch["next_observations"], batch["rewards"], batch["terminals"]
        
        if self._eta > 0:
            actions.requires_grad_(True)

        # update actor
        a, log_probs = self.actforward(obss)
        # qas: [num_critics, batch_size, 1]
        qas = self.critics(obss, a)
        actor_loss = -torch.min(qas, 0)[0].mean() + self._alpha * log_probs.mean()
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

        # update critic
        if self._max_q_backup:
            with torch.no_grad():
                batch_size = obss.shape[0]
                tmp_next_obss = next_obss.unsqueeze(1).repeat(1, 10, 1) \
                    .view(batch_size * 10, next_obss.shape[-1])
                tmp_next_actions, _ = self.actforward(tmp_next_obss)
                tmp_next_qs = self.critics_old(tmp_next_obss, tmp_next_actions) \
                    .view(self._num_critics, batch_size, 10, 1).max(2)[0] \
                    .view(self._num_critics, batch_size, 1)
                next_q = tmp_next_qs.min(0)[0]
        else:
            with torch.no_grad():
                next_actions, next_log_probs = self.actforward(next_obss)
                next_q = self.critics_old(next_obss, next_actions).min(0)[0]
                if not self._deterministic_backup:
                    next_q -= self._alpha * next_log_probs

        # target_q: [batch_size, 1]
        target_q = rewards + self._gamma * (1 - terminals) * next_q
        # qs: [num_critics, batch_size, 1]
        qs = self.critics(obss, actions)
        critics_loss = ((qs - target_q.unsqueeze(0)).pow(2)).mean(dim=(1, 2)).sum()

        if self._eta > 0:
            obss_tile = obss.unsqueeze(0).repeat(self._num_critics, 1, 1)
            actions_tile = actions.unsqueeze(0).repeat(self._num_critics, 1, 1).requires_grad_(True)
            qs_preds_tile = self.critics(obss_tile, actions_tile)
            qs_pred_grads, = torch.autograd.grad(qs_preds_tile.sum(), actions_tile, retain_graph=True, create_graph=True)
            qs_pred_grads = qs_pred_grads / (torch.norm(qs_pred_grads, p=2, dim=2).unsqueeze(-1) + 1e-10)
            qs_pred_grads = qs_pred_grads.transpose(0, 1)
            
            qs_pred_grads = torch.einsum('bik,bjk->bij', qs_pred_grads, qs_pred_grads)
            masks = torch.eye(self._num_critics, device=obss.device).unsqueeze(dim=0).repeat(qs_pred_grads.size(0), 1, 1)
            qs_pred_grads = (1 - masks) * qs_pred_grads
            grad_loss = torch.mean(torch.sum(qs_pred_grads, dim=(1, 2))) / (self._num_critics - 1)

            critics_loss += self._eta * grad_loss

        self.critics_optim.zero_grad()
        critics_loss.backward()
        self.critics_optim.step()

        self._sync_weight()

        result =  {
            "loss/actor": actor_loss.item(),
            "loss/critics": critics_loss.item()
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()
        
        return result



