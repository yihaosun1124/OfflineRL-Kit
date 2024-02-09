import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from typing import Dict, Union, Tuple, Callable
from offlinerlkit.policy import BasePolicy


class BCPolicy(BasePolicy):

    def __init__(
        self,
        actor: nn.Module,
        actor_optim: torch.optim.Optimizer
    ) -> None:

        super().__init__()
        self.actor = actor
        self.actor_optim = actor_optim
    
    def train(self) -> None:
        self.actor.train()

    def eval(self) -> None:
        self.actor.eval()
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy()
        return action
    
    def learn(self, batch: Dict) -> Dict[str, float]:
        obss, actions = batch["observations"], batch["actions"]
        
        a = self.actor(obss)
        actor_loss = ((a - actions).pow(2)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        return {
            "loss/actor": actor_loss.item()
        }