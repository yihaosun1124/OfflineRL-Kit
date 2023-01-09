import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union, Optional


# for SAC
class ActorProb(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        dist_net: nn.Module,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        self.dist_net = dist_net.to(device)

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.distributions.Normal:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.backbone(obs)
        dist = self.dist_net(logits)
        return dist


# for TD3
class Actor(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        action_dim: int,
        max_action: float = 1.0,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        latent_dim = getattr(backbone, "output_dim")
        output_dim = action_dim
        self.last = nn.Linear(latent_dim, output_dim).to(device)
        self._max = max_action

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.backbone(obs)
        actions = self._max * torch.tanh(self.last(logits))
        return actions