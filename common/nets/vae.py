import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional


# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        latent_dim: int,
        max_action: Union[int, float],
        device: str = "cpu"
    ) -> None:
        super(VAE, self).__init__()
        self.e1 = nn.Linear(input_dim + output_dim, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)

        self.d1 = nn.Linear(input_dim + latent_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Linear(hidden_dim, output_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = torch.device(device)

        self.to(device=self.device)


    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = F.relu(self.e1(torch.cat([obs, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability 
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(obs, z)

        return u, mean, std

    def decode(self, obs: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((obs.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)

        a = F.relu(self.d1(torch.cat([obs, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))