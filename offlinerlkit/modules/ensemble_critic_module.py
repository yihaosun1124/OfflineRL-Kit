import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union, Optional, List, Tuple

from offlinerlkit.nets import EnsembleLinear


class EnsembleCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        activation: nn.Module = nn.ReLU,
        num_ensemble: int = 10,
        device: str = "cpu"
    ) -> None:
        super().__init__()
        input_dim = obs_dim + action_dim
        hidden_dims = [input_dim] + list(hidden_dims)
        model = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            model += [EnsembleLinear(in_dim, out_dim, num_ensemble), activation()]
        model.append(EnsembleLinear(hidden_dims[-1], 1, num_ensemble))
        self.model = nn.Sequential(*model)

        self.device = torch.device(device)
        self.model = self.model.to(device)
        self._num_ensemble = num_ensemble

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        actions: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if actions is not None:
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
            obs = torch.cat([obs, actions], dim=-1)
        values = self.model(obs)
        # values: [num_ensemble, batch_size, 1]
        return values