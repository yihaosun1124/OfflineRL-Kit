import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional


class EnsembleLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_ensemble: int,
        num_elites: int,
        weight_decay: float = 0.0,
        load_model: bool = False
    ) -> None:
        super().__init__()

        self.num_ensemble = num_ensemble
        self.num_elites = num_elites

        self.register_parameter("weight", nn.Parameter(torch.zeros(num_ensemble, input_dim, output_dim)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(num_ensemble, 1, output_dim)))

        nn.init.trunc_normal_(self.weight, std=1/(2*input_dim**0.5))

        self.register_parameter("saved_weight", nn.Parameter(self.weight.detach().clone()))
        self.register_parameter("saved_bias", nn.Parameter(self.bias.detach().clone()))

        if not load_model:
            self.register_parameter("elites", nn.Parameter(torch.tensor(list(range(0, self.num_ensemble))), requires_grad=False))
        else:
            self.register_parameter("elites", nn.Parameter(torch.tensor(list(range(0, self.num_elites))), requires_grad=False))

        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight[self.elites]
        bias = self.bias[self.elites]

        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, weight)
        else:
            x = torch.einsum('bij,bjk->bik', x, weight)

        x = x + bias

        return x

    def set_elites(self, indexes: List[int]) -> None:
        assert len(indexes) <= self.num_ensemble and max(indexes) < self.num_ensemble
        self.register_parameter('elites', nn.Parameter(torch.tensor(indexes), requires_grad=False))
        self.weight.data.copy_(self.saved_weight.data)
        self.bias.data.copy_(self.saved_bias.data)

    def update_save(self, indexes: List[int]) -> None:
        self.saved_weight.data[indexes] = self.weight.data[indexes]
        self.saved_bias.data[indexes] = self.bias.data[indexes]
    
    def reset_elites(self) -> None:
        self.register_parameter('elites', nn.Parameter(torch.tensor(list(range(0, self.num_ensemble))), requires_grad=False))
    
    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = self.weight_decay * (0.5*((self.weight**2).sum()))
        return decay_loss