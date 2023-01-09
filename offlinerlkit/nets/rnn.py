import torch
import torch.nn as nn
from torch.nn import functional as F


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


def soft_clamp(x : torch.Tensor, _min=None, _max=None):
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


class ResBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        activation=Swish(),
        layer_norm=True,
        with_residual=True,
        dropout=0.1
    ):
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation
        self.layer_norm = nn.LayerNorm(output_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.with_residual = with_residual
    
    def forward(self, x):
        y = self.activation(self.linear(x))
        if self.dropout is not None:
            y = self.dropout(y)
        if self.with_residual:
            y = x + y
        if self.layer_norm is not None:
            y = self.layer_norm(y)
        return y


class RNNModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=[200, 200, 200, 200],
        rnn_num_layers=3,
        dropout_rate=0.1,
        device="cpu"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.device = torch.device(device)

        self.activation = Swish()
        self.rnn_layer = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dims[0],
            num_layers=rnn_num_layers,
            batch_first=True
        )
        module_list = []
        self.input_layer = ResBlock(input_dim, hidden_dims[0], dropout=dropout_rate, with_residual=False)
        dims = list(hidden_dims)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            module_list.append(ResBlock(in_dim, out_dim, dropout=dropout_rate))
        self.backbones = nn.ModuleList(module_list)
        self.merge_layer = nn.Linear(dims[0] + dims[-1], hidden_dims[0])
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        self.to(self.device)

    def forward(self, input, h_state=None):
        batch_size, num_timesteps, _ = input.shape
        input = torch.as_tensor(input, dtype=torch.float32).to(self.device)
        rnn_output, h_state = self.rnn_layer(input, h_state)
        rnn_output = rnn_output.reshape(-1, self.hidden_dims[0])
        input = input.view(-1, self.input_dim)
        output = self.input_layer(input)
        output = torch.cat([output, rnn_output], dim=-1)
        output = self.activation(self.merge_layer(output))
        for layer in self.backbones:
            output = layer(output)
        output = self.output_layer(output)
        output = output.view(batch_size, num_timesteps, -1)
        return output, h_state


if __name__ == "__main__":
    model = RNNModel(14, 12)
    x = torch.randn(64, 20, 14)
    y, _ = model(x)
    print(y.shape)