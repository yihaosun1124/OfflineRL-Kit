import numpy as np
import torch
import torch.nn as nn


class NormalWrapper(torch.distributions.Normal):
    def log_prob(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


class TanhNormalWrapper(torch.distributions.Normal):
    def log_prob(self, action, raw_action=None):
        if raw_action is None:
            raw_action = self.arctanh(action)
        log_prob = super().log_prob(raw_action).sum(-1, keepdim=True)
        eps = 1e-6
        log_prob = log_prob - torch.log((1 - action.pow(2)) + eps).sum(-1, keepdim=True)
        return log_prob

    def mode(self):
        raw_action = self.mean
        action = torch.tanh(self.mean)
        return action, raw_action

    def arctanh(self, x):
        one_plus_x = (1 + x).clamp(min=1e-6)
        one_minus_x = (1 - x).clamp(min=1e-6)
        return 0.5 * torch.log(one_plus_x / one_minus_x)

    def rsample(self):
        raw_action = super().rsample()
        action = torch.tanh(raw_action)
        return action, raw_action


class DiagGaussian(nn.Module):
    def __init__(
        self,
        latent_dim,
        output_dim,
        unbounded=False,
        conditioned_sigma=False,
        max_mu=1.0,
        sigma_min=-5.0,
        sigma_max=2.0
    ):
        super().__init__()
        self.mu = nn.Linear(latent_dim, output_dim)
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = nn.Linear(latent_dim, output_dim)
        else:
            self.sigma_param = nn.Parameter(torch.zeros(output_dim, 1))
        self._unbounded = unbounded
        self._max = max_mu
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max

    def forward(self, logits):
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=self._sigma_min, max=self._sigma_max).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        return NormalWrapper(mu, sigma)


class TanhDiagGaussian(DiagGaussian):
    def __init__(
        self,
        latent_dim,
        output_dim,
        unbounded=False,
        conditioned_sigma=False,
        max_mu=1.0,
        sigma_min=-5.0,
        sigma_max=2.0
    ):
        super().__init__(
            latent_dim=latent_dim,
            output_dim=output_dim,
            unbounded=unbounded,
            conditioned_sigma=conditioned_sigma,
            max_mu=max_mu,
            sigma_min=sigma_min,
            sigma_max=sigma_max
        )

    def forward(self, logits):
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=self._sigma_min, max=self._sigma_max).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        return TanhNormalWrapper(mu, sigma)