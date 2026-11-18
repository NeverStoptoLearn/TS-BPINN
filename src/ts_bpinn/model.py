import math
import torch
from torch import nn
from torch.nn import functional as F


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.weight_log_sigma = nn.Parameter(torch.full((out_features, in_features), -5.0))
        self.bias_log_sigma = nn.Parameter(torch.full((out_features,), -5.0))
        nn.init.xavier_uniform_(self.weight_mu)

    def forward(self, x):
        weight = self.weight_mu + self.weight_log_sigma.exp() * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + self.bias_log_sigma.exp() * torch.randn_like(self.bias_mu)
        return F.linear(x, weight, bias)

    def kl_divergence(self):
        # q = N(mu, sigma^2); standard normal prior.
        return sum(0.5 * (mu.square() + (2 * logsigma).exp() - 1 - 2 * logsigma).sum()
                   for mu, logsigma in [(self.weight_mu, self.weight_log_sigma), (self.bias_mu, self.bias_log_sigma)])


class BayesianPINN(nn.Module):
    def __init__(self, length=10.0, hidden=(256, 128, 256), initial_position=3.0, initial_amplitude=0.8):
        super().__init__()
        if not hidden or min(hidden) < 1 or not 0 < initial_position < length or initial_amplitude <= 0:
            raise ValueError('Invalid network or source initialization')
        widths = [2, *hidden, 1]
        self.layers = nn.ModuleList([BayesianLinear(a, b) for a, b in zip(widths[:-1], widths[1:])])
        self.length = float(length)
        self.position_logit = nn.Parameter(torch.tensor(math.log(initial_position / (length - initial_position))))
        self.amplitude_raw = nn.Parameter(torch.tensor(math.log(math.expm1(initial_amplitude))))

    @property
    def position(self):
        return self.length * self.position_logit.sigmoid()

    @property
    def amplitude(self):
        return F.softplus(self.amplitude_raw)

    def forward(self, x, t):
        value = torch.cat([x, t], dim=1)
        for layer in self.layers[:-1]:
            value = F.silu(layer(value))
        return self.layers[-1](value)

    def mean_kl(self):
        count = sum(layer.weight_mu.numel() + layer.bias_mu.numel() for layer in self.layers)
        return sum(layer.kl_divergence() for layer in self.layers) / count
