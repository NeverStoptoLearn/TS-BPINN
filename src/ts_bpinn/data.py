"""A consistent Gaussian initial condition and advection-dispersion reference."""
from dataclasses import dataclass, asdict
import numpy as np
import torch


@dataclass
class Physics:
    length: float = 10.0
    duration: float = 1.0
    velocity: float = 1.0
    dispersion: float = 0.1
    width: float = 1.0
    true_position: float = 4.0
    true_amplitude: float = 1.0
    noise_std: float = 0.1

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Physics values must be finite')
        if min(self.length, self.duration, self.dispersion, self.width, self.true_amplitude) <= 0:
            raise ValueError('Domain, dispersion, width and amplitude must be positive')
        if not 0 < self.true_position < self.length or self.noise_std < 0:
            raise ValueError('Invalid source position or noise standard deviation')


def analytical_solution(x, t, position, amplitude, physics):
    """Infinite-domain Gaussian ADE solution; amplitude is the initial peak."""
    variance = physics.width**2 + 2 * physics.dispersion * t
    return amplitude * physics.width / torch.sqrt(variance) * torch.exp(
        -(x - position - physics.velocity * t)**2 / (2 * variance))


def make_data(physics, nx=100, nt=100, observation_positions=(3, 5, 8), observation_count=6, device='cpu'):
    if nx < 3 or nt < 2 or observation_count < 2:
        raise ValueError('Grid requires nx >= 3, nt >= 2, observation_count >= 2')
    if not observation_positions or any(not 0 <= p <= physics.length for p in observation_positions):
        raise ValueError('Observation positions must lie in the domain')
    x_axis = torch.linspace(0, physics.length, nx, device=device)
    t_axis = torch.linspace(0, physics.duration, nt, device=device)
    xx, tt = torch.meshgrid(x_axis, t_axis, indexing='ij')
    xo, to = torch.meshgrid(torch.tensor(observation_positions, dtype=torch.float32, device=device),
                           torch.linspace(0, physics.duration, observation_count, device=device), indexing='ij')
    reference = lambda x, t: analytical_solution(x, t, physics.true_position, physics.true_amplitude, physics)
    obs = reference(xo, to) + physics.noise_std * torch.randn_like(xo)
    return {'x': xx.reshape(-1, 1), 't': tt.reshape(-1, 1),
            'x_ic': x_axis[:, None], 't_ic': torch.zeros(nx, 1, device=device),
            'x_bc': torch.cat([torch.zeros(nt, 1, device=device), torch.full((nt, 1), physics.length, device=device)]),
            't_bc': t_axis.repeat(2)[:, None],
            'x_obs': xo.reshape(-1, 1), 't_obs': to.reshape(-1, 1), 'C_obs': obs.reshape(-1, 1),
            'shape': (nx, nt)}
