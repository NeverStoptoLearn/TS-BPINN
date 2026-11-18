import torch
from .data import analytical_solution


def loss_terms(model, data, physics):
    x = data['x'].detach().clone().requires_grad_(True)
    t = data['t'].detach().clone().requires_grad_(True)
    prediction = model(x, t)
    ct, cx = torch.autograd.grad(prediction, (t, x), torch.ones_like(prediction), create_graph=True)
    cxx = torch.autograd.grad(cx, x, torch.ones_like(cx), create_graph=True)[0]
    residual = ct + physics.velocity * cx - physics.dispersion * cxx
    initial = model.amplitude * torch.exp(-0.5 * ((data['x_ic'] - model.position) / physics.width)**2)
    boundary = analytical_solution(data['x_bc'], data['t_bc'], physics.true_position, physics.true_amplitude, physics)
    return {'pde': residual.square().mean(),
            'initial': (model(data['x_ic'], data['t_ic']) - initial).square().mean(),
            'boundary': (model(data['x_bc'], data['t_bc']) - boundary).square().mean(),
            'observation': (model(data['x_obs'], data['t_obs']) - data['C_obs']).square().mean(),
            'kl': model.mean_kl()}


def weighted_loss(terms, weights, kl_weight):
    if len(weights) != 4 or min(weights) < 0 or kl_weight < 0:
        raise ValueError('Four nonnegative loss weights and a nonnegative KL weight are required')
    keys = ('pde', 'initial', 'boundary', 'observation')
    return sum(w * terms[k] for k, w in zip(keys, weights)) + kl_weight * terms['kl']
