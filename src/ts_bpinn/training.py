import torch
from .losses import loss_terms, weighted_loss


def train(model, data, physics, stages, learning_rate=1e-3, kl_weight=1e-4):
    if learning_rate <= 0 or not stages:
        raise ValueError('Positive learning rate and at least one stage are required')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    model.train()
    for stage_index, stage in enumerate(stages, 1):
        if stage['epochs'] < 1:
            raise ValueError('Each stage needs at least one epoch')
        for epoch in range(stage['epochs']):
            optimizer.zero_grad(set_to_none=True)
            terms = loss_terms(model, data, physics)
            loss = weighted_loss(terms, stage['weights'], kl_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            history.append({'stage': stage_index, 'epoch': epoch + 1, 'loss': float(loss.detach()),
                            **{k: float(v.detach()) for k, v in terms.items()},
                            'position': float(model.position.detach()), 'amplitude': float(model.amplitude.detach())})
    return history


@torch.no_grad()
def predict(model, x, t, samples=100):
    if samples < 2:
        raise ValueError('At least two stochastic samples are required')
    model.eval()
    # Save empirical marginal quantiles rather than assuming Gaussian coverage.
    values = torch.stack([model(x, t) for _ in range(samples)])
    return {'mean': values.mean(0), 'std': values.std(0),
            'lower': torch.quantile(values, 0.025, dim=0), 'upper': torch.quantile(values, 0.975, dim=0)}
