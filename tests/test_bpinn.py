import numpy as np
import pytest
import torch
from ts_bpinn.data import Physics, analytical_solution, make_data
from ts_bpinn.model import BayesianPINN
from ts_bpinn.losses import loss_terms, weighted_loss
from ts_bpinn.training import train, predict


def test_reference_matches_initial_condition():
    p = Physics()
    x = torch.linspace(0, p.length, 20)[:, None]
    actual = analytical_solution(x, torch.zeros_like(x), p.true_position, p.true_amplitude, p)
    expected = p.true_amplitude * torch.exp(-0.5 * ((x - p.true_position) / p.width)**2)
    torch.testing.assert_close(actual, expected)


def test_reference_satisfies_ade():
    p = Physics()
    x = torch.linspace(1, 9, 20, dtype=torch.float64)[:, None].requires_grad_()
    t = torch.linspace(0.1, 1, 20, dtype=torch.float64)[:, None].requires_grad_()
    c = analytical_solution(x, t, p.true_position, p.true_amplitude, p)
    ct, cx = torch.autograd.grad(c, (t, x), torch.ones_like(c), create_graph=True)
    cxx = torch.autograd.grad(cx, x, torch.ones_like(cx), create_graph=True)[0]
    assert (ct + p.velocity * cx - p.dispersion * cxx).abs().max() < 1e-10


def test_observations_follow_transport_in_time():
    p = Physics(noise_std=0)
    data = make_data(p, nx=8, nt=5)
    expected = analytical_solution(data['x_obs'], data['t_obs'], p.true_position, p.true_amplitude, p)
    torch.testing.assert_close(data['C_obs'], expected)
    values = data['C_obs'].reshape(3, -1)
    assert not torch.equal(values[:, 0], values[:, -1])


def test_source_bounds_losses_and_gradient():
    p = Physics()
    model = BayesianPINN(hidden=[8, 8])
    terms = loss_terms(model, make_data(p, nx=8, nt=5), p)
    loss = weighted_loss(terms, [0.2, 1, 1, 1], 1e-4)
    loss.backward()
    assert 0 < model.position < p.length and model.amplitude > 0
    assert torch.isfinite(loss)
    assert torch.isfinite(model.position_logit.grad)
    assert torch.isfinite(model.amplitude_raw.grad)
    assert model.layers[0].weight_log_sigma.grad is not None
    assert terms['kl'] >= 0


def test_two_stages_train_and_checkpoint_reload(tmp_path):
    torch.set_num_threads(2)
    p = Physics()
    model = BayesianPINN(hidden=[8, 8])
    data = make_data(p, nx=8, nt=5)
    initial_position = model.position.detach().clone()
    history = train(model, data, p, [{'epochs': 2, 'weights': [0.2, 1, 1, 0]}, {'epochs': 2, 'weights': [0.2, 1, 1, 1]}])
    assert len(history) == 4 and history[-1]['stage'] == 2
    assert not torch.equal(initial_position, model.position.detach())
    torch.save(model.state_dict(), tmp_path / 'model.pt')
    restored = BayesianPINN(hidden=[8, 8])
    restored.load_state_dict(torch.load(tmp_path / 'model.pt', weights_only=True))
    for original, loaded in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(original, loaded)
    pred = predict(restored, data['x'], data['t'], samples=5)
    assert pred['mean'].shape == data['x'].shape
    assert torch.isfinite(pred['mean']).all()
    assert torch.all(pred['lower'] <= pred['upper'])
    assert pred['std'].max() > 0


def test_invalid_data_is_rejected():
    with pytest.raises(ValueError):
        Physics(noise_std=-1)
    with pytest.raises(ValueError):
        make_data(Physics(), nx=1)
