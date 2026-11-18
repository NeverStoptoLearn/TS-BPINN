from dataclasses import asdict
import argparse
import csv
import json
import numpy as np
import torch
from .data import Physics, make_data, analytical_solution
from .model import BayesianPINN
from .training import train, predict
from .experiment_io import start_run, save_json


def run(config, output=None):
    threads = int(config.get('threads', 2))
    if threads < 1:
        raise ValueError('threads must be positive')
    torch.set_num_threads(threads)
    physics = Physics(**config.get('physics', {}))
    device = config.get('device', 'cpu')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    model_config = config.get('model', {})
    model = BayesianPINN(length=physics.length, **model_config).to(device)
    data = make_data(physics, device=device, **config.get('data', {}))
    path = start_run(config, output)
    history = train(model, data, physics, **config['training'])
    with (path / 'training.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    torch.save({'state_dict': model.state_dict(), 'model_config': model_config, 'physics': asdict(physics)}, path / 'model.pt')
    prediction = predict(model, data['x'], data['t'], samples=config.get('prediction_samples', 100))
    reference = analytical_solution(data['x'], data['t'], physics.true_position, physics.true_amplitude, physics)
    arrays = {k: v.detach().cpu().numpy().reshape(data['shape']) for k, v in prediction.items()}
    arrays['reference'] = reference.detach().cpu().numpy().reshape(data['shape'])
    arrays['x'] = data['x'].cpu().numpy().reshape(data['shape'])
    arrays['t'] = data['t'].cpu().numpy().reshape(data['shape'])
    arrays['x_obs'] = data['x_obs'].cpu().numpy()
    arrays['t_obs'] = data['t_obs'].cpu().numpy()
    arrays['C_obs'] = data['C_obs'].cpu().numpy()
    np.savez_compressed(path / 'predictions.npz', **arrays)
    save_json(path / 'summary.json', {'position': float(model.position.detach().cpu()),
              'amplitude': float(model.amplitude.detach().cpu()),
              'rmse': float(np.sqrt(np.mean((arrays['mean'] - arrays['reference'])**2))),
              'interval': 'pointwise 2.5% and 97.5% stochastic weight-sample quantiles'})
    from .visualization import plot_results
    plot_results(arrays, history, path)
    return path


def main():
    parser = argparse.ArgumentParser(description='Two-stage Bayesian PINN groundwater inversion')
    parser.add_argument('--config', default='configs/demo.json')
    parser.add_argument('--output')
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as stream:
        config = json.load(stream)
    print(run(config, args.output))
