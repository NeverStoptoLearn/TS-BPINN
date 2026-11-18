import json
from pathlib import Path
import numpy as np
import torch
from ts_bpinn.cli import run
from ts_bpinn.model import BayesianPINN


def test_complete_run_saves_current_results_and_loadable_checkpoint(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'configs/demo.json').read_text())
    config['training']['stages'] = [{'epochs': 1, 'weights': [0.2, 1, 1, 0]}, {'epochs': 1, 'weights': [0.2, 1, 1, 1]}]
    config['prediction_samples'] = 3
    result = run(config, tmp_path / 'result')
    for filename in ('config.json', 'environment.json', 'summary.json', 'predictions.npz', 'model.pt', 'training.csv', 'concentration.png', 'uncertainty.png'):
        assert (result / filename).stat().st_size > 0
    with np.load(result / 'predictions.npz', allow_pickle=False) as arrays:
        assert np.isfinite(arrays['mean']).all()
        assert arrays['mean'].shape == arrays['reference'].shape
    checkpoint = torch.load(result / 'model.pt', weights_only=True)
    model = BayesianPINN(length=checkpoint['physics']['length'], **checkpoint['model_config'])
    model.load_state_dict(checkpoint['state_dict'])
    assert np.isfinite(float(model.position.detach()))
