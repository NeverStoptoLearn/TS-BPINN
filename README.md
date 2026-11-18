# TS-BPINN

Research code for **Two-Stage Bayesian Physics-Informed Neural Networks for Groundwater Contaminant Source Identification**.

**Authors:** Zhengchen Zhou, Xuesong Yan, Chengyu Hu  
**Publication:** Journal of Water Resources Planning and Management, 152(2) (2026), 04025075  
**Paper:** [10.1061/JWRMD5.WRENG-7084](https://doi.org/10.1061/JWRMD5.WRENG-7084)

Identify a groundwater contaminant source location and initial concentration amplitude using a stochastic neural network, an advection-dispersion residual and concentration observations.

## Model and configuration

The model solves `C_t + v C_x - D C_xx = 0`. Synthetic observations, initial conditions and plotting references use one Gaussian ADE solution:

```text
variance = width² + 2 D t
C(x,t) = amplitude * width / sqrt(variance)
         * exp(-(x - position - v t)² / (2 variance))
```

The amplitude is the Gaussian peak at time zero. Boundary targets are the same analytical solution evaluated at the finite interval endpoints. Observations include configurable Gaussian measurement noise. Learned source position is bounded inside the domain and learned amplitude is positive.

`physics`, `data`, `model`, `training` and `device` configure the experiment. The full configuration uses stochastic layers `2 → 256 → 128 → 256 → 1`, SiLU activations and Adam. Loss weights are ordered `[PDE, initial, boundary, observation]`: Stage 1 uses `[0.2, 1, 1, 0]`; Stage 2 uses `[0.2, 1, 1, 1]`. The normalized Gaussian-weight KL regularizer is controlled by `kl_weight` (default configuration `1e-4`; zero disables it).

Both CPU and CUDA are supported; all tensors and source parameters use the selected device. CUDA availability is checked before training.

## Outputs

- `model.pt`: trained state dictionary, model configuration and physical parameters.
- `training.csv`: total/component losses and source-parameter history.
- `predictions.npz`: coordinates, reference, predictive mean/standard deviation, sample quantiles and observations.
- `summary.json`: inferred parameters and concentration RMSE.
- `concentration.png`, `loss.png`, `uncertainty.png`: visualization.

Uncertainty bands are pointwise empirical 2.5% and 97.5% quantiles across stochastic weight samples. They quantify model variability and exclude added measurement noise; nominal 95% coverage is not guaranteed.

Reload a checkpoint with:

```python
import torch
from ts_bpinn.model import BayesianPINN

checkpoint = torch.load("runs/example/model.pt", map_location="cpu", weights_only=True)
model = BayesianPINN(length=checkpoint["physics"]["length"], **checkpoint["model_config"])
model.load_state_dict(checkpoint["state_dict"])
model.eval()
```

## Installation

Use Python 3.10–3.12 in a dedicated environment:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Linux/macOS, then install:

```bash
python -m pip install -e ".[dev]"
```

For a CPU-only PyTorch installation on Windows/Linux, first run `python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu`. The optional top-level `threads` setting controls CPU training threads (default 2).

## Run

From the project root:

```bash
python -m ts_bpinn --config configs/demo.json
python -m ts_bpinn --config configs/experiment.json
```

`demo.json` provides a small execution example; `experiment.json` provides a larger configuration. An explicit output path can be supplied with `--output runs/example`. The destination must not already exist. Without that argument, a new timestamped directory is created under `runs/`.

Every run saves `config.json` and `environment.json` alongside results. Sampling follows the algorithm's natural randomness; no fixed random seed is configured. The demo validates execution, not publication-level accuracy or convergence.

## Project structure

```text
configs/             Small and larger experiment configurations
src/ts_bpinn/       Importable Python implementation and command-line entry point
notebooks/demo.ipynb  Lightweight example calling the package
tests/               Numerical and workflow checks
.github/workflows/   Automated tests and demo execution
```

## Tests

```bash
python -m pytest -q
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{zhou2026tsbpinn,
  title   = {Two-Stage Bayesian Physics-Informed Neural Networks for Groundwater Contaminant Source Identification},
  author  = {Zhou, Zhengchen and Yan, Xuesong and Hu, Chengyu},
  journal = {Journal of Water Resources Planning and Management},
  volume  = {152},
  number  = {2},
  pages   = {04025075},
  year    = {2026},
  doi     = {10.1061/JWRMD5.WRENG-7084}
}
```

## Contact

Zhengchen Zhou (Zhouchen), China University of Geosciences (Wuhan)  
Email: [chenzz_14@outlook.com](mailto:chenzz_14@outlook.com)  
ORCID: [0009-0005-0160-1988](https://orcid.org/0009-0005-0160-1988)
