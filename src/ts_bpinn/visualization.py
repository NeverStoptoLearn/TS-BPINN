import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_results(arrays, history, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, key in zip(axes, ('reference', 'mean', 'std')):
        image = ax.pcolormesh(arrays['t'], arrays['x'], arrays[key], shading='auto')
        ax.set(xlabel='Time', ylabel='Position', title=key)
        fig.colorbar(image, ax=ax)
    fig.tight_layout(); fig.savefig(path / 'concentration.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots()
    ax.plot([r['loss'] for r in history])
    ax.set(xlabel='Training step', ylabel='Total loss', yscale='log')
    fig.tight_layout(); fig.savefig(path / 'loss.png', dpi=160); plt.close(fig)
    index = arrays['t'].shape[1] - 1
    x = arrays['x'][:, index]
    fig, ax = plt.subplots()
    ax.plot(x, arrays['reference'][:, index], label='Reference')
    ax.plot(x, arrays['mean'][:, index], label='Predictive mean')
    ax.fill_between(x, arrays['lower'][:, index], arrays['upper'][:, index], alpha=0.2, label='95% weight-sample interval')
    ax.set(xlabel='Position', ylabel='Concentration', title=f"Time = {arrays['t'][0, index]:g}")
    ax.legend(); fig.tight_layout(); fig.savefig(path / 'uncertainty.png', dpi=160); plt.close(fig)
