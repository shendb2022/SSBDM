from pathlib import Path
from types import SimpleNamespace
from experiment import run
import time

ROOT = Path(__file__).resolve().parent

# Shared settings and all dataset parameters are configured here.
modelConfig = {
    'dataset': 'Sandiego',
    'state': 'test',
    'background_atoms': 20,
    'num_superpixels_ratio': 0.01,
    'alpha': 0.01,
    'beta': 0.1,
    'data_root': 'datasets',
    'data_path': None,
    'background_path': None,
    'kernel_gamma': 0.0001,
    'batch_size': 100,
    'pca_components': 10,
    'compactness': 0.01,
    'sigma': 0.5,
    'suppression_scale': 0.16,
    'mu': 0.1,
    'max_iter': 100,
    'tolerance': 0.001,
    'device': 'cuda',
    'dtype': 'float32',
    'threads': 4,
    'output': None,
    'reference': None,
    'data_key': 'data',
    'gt_path': None,
    'gt_key': None,
    'target_label': 1,
}


def load_config(dataset=None, background_atoms=None,
                alpha=None, beta=None, num_superpixels_ratio=None, **overrides):
    config = modelConfig.copy()
    explicit = dict(dataset=dataset, background_atoms=background_atoms,
                    alpha=alpha, beta=beta, num_superpixels_ratio=num_superpixels_ratio)
    config.update({key: value for key, value in explicit.items() if value is not None})
    unknown = set(overrides) - set(config)
    if unknown:
        raise ValueError(f'unknown configuration keys: {sorted(unknown)}')
    config.update(overrides)
    config['data_root'] = str((ROOT / Path(config['data_root'])).resolve())
    for key in ('data_path', 'gt_path'):
        if config[key] is not None:
            config[key] = str((Path(config['data_root']) / Path(config[key])).resolve())
    return config


def main(dataset=None, background_atoms=None,
         alpha=None, beta=None, num_superpixels_ratio=None, **overrides):
    """Run with top-level defaults and optional explicit parameter overrides."""
    return run(
        SimpleNamespace(**load_config(dataset, background_atoms, alpha, beta, num_superpixels_ratio, **overrides)))


if __name__ == '__main__':
    start = time.perf_counter()
    main('Sandiego', 20, 0.01, 0.1, 0.01)
    # main('Sandiego2', 10, 0.1, 0.01, 0.01)
    # main('Cuprite', 10, 0.01, 0.1, 0.008)
    # main('Xuzhou', 10, 0.001, 0.01, 0.002)
    end = time.perf_counter()
    print('time is %s' % (end - start))
