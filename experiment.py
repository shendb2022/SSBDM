"""Run SSBDM with supplied dictionary1 dictionaries and save MAT/jet PNG results."""
from dataclasses import asdict
from pathlib import Path
import time

import matplotlib.pyplot as plt
from scipy.io import savemat
import torch

from utils import ROOT, prepare_dataset, load_reference, detection_map, evaluate, suppress_background, save_jet_map
from solver import Config, solve


def run(args):
    if args.state != 'test':
        raise ValueError('state must be test')
    if args.device not in ('cuda', 'cpu') or args.dtype not in ('float32', 'float64'):
        raise ValueError('device must be cuda/cpu and dtype must be float32/float64')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; set device to cpu')
    if args.threads < 1:
        raise ValueError('threads must be positive')
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    name = Path(args.dataset).stem
    data_path = Path(args.data_path) if args.data_path else Path(args.data_root) / f'{name}.mat'
    background_path = Path(
        args.background_path) if args.background_path else ROOT / 'dictionary' / f'{name}_bg_dictionary.mat'
    out = Path(args.output) if args.output else ROOT / 'result' / name
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    inputs = load_reference(args.reference) if args.reference else prepare_dataset(
        data_path, background_path,
        data_key=args.data_key, gt_path=args.gt_path, gt_key=args.gt_key,
        target_label=args.target_label,
        background_atoms=args.background_atoms, kernel_gamma=args.kernel_gamma,
        batch_size=args.batch_size, pca_components=args.pca_components,
        num_superpixels_ratio=args.num_superpixels_ratio, compactness=args.compactness,
        sigma=args.sigma)
    sizes = list(map(len, inputs.indices))
    # print(f'X={inputs.x.shape}, M={inputs.m.shape}, groups={len(sizes)}, sizes={min(sizes)}..{max(sizes)}', flush=True)
    cfg = Config(alpha=args.alpha, beta=args.beta, mu=args.mu,
                 max_iter=args.max_iter, tolerance=args.tolerance)
    selected = {1, 10, 25, 50, 75, cfg.max_iter}
    # selected = {i for i in range(40, 101)}
    residuls = []

    def observe(iteration, state, metrics):
        if iteration in selected or metrics['residual'] < cfg.tolerance:
            print(f"iter={iteration:3d} mu={metrics['mu']:.6g} residual={metrics['residual']:.6g}", flush=True)
            residuls.append(metrics['residual'])

    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    result = solve(inputs.x, inputs.m, inputs.indices, inputs.laplacians, cfg,
                   device=args.device, dtype=getattr(torch, args.dtype), callback=observe)
    final_y = result.state['Y'].cpu().numpy()
    score = detection_map(inputs.m, final_y, inputs.labels.shape)
    metrics, fpr, tpr, thresholds = evaluate(inputs.groundtruth, score)
    display_map = suppress_background(score, args.suppression_scale)
    summary = dict(dataset=name, device=args.device, dtype=args.dtype,
                   target_source='saved reference' if args.reference else 'nearest target mean pixel',
                   input_normalization='saved reference' if args.reference else 'minmax float32 quantization then float64',
                   parameters=asdict(cfg), suppression_scale=args.suppression_scale,
                   background=inputs.background_info or dict(mode="reference"),
                   preprocessing_parameters=dict(pca_components=args.pca_components,
                                                 num_superpixels_ratio=args.num_superpixels_ratio,
                                                 actual_superpixels=len(sizes),
                                                 target_label=args.target_label,
                                                 compactness=args.compactness, sigma=args.sigma),
                   iterations=len(result.history),
                   converged=result.history[-1]['residual'] < cfg.tolerance,
                   final_residual=result.history[-1]['residual'], **metrics,

                   solver_setup_seconds=result.setup_seconds,
                   solver_iteration_seconds=result.iteration_seconds,
                   gpu_peak_allocated_mb=torch.cuda.max_memory_allocated() / 1024 ** 2 if args.device == 'cuda' else 0,
                   output=str(out.resolve()))
    path = out / 'SSBDM.mat'
    savemat(path, dict(map=display_map, detection_map=score,
                       AUC=metrics['auc'], groundtruth=inputs.groundtruth,
                       target_coefficients=final_y[-1], fpr=fpr, tpr=tpr,
                       thresholds=thresholds, **summary), do_compression=True)
    save_jet_map(out / 'SSBDM_jet.png', display_map)
    print(f"AUC={metrics['auc']:.6f}; saved: {path.resolve()}", flush=True)
    # plt.plot(residuls)
    # plt.xticks([0, 10, 20, 30, 40, 50, 60],[40, 50, 60, 70, 80, 90, 100])
    # plt.show()
    return summary
