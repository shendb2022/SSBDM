"""Preprocessing matching main_sandiego.m, including both pixel conventions."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_curve, roc_auc_score

from superpixel import hybrid_segment
from ts_generation import ts_generation

ROOT = Path(__file__).resolve().parent


@dataclass
class Inputs:
    x: np.ndarray
    m: np.ndarray
    labels: np.ndarray
    indices: list
    laplacians: list
    groundtruth: np.ndarray
    cube: np.ndarray | None = None
    background_info: dict | None = None


def normalize(array):
    array = np.asarray(array, dtype=np.float64)
    span = array.max() - array.min()
    return (array - array.min()) / span if span else np.zeros_like(array)


def image_to_matrix(cube):
    """MATLAB reshape(data,h*w,bands)' : pixel = row + column * height."""
    return np.ascontiguousarray(cube.reshape(-1, cube.shape[2], order='F').T)


def build_dictionary(a, d):
    a, d = np.asarray(a, dtype=np.float64), np.asarray(d, dtype=np.float64).reshape(-1, 1)
    b = np.column_stack((a, d))
    products = [b[:, i] * b[:, j] for i in range(b.shape[1] - 1) for j in range(i + 1, b.shape[1])]
    return np.column_stack([a, *products, d])


def graph_inputs(cube, labels, sigma=0.5):
    """find(segment_label'==g) orders pixels by row, then column.

    Index into X using MATLAB column-major positions, while retaining the
    row-major ordering used to construct each graph Laplacian.
    """
    indices, laplacians = [], []
    h, w = labels.shape
    if not np.array_equal(np.unique(labels), np.arange(1, labels.max() + 1)):
        raise ValueError('labels must be contiguous positive integers starting at 1')
    for g in range(1, int(labels.max()) + 1):
        rows, cols = np.nonzero(labels == g)
        indices.append((rows + cols * h).astype(np.int64))
        points = cube[rows, cols]
        affinity = np.exp(-cdist(points, points, 'sqeuclidean') / sigma ** 2)
        np.fill_diagonal(affinity, 0)
        laplacians.append(np.diag(affinity.sum(axis=1)) - affinity)
    return indices, laplacians


def load_dataset(path, *, data_key='data', gt_path=None, gt_key=None, target_label=1,
                 return_background_target=False):
    raw = loadmat(path)
    # Original amk_atgp.standard saves normalized images as float32; MATLAB
    # then promotes those stored values to double before PCA/segmentation.
    # Preserve that quantization even when loading the original integer cube.
    if data_key not in raw:
        raise ValueError(f'data variable {data_key!r} is missing from {path}')
    cube = normalize(np.maximum(raw[data_key], 0)).astype(np.float32).astype(np.float64)
    labels = loadmat(gt_path) if gt_path is not None else raw
    truth = labels.get(gt_key) if gt_key is not None else labels.get('groundtruth', labels.get('map'))
    if cube.ndim != 3:
        raise ValueError('dataset data must have shape H x W x bands')
    if truth is None or truth.shape != cube.shape[:2]:
        raise ValueError('dataset must contain a matching groundtruth (or map) image')
    truth = (truth == target_label).astype(np.uint8)
    # Use original ts_generation(type=7) consistently for every dataset.
    # A prestored d may use a different target or normalization, so ignore it.
    if not np.any(truth):
        raise ValueError('dataset must contain target pixels labeled 1')
    target = ts_generation(cube.astype(np.float32), truth, type=7).astype(np.float64)
    print('Extracted target spectrum from labels (nearest target mean).', flush=True)
    if target.shape != (cube.shape[-1], 1):
        raise ValueError('target spectrum d must have shape bands x 1')
    if return_background_target:
        # Share the type-7 prior with detection; ignore prestored d for both.
        background_target = target.astype(np.float32)
        if background_target.shape != target.shape or not np.isfinite(background_target).all():
            raise ValueError('dictionary seed d must be finite and have shape bands x 1')
        return cube, target, truth, background_target
    return cube, target, truth


def prepare_dataset(dataset_path, background_path=None, *,
                    data_key='data', gt_path=None, gt_key=None, target_label=1,
                    background_atoms=20, kernel_gamma=1e-4, batch_size=100,
                    pca_components=10, num_superpixels_ratio=0.01, compactness=0.01,
                    sigma=0.5):
    from get_background import get_background
    cube, target, truth, background_target = load_dataset(
        dataset_path, data_key=data_key, gt_path=gt_path, gt_key=gt_key,
        target_label=target_label, return_background_target=True)
    x = image_to_matrix(cube)
    num_superpixels = int(x.shape[-1] * num_superpixels_ratio)
    if not 1 <= pca_components <= cube.shape[-1]:
        raise ValueError('pca_components must be between 1 and the number of bands')
    if not isinstance(num_superpixels, (int, np.integer)) or not 1 <= num_superpixels <= cube.shape[0] * cube.shape[
        1] or sigma <= 0:
        raise ValueError('num_superpixels must be a valid positive integer and sigma must be positive')
    a, info = get_background(x, background_target, background_path, n_atoms=background_atoms,
                             gamma=kernel_gamma, batch_size=batch_size)
    m = build_dictionary(a, target)
    # Original PCA finds covariance directions but projects uncentered data.
    _, vectors = eigh(np.cov(x), driver='evd')
    scores = x.T @ vectors[:, ::-1][:, :pca_components]
    features = scores.reshape(*cube.shape[:2], pca_components, order='F')
    labels = hybrid_segment(features, num_superpixels, compactness)
    indices, laplacians = graph_inputs(cube, labels, sigma)
    inputs = Inputs(x, m, labels, indices, laplacians, truth, cube)
    inputs.background_info = info
    return inputs


def prepare_sandiego(root=ROOT):
    root = Path(root)
    return prepare_dataset(root / 'Sandiego.mat', root / 'dictionary/Sandiego_bg_dictionary.mat')


def load_reference(path):
    """Read X/M/segment_label/Lgs/groundtruth from MATLAB exporter."""
    raw = loadmat(path)
    labels = raw['segment_label'].astype(np.int64)
    h, w = labels.shape
    if not np.array_equal(np.unique(labels), np.arange(1, labels.max() + 1)):
        raise ValueError('reference labels must be contiguous and start at 1')
    indices, laplacians = [], []
    for g in range(1, int(labels.max()) + 1):
        rows, cols = np.nonzero(labels == g)
        indices.append((rows + cols * h).astype(np.int64))
        n = len(rows)
        laplacians.append(np.asarray(raw['Lgs'][g - 1, :n, :n], dtype=np.float64))
    return Inputs(np.asarray(raw['X'], dtype=np.float64), np.asarray(raw['M'], dtype=np.float64),
                  labels, indices, laplacians, raw['groundtruth'])


def detection_map(m, y, shape):
    energy = np.linalg.norm(m[:, -1]) * np.abs(y[-1])
    return normalize(energy.reshape(shape, order='F'))


def evaluate(truth, scores):
    truth, scores = np.asarray(truth), np.asarray(scores)
    if truth.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError('truth and finite detection scores must have matching shapes')
    if not np.array_equal(np.unique(truth), [0, 1]):
        raise ValueError('ground truth must contain background 0 and target 1')
    fpr, tpr, thresholds = roc_curve(truth.ravel(), scores.ravel(), drop_intermediate=False)
    return dict(auc=float(roc_auc_score(truth.ravel(), scores.ravel())),
                target_pixels=int(truth.sum()), background_pixels=int((truth == 0).sum())), fpr, tpr, thresholds


def suppress_background(scores, scale=0.16):
    """Original MATLAB display transform: exp(-(map-1).^2 / 0.16)."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('suppression_scale must be positive and finite')
    return np.exp(-(np.asarray(scores) - 1) ** 2 / scale)


def save_jet_map(path, image):
    """Save the final suppressed map with jet colors and fixed [0,1] range."""
    from matplotlib.image import imsave
    imsave(path, image, cmap='jet', vmin=0, vmax=1)
