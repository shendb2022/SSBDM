"""SSBDM ADMM updates faithful to SSBDM3_noweighted_sparsity.m."""
from dataclasses import dataclass
import time
import numpy as np
import torch


@dataclass(frozen=True)
class Config:
    alpha: float = 0.01
    beta: float = 0.1
    mu: float = 0.1
    max_iter: int = 100
    tolerance: float = 0.001


@dataclass
class Result:
    state: dict
    history: list
    setup_seconds: float
    iteration_seconds: float
    callback_seconds: float


def soft(x, threshold):
    # MATLAB soft.m deliberately uses this expression rather than sign(x).
    # MATLAB eps is double epsilon, even when the experiment uses float32.
    threshold = threshold + np.finfo(np.float64).eps
    magnitude = torch.clamp(torch.abs(x) - threshold, min=0)
    return magnitude / (magnitude + threshold) * x


def _validate(x, m, indices, laps, cfg):
    if x.ndim != 2 or m.ndim != 2 or x.shape[0] != m.shape[0]:
        raise ValueError('X and M must be bands x pixels/atoms matrices with matching bands')
    if not np.isfinite(x).all() or not np.isfinite(m).all():
        raise ValueError('X and M must be finite')
    if cfg.mu <= 0 or cfg.alpha < 0 or cfg.beta < 0 or cfg.max_iter < 1 or cfg.tolerance < 0:
        raise ValueError('invalid solver parameters')
    if len(indices) != len(laps) or not indices:
        raise ValueError('one Laplacian is required per superpixel')
    flat = np.concatenate(indices)
    if not np.array_equal(np.sort(flat), np.arange(x.shape[1])):
        raise ValueError('superpixel indices must partition every pixel exactly once')
    for idx, lap in zip(indices, laps):
        if len(idx) == 0 or lap.shape != (len(idx), len(idx)) or not np.isfinite(lap).all():
            raise ValueError('invalid graph shape or values')
        if not np.allclose(lap, lap.T, rtol=1e-10, atol=1e-10):
            raise ValueError('graph Laplacians must be symmetric')


@torch.no_grad()
def solve(x, m, indices, laplacians, config=None, *, device='cuda', dtype=torch.float64, callback=None,
          graph_batch_size=None):
    cfg = config or Config()
    x, m = np.asarray(x), np.asarray(m)
    _validate(x, m, indices, laplacians, cfg)
    if graph_batch_size is None:
        width = max(map(len, indices))
        graph_bytes = len(indices) * width * width * torch.empty((), dtype=dtype).element_size()
        graph_batch_size = len(indices) if graph_bytes <= 256 * 1024 ** 2 else 16
    if not isinstance(graph_batch_size, int) or graph_batch_size < 1:
        raise ValueError("graph_batch_size must be a positive integer")
    device = torch.device(device)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    sync()
    started = time.perf_counter()
    x = torch.as_tensor(x, dtype=dtype, device=device)
    m = torch.as_tensor(m, dtype=dtype, device=device)
    bands, n = x.shape
    q = m.shape[1]
    v1 = torch.zeros_like(x)
    v2 = torch.zeros((q, n), dtype=dtype, device=device)
    v3 = torch.zeros_like(v2)
    d1, d2, d3 = torch.zeros_like(v1), torch.zeros_like(v2), torch.zeros_like(v3)
    chol_y = torch.linalg.cholesky(m.T @ m + 2 * torch.eye(q, dtype=dtype, device=device))
    m_normalized = m / (torch.linalg.vector_norm(m, dim=0) + 1e-8)
    # Group similar sizes so a single large region does not pad every graph.
    # This bounds Cholesky workspace and keeps Xuzhou within GPU memory.
    order = sorted(range(len(indices)), key=lambda g: len(indices[g]))
    graph_batches = []
    for offset in range(0, len(order), graph_batch_size):
        group_ids = order[offset:offset + graph_batch_size]
        width = max(len(indices[g]) for g in group_ids)
        packed_idx = np.zeros((len(group_ids), width), dtype=np.int64)
        mask = np.zeros((len(group_ids), width), dtype=bool)
        padded_laps = np.zeros((len(group_ids), width, width), dtype=np.float64)
        for local, g in enumerate(group_ids):
            k = len(indices[g])
            packed_idx[local, :k] = indices[g]
            mask[local, :k] = True
            padded_laps[local, :k, :k] = laplacians[g]
        packed_idx = torch.as_tensor(packed_idx, device=device)
        mask = torch.as_tensor(mask, device=device)
        lap = torch.as_tensor(padded_laps, dtype=dtype, device=device)
        graph_batches.append((packed_idx, mask, packed_idx[mask], 2 * cfg.alpha * lap))
    del lap, padded_laps
    history, mu = [], cfg.mu
    sync()
    setup_seconds = time.perf_counter() - started
    iteration_seconds = callback_seconds = 0.
    for iteration in range(1, cfg.max_iter + 1):
        tick = time.perf_counter()
        y = torch.cholesky_solve(m.T @ (v1 + d1) + v2 + d2 + v3 + d3, chol_y)
        my = m @ y
        v1 = (x + mu * (my - d1)) / (1 + mu)
        for packed_idx, mask, valid_indices, scaled_lap in graph_batches:
            # G x padded_pixels x atoms; filler RHS=0 and graph blocks=0.
            rhs = (y - d2)[:, packed_idx].permute(1, 2, 0).contiguous()
            rhs *= mask.unsqueeze(-1)
            rhs *= mu
            system = scaled_lap.clone()
            system.diagonal(dim1=-2, dim2=-1).add_(mu)
            graph_chol = torch.linalg.cholesky(system)
            solved = torch.cholesky_solve(rhs, graph_chol)
            v2[:, valid_indices] = solved[mask].T
            del rhs, system, graph_chol, solved
        temp = y - d3
        normalized_my = my / (torch.linalg.vector_norm(my, dim=0) + 1e-8)
        distance = 1 - m_normalized.T @ normalized_my
        row_weight = 1 / (torch.linalg.vector_norm(temp, dim=1, keepdim=True) + 1e-8)
        v3 = soft(temp, (cfg.beta / mu) * row_weight * distance)
        diff1, diff2, diff3 = my - v1, y - v2, y - v3
        d1, d2, d3 = d1 - diff1, d2 - diff2, d3 - diff3
        norms = torch.stack([torch.linalg.vector_norm(v) for v in (diff1, diff2, diff3)])
        residuals = norms.cpu().tolist()  # also synchronizes before timing/stopping
        residual = sum(residuals)
        mu_next = min(mu * 1.1, 1e30)
        metrics = dict(iteration=iteration, mu=mu, mu_next=mu_next, residual=residual,
                       residual_v1=residuals[0], residual_v2=residuals[1], residual_v3=residuals[2])
        if not np.isfinite(residual):
            raise FloatingPointError(f'non-finite residual at iteration {iteration}')
        iteration_seconds += time.perf_counter() - tick
        state = dict(Y=y, V1=v1, V2=v2, V3=v3, D1=d1, D2=d2, D3=d3)
        history.append(metrics)
        if callback:
            tick = time.perf_counter()
            callback(iteration, state, metrics)
            sync()
            callback_seconds += time.perf_counter() - tick
        mu = mu_next
        if residual < cfg.tolerance:
            break
    return Result(state, history, setup_seconds, iteration_seconds, callback_seconds)
