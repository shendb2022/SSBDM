"""Load or regenerate the adaptive multi-kernel ATGP background dictionary.

The AMK/ATGP functions preserve upstream amk_atgp.py equations and seed dtype.
Generation never writes back to the input hyperspectral dataset.
"""
from pathlib import Path
import numpy as np
import scipy.io as sio


def standard(X):
    '''
    Standardization
    universal
    :param X:
    :return:
    '''
    min_x = np.min(X)
    max_x = np.max(X)
    if min_x == max_x:
        return np.zeros_like(X)
    return np.float32((X - min_x) / (max_x - min_x))


def linear_kernel(X, Y):
    """线性核函数"""
    return np.dot(X.T, Y)


def rbf_kernel(X, Y, gamma=1.0):
    """RBF核函数 - 统一版本"""
    XX = np.sum(X ** 2, axis=0, keepdims=True)
    YY = np.sum(Y ** 2, axis=0, keepdims=True)
    XY = np.dot(X.T, Y)
    dists = XX.T + YY - 2 * XY
    return np.exp(-gamma * dists)


def poly_kernel(X, Y, gamma=1, degree=2, coef0=1.0):
    """多项式核函数 - 统一版本"""
    return (gamma * np.dot(X.T, Y) + coef0) ** degree


def update_kernel_weights(X, U, kernel_funcs, param_list, use_batch=False, batch_size=1000):
    """统一的核权重更新函数"""
    L, N = X.shape
    _, m = U.shape
    residuals = []

    for func, params in zip(kernel_funcs, param_list):
        if use_batch and N > batch_size:
            # 分批计算模式
            total_residual = 0
            batch_count = 0

            for i in range(0, N, batch_size):
                i_end = min(i + batch_size, N)
                X_batch = X[:, i:i_end]

                k_xx_batch = np.diag(func(X_batch, X_batch, **params))
                k_xd_batch = func(X_batch, U, **params)
                k_dd = func(U, U, **params)

                k_dd_inv = np.linalg.pinv(k_dd + 1e-8 * np.eye(m))
                proj_batch = np.einsum('ni,ij,nj->n', k_xd_batch, k_dd_inv, k_xd_batch)
                res_batch = k_xx_batch - proj_batch

                total_residual += np.sum(res_batch)
                batch_count += (i_end - i)

            avg_residual = total_residual / batch_count
            avg_residual = max(avg_residual, 1e-8)
            residuals.append(avg_residual)
        else:
            # 全矩阵计算模式
            k_xx = np.diag(func(X, X, **params))
            k_xd = func(X, U, **params)
            k_dd = func(U, U, **params)

            k_dd_inv = np.linalg.pinv(k_dd + 1e-8 * np.eye(m))
            proj = np.einsum('ni,ij,nj->n', k_xd, k_dd_inv, k_xd)
            res = k_xx - proj
            mean_res = np.mean(res)
            mean_res = max(mean_res, 1e-8)
            residuals.append(mean_res)
            # residuals.append(np.mean(res) + 1e-8)

    # 统一的权重计算
    inv_res = [1 / r for r in residuals]
    alpha = np.array(inv_res) / np.sum(inv_res)
    return alpha


def combined_kernel(X1, X2, kernel_funcs, param_list, alpha, use_batch=False, batch_size=1000):
    """统一的组合核函数"""
    n1 = X1.shape[1]
    n2 = X2.shape[1]

    if use_batch and (n1 > batch_size or n2 > batch_size):
        # 分批计算组合核
        K_total = np.zeros((n1, n2))

        for i in range(0, n1, batch_size):
            i_end = min(i + batch_size, n1)
            X1_batch = X1[:, i:i_end]

            for a, func, params in zip(alpha, kernel_funcs, param_list):
                K_batch = func(X1_batch, X2, **params)
                K_total[i:i_end, :] += a * K_batch

        return K_total
    else:
        # 全矩阵计算
        K_total = 0
        for a, f, p in zip(alpha, kernel_funcs, param_list):
            K_total += a * f(X1, X2, **p)
        return K_total


def ATGP_adaptive_kernel_unified(X, d, n, kernel_funcs, param_list,
                                 use_batch=False, batch_size=100, verbose=True):
    """
    统一的ATGP自适应核版本

    Parameters:
    -----------
    use_batch : bool
        是否使用分批计算
    batch_size : int
        分批大小
    """
    L, N = X.shape
    U = d.copy() if d.ndim == 2 else d.reshape(-1, 1)
    indices = []

    for j in range(1, n + 1):
        try:
            if verbose and j % 10 == 0:
                print(f"Extracting {j}-th endmember...")

            # 更新核权重
            alpha = update_kernel_weights(X, U, kernel_funcs, param_list, use_batch, batch_size)

            # 初始化残差向量
            residual_comb = np.zeros(N)
            # residual_safe = np.zeros(N)

            if use_batch and N > batch_size:
                # 分批计算残差
                for i in range(0, N, batch_size):
                    i_end = min(i + batch_size, N)
                    X_batch = X[:, i:i_end]

                    # 组合核残差
                    k_xx_comb = np.diag(
                        combined_kernel(X_batch, X_batch, kernel_funcs, param_list, alpha, use_batch, batch_size))
                    k_xd_comb = combined_kernel(X_batch, U, kernel_funcs, param_list, alpha, use_batch, batch_size)
                    k_dd_comb = combined_kernel(U, U, kernel_funcs, param_list, alpha, use_batch, batch_size)
                    k_dd_inv_comb = np.linalg.pinv(k_dd_comb + 1e-8 * np.eye(k_dd_comb.shape[0]))
                    proj_comb = np.einsum('ni,ij,nj->n', k_xd_comb, k_dd_inv_comb, k_xd_comb)
                    residual_comb[i:i_end] = k_xx_comb - proj_comb

            else:
                # 全矩阵计算残差
                # 组合核残差
                k_xx_comb = np.diag(combined_kernel(X, X, kernel_funcs, param_list, alpha))
                k_xd_comb = combined_kernel(X, U, kernel_funcs, param_list, alpha)
                k_dd_comb = combined_kernel(U, U, kernel_funcs, param_list, alpha)
                k_dd_inv_comb = np.linalg.pinv(k_dd_comb + 1e-8 * np.eye(k_dd_comb.shape[0]))
                proj_comb = np.einsum('ni,ij,nj->n', k_xd_comb, k_dd_inv_comb, k_xd_comb)
                residual_comb = k_xx_comb - proj_comb

            residual = residual_comb
            if verbose:
                print(f"max residual = {np.max(residual):.6f}")
            # 找到最大残差对应的像素
            argmax = np.argmax(residual)
            indices.append(argmax)

            # 更新端元矩阵
            new_endmember = X[:, argmax:argmax + 1]
            U = np.hstack([U, new_endmember])

        except Exception as e:
            print(f"Error at iteration {j}: {e}")
            # 简单的fallback策略
            if U.shape[1] == 1:
                norms = np.linalg.norm(X, axis=0)
                argmax = np.argmax(norms)
            else:
                # 使用线性投影残差
                U_pinv = np.linalg.pinv(U)
                proj = U @ (U_pinv @ X)
                residuals_linear = np.sum((X - proj) ** 2, axis=0)
                argmax = np.argmax(residuals_linear)

            indices.append(argmax)
            new_endmember = X[:, argmax:argmax + 1]
            U = np.hstack([U, new_endmember])

    return U[:, 1:], indices


def generate_background(x, target, n_atoms=20, gamma=1e-4, batch_size=100):
    x = np.asarray(x, dtype=np.float32)
    target = np.asarray(target).reshape(-1, 1)
    if x.ndim != 2 or target.shape != (x.shape[0], 1):
        raise ValueError('X and target must have matching spectral dimensions')
    if not 1 <= n_atoms <= x.shape[1] or batch_size < 1 or gamma <= 0:
        raise ValueError('invalid background atom count, batch size or kernel gamma')
    if not np.isfinite(x).all() or not np.isfinite(target).all():
        raise ValueError('X and target must be finite')
    kernels = [linear_kernel, rbf_kernel, poly_kernel]
    parameters = [{}, {'gamma': gamma}, {'gamma': 1., 'degree': 2, 'coef0': 1.}]
    return ATGP_adaptive_kernel_unified(x, target, n_atoms, kernels, parameters,
                                        use_batch=True, batch_size=batch_size, verbose=False)


def get_background(x, target, path=None, *, n_atoms=20, gamma=1e-4, batch_size=100):
    """Load a cached dictionary; generate and cache it if the file is missing."""
    path = Path(path) if path is not None else None
    if path is not None and path.exists():
        cached = sio.loadmat(path)
        a = np.asarray(cached['A'])
        if 'dictionary_seed' in cached and not np.array_equal(cached['dictionary_seed'], target):
            raise ValueError(
                f'dictionary seed changed for {path}; use another background_path or regenerate this cache')
        if 'kernel_gamma' in cached and float(cached['kernel_gamma'].item()) != gamma:
            raise ValueError(
                f'dictionary kernel gamma changed for {path}; use another background_path or regenerate this cache')
        if a.shape != (x.shape[0], n_atoms) or not np.isfinite(a).all():
            raise ValueError(
                f'expected a finite background dictionary of shape {(x.shape[0], n_atoms)}, got {a.shape}; choose a matching atom count or another background_path')
        print(f'Loaded background dictionary: {path}', flush=True)
        return a, dict(mode='load', path=str(path.resolve()), atoms=n_atoms,
                       seed_source=str(cached[
                                           'seed_source'].item()) if 'seed_source' in cached else 'legacy cache without seed metadata')
    print(f'Generating background dictionary ({n_atoms} atoms)...', flush=True)
    a, indices = generate_background(x, target, n_atoms, gamma, batch_size)
    info = dict(mode='generate', atoms=n_atoms, kernel_gamma=gamma, batch_size=batch_size,
                pixel_indices=[int(i) for i in indices],
                seed_dtype=str(np.asarray(target).dtype), seed_source='ts_generation type7')
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        sio.savemat(path, dict(A=a, pixel_indices=np.asarray(indices),
                               dictionary_seed=np.asarray(target), kernel_gamma=gamma,
                               seed_source='ts_generation type7',
                               generation_version='type7-v1'), do_compression=True)
        info['path'] = str(path.resolve())
        print(f'Saved background dictionary: {path}', flush=True)
    return a, info
