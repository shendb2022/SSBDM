"""NumPy port of the original hybridseg SAD-SLIC and connectivity algorithm."""
import numpy as np


def hybrid_segment(features, desired, compactness=0.01):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 3 or not np.isfinite(features).all():
        raise ValueError('features must be a finite H x W x bands array')
    height, width, bands = features.shape
    size = height * width
    if not 1 <= desired <= size or not 0 <= compactness <= 1:
        raise ValueError('invalid superpixel count or compactness')
    step = int(np.sqrt(size / desired) + 0.5)
    if step > min(height, width):
        raise ValueError('superpixel count is incompatible with image size')
    spectra = features.reshape(size, bands)
    squared_norms = np.zeros(size)
    for band in range(bands):
        squared_norms += spectra[:, band] * spectra[:, band]
    spectral_norms = np.sqrt(squared_norms)
    if np.any(spectral_norms == 0):
        raise ValueError('SAD segmentation is undefined for zero spectral vectors')
    strips_x, strips_y = int(0.5 + width / step), int(0.5 + height / step)
    if width - step * strips_x < 0:
        strips_x -= 1
    if height - step * strips_y < 0:
        strips_y -= 1
    err_x, err_y = (width - step * strips_x) / strips_x, (height - step * strips_y) / strips_y
    sx = np.array([x * step + step // 2 + int(x * err_x) for y in range(strips_y) for x in range(strips_x)],
                  dtype=float)
    sy = np.array([y * step + step // 2 + int(y * err_y) for y in range(strips_y) for x in range(strips_x)],
                  dtype=float)
    count = len(sx)
    centers = features[sy.astype(int), sx.astype(int)].copy()
    labels = np.full(size, -1, dtype=np.int32)
    grid_y, grid_x = np.indices((height, width))
    for _ in range(10):
        distances = np.full((height, width), np.inf)
        image_labels = labels.reshape(height, width)
        for g in range(count):
            x1, x2 = max(0, int(sx[g] - step)), min(width, int(sx[g] + step))
            y1, y2 = max(0, int(sy[g] - step)), min(height, int(sy[g] + step))
            pixels = features[y1:y2, x1:x2]
            # Accumulate bands in the C++ loop order: seed-pixel acos NaNs at
            # round-off boundaries affect assignments, so do not clip them.
            inner = np.zeros(pixels.shape[:2])
            center_norm = 0.0
            for band in range(bands):
                inner += pixels[:, :, band] * centers[g, band]
                center_norm += centers[g, band] * centers[g, band]
            denominator = spectral_norms.reshape(height, width)[y1:y2, x1:x2] * np.sqrt(center_norm)
            with np.errstate(invalid='ignore', divide='ignore'):
                spectral_distance = np.arccos(inner / denominator)
            spatial_distance = np.sqrt((grid_x[y1:y2, x1:x2] - sx[g]) ** 2 + (grid_y[y1:y2, x1:x2] - sy[g]) ** 2)
            candidate = spectral_distance * (1 - compactness) + spatial_distance * (compactness / step)
            better = candidate < distances[y1:y2, x1:x2]
            distances[y1:y2, x1:x2][better] = candidate[better]
            image_labels[y1:y2, x1:x2][better] = g
        # As in the original C++, temporarily unassigned pixels do not
        # contribute to centroids; later iterations may assign them.
        valid = labels >= 0
        counts = np.bincount(labels[valid], minlength=count).clip(min=1)
        sums = np.zeros((count, bands))
        np.add.at(sums, labels[valid], spectra[valid])
        inverse = 1.0 / counts
        centers = sums * inverse[:, None]
        sx = np.bincount(labels[valid], weights=grid_x.ravel()[valid], minlength=count) * inverse
        sy = np.bincount(labels[valid], weights=grid_y.ravel()[valid], minlength=count) * inverse
    if np.any(labels < 0):
        raise ValueError('SAD clustering left pixels unassigned after 10 iterations')
    return _connectivity(labels, height, width, count)


def _connectivity(labels, height, width, count):
    """Original raster traversal and left/up/right/down flood-fill order."""
    output = np.full(height * width, -1, dtype=np.int32)
    label, adjacent = 0, 0
    small_region = (height * width // count) >> 2
    offsets = ((-1, 0), (0, -1), (1, 0), (0, 1))
    for origin in range(height * width):
        if output[origin] >= 0:
            continue
        output[origin] = label
        row, col = divmod(origin, width)
        for dx, dy in offsets:
            x, y = col + dx, row + dy
            if 0 <= x < width and 0 <= y < height and output[y * width + x] >= 0:
                adjacent = output[y * width + x]
        queue = [origin]
        cursor = 0
        while cursor < len(queue):
            current = queue[cursor]
            row, col = divmod(current, width)
            for dx, dy in offsets:
                x, y = col + dx, row + dy
                if 0 <= x < width and 0 <= y < height:
                    neighbor = y * width + x
                    if output[neighbor] < 0 and labels[origin] == labels[neighbor]:
                        output[neighbor] = label
                        queue.append(neighbor)
            cursor += 1
        if len(queue) <= small_region:
            output[queue] = adjacent
            label -= 1
        label += 1
    return output.reshape(height, width) + 1
