import numpy as np


def pool(x, pool_size=7):
    """
    Implement the pooling featurizer.
    Args:
        x: n x d matrix with a d-dimensional feature for each of the n points
        pool_size: size of the pooling window
    Returns:
        feat: n x dd matrix with the pooled features for each of the n points
            dd = d // pool_size ** 2
    """
    # Each sample is a 28 x 28 image, so reshaping to that size.
    x = x.reshape(-1, 28, 28)
    # BEGIN YOUR CODE
    x = x.reshape(-1, 28, 28)
    N, H, W = x.shape
    x_reshaped = x.reshape(N, H // pool_size, pool_size, W // pool_size, pool_size)
    pooled = x_reshaped.mean(axis=(2, 4))
    feat = pooled.reshape(N, -1)
    return feat
    # END YOUR CODE


def hog(x, pool_size=7, angle_bins=18):
    """
    Implement the Histogram of Gradient featurizer.
    Args:
        x: n x d matrix with a d-dimensional feature for each of the n points
        pool_size: size of the pooling window
        angle_bins: number of bins to use for the angle histogram
            For example, if angle_bins=18, then you should split the gradient
            orientation into 18 equal bins between 0 and 360, each one spanning
            20 degrees.
    Returns:
        feat: n x dd matrix with the HOG features for each of the n points
            dd = d // pool_size ** 2 * angle_bins
    """
    # Each sample is a 28 x 28 image, so reshaping to that size.
    x = x.reshape(-1, 28, 28)
    # BEGIN YOUR CODE
    n, h, w = x.shape
    gx = np.zeros_like(x)
    gy = np.zeros_like(x)
    
    gx[:, :, 1:-1] = x[:, :, 2:] - x[:, :, :-2]
    gy[:, 1:-1, :] = x[:, 2:, :] - x[:, :-2, :]

    mag = np.sqrt(gx**2 + gy**2)
    angle = np.rad2deg(np.arctan2(gy, gx)) % 360

    num_cells_h = h // pool_size
    num_cells_w = w // pool_size
    hog_features = np.zeros((n, num_cells_h, num_cells_w, angle_bins))

    for i in range(n):
        for y in range(num_cells_h):
            for x_ in range(num_cells_w):
                y_start = y * pool_size
                y_end = (y + 1) * pool_size
                x_start = x_ * pool_size
                x_end = (x_ + 1) * pool_size

                cell_mag = mag[i, y_start:y_end, x_start:x_end].flatten()
                cell_angle = angle[i, y_start:y_end, x_start:x_end].flatten()

                hist, _ = np.histogram(cell_angle, bins=angle_bins, range=(0, 360), weights=cell_mag)
                hog_features[i, y, x_, :] = hist

    return hog_features.reshape(n, -1)
    # END YOUR CODE


def featurize(x, type='raw', pool_size=7, angle_bins=18):
    if type == 'raw':
        x = x.reshape(x.shape[0], -1) - 0.5
    elif type == 'pool':
        x = pool(x, pool_size=pool_size)
    elif type == 'hog':
        x = hog(x, pool_size=pool_size, angle_bins=angle_bins)
    return x
