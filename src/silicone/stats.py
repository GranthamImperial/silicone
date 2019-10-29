"""
Silicone's custom statistical operations.
"""
import numpy as np
import pandas as pd


def rolling_window_find_quantiles(xs, ys, quantiles, nwindows=10, decay_length_factor=1):
    """
    Perform quantile analysis over a number of different windows.

    This is a custom implementation which is a bit like a quantile regression, but not
    quite the same. TODO: describe method fully here

    Parameters
    ----------
    xs : np.ndarray, :obj:`pd.Series`
        The x values to use in the regression

    ys : np.ndarray, :obj:`pd.Series`
        The y values to use in the regression

    quantiles : list-like
        The quantiles to find in each window

    nwindows : int
        The number of windows to use

    decay_length_factor : float
        Parameter which controls how strongly points away from the window's centre
        should be weighted compared to points at the centre. Larger values give points
        further away increasingly less weight, smaller values give points further away
        increasingly more weight.

    Returns
    -------
    :obj:`pd.DataFrame`
        Quantile values at the window centres.
    """

    assert xs.size == ys.size
    if xs.size == 1:
        return pd.DataFrame(index=[xs[0]] * nwindows, columns=quantiles, data=ys[0])
    step = (max(xs) - min(xs)) / (nwindows + 1)
    decay_length = step / 2 * decay_length_factor
    # We re-form the arrays in case they were pandas series with integer labels that would mess up the sorting.
    xs = np.array(xs)
    ys = np.array(ys)
    sort_order = np.argsort(ys)
    ys = ys[sort_order]
    xs = xs[sort_order]
    if max(xs) == min(xs):
        # We must prevent singularity behaviour if all the points are at the same x value.
        box_centers = np.array([xs[0]])
        decay_length = 1
    else:
        # We want to include the max x point, but not any point above it.
        # The 0.99 factor prevents rounding error inclusion.
        box_centers = np.arange(min(xs), max(xs) + step * 0.99, step)
    quantmatrix = pd.DataFrame(index=box_centers, columns=quantiles)
    for ind in range(box_centers.size):
        weights = 1.0 / (1.0 + ((xs - box_centers[ind]) / decay_length) ** 2)
        weights /= sum(weights)
        # We want to calculate the weights at the midpoint of step corresponding to the y-value.
        cumsum_weights = np.cumsum(weights)
        for i_quantile in range(quantiles.__len__()):
            quantmatrix.iloc[ind, i_quantile] = min(
                ys[cumsum_weights >= quantiles[i_quantile]]
            )
    return quantmatrix
