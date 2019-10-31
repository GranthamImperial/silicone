"""
Silicone's custom statistical operations.
"""
import numpy as np
import pandas as pd


def rolling_window_find_quantiles(
    xs, ys, quantiles, nwindows=10, decay_length_factor=1
):
    """
    Perform quantile analysis in the y-direction for x-weighted data.

    Divides the x-axis into nwindows of equal length and weights data by how close they
    are to the center of these boxes. Then returns the quantiles of this weighted data.
    Quantiles are defined so that the values returned are always equal to a y-value in
    the data - there is no interpolation. Extremal points are given their full
    weighting, meaning this will not agree with the np.quantiles under uniform weighting
    (which effectively gives 0 weight to min and max values)

    The weighting of a point at :math:`x` for a window centered at :math:`x_0` is:

    .. math::
        w = \\frac{1}{1 + \\left (\\frac{x - x_0} {\\text{box_length} \\times \\text{decay_length_factor}} \\right )^2}

    Parameters
    ----------
    xs : np.ndarray, :obj:`pd.Series`
        The x co-ordinates to use in regression.

    xs : np.ndarray, :obj:`pd.Series`
        The x co-ordinates to use in regression.

    quantiles : list-like
        The quantiles to calculate in each window

    nwindows : positive int
        How many points to evaluate between x_max and x_min.

    decay_length_factor : float
        gives the distance over which the weighting of the values falls to 1/4,
        relative to half the distance between boxes. Defaults to 1. Formula is
        :math:`w = \\left ( 1 + \\left( \\frac{\\text{distance}}{\\text{box_length} \\times \\text{decay_length_factor}} \\right)^2 \\right)^{-1}`.

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
    # We re-form the arrays in case they were pandas series with integer labels that
    # would mess up the sorting.
    xs = np.array(xs)
    ys = np.array(ys)
    sort_order = np.argsort(ys)
    ys = ys[sort_order]
    xs = xs[sort_order]
    if max(xs) == min(xs):
        # We must prevent singularity behaviour if all the points have the same x.
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
        cumsum_weights = np.cumsum(weights)
        for i_quantile in range(quantiles.__len__()):
            quantmatrix.iloc[ind, i_quantile] = min(
                ys[cumsum_weights >= quantiles[i_quantile]]
            )
    return quantmatrix
