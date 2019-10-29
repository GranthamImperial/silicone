import numpy as np

import silicone.stats as stats


def test_rolling_window_find_quantiles():
    xs = np.array([0, 0, 1, 1])
    ys = np.array([0, 1, 0, 1])
    desired_quantiles = [0.4, 0.5, 0.6]
    quantiles = stats.rolling_window_find_quantiles(xs, ys, desired_quantiles, 9, 2 * 9)
    assert all(quantiles.iloc[0] == [0, 0, 1])
    assert all(quantiles.iloc[1] == [0, 0, 1])

    xs = np.array([0, 0, 1, 1])
    ys = np.array([0, 0, 1, 1])
    quantiles = stats.rolling_window_find_quantiles(xs, ys, desired_quantiles, 9, 2 * 9)
    assert all(quantiles.iloc[0, :] == 0)
    assert all(quantiles.iloc[-1, :] == 1)
    assert all(quantiles.iloc[5, :] == [0, 0, 1])

    desired_quantiles = [0, 0.5, 1]
    quantiles = stats.rolling_window_find_quantiles(
        np.array([1]), np.array([1]), desired_quantiles, 9, 2 * 9
    )
    assert all(quantiles.iloc[0, :] == [1, 1, 1])

    desired_quantiles = [0, 0.5, 1]
    quantiles = stats.rolling_window_find_quantiles(
        np.array([1, 1]), np.array([1, 1]), desired_quantiles, 9, 2 * 9
    )
    assert all(quantiles.iloc[0, :] == [1, 1, 1])


def test_rolling_window_quantiles():
    assert False, "expand tests here and use parametrise"
