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


def test_rolling_window_find_quantiles_same_points():
    # If all the x-values are the same, this should just be our interpretation of
    # quantiles at all points
    xs = np.array([1] * 11)
    ys = np.array(range(11))
    desired_quantiles = [0, 0.4, 0.5, 0.6, 0.85, 1]
    quantiles = stats.rolling_window_find_quantiles(xs, ys, desired_quantiles, 9, 2 * 9)

    cumsum_weights = (1 + np.arange(11)) / 11
    calculated_quantiles = []
    for quant in desired_quantiles:
        calculated_quantiles.append(min(ys[cumsum_weights >= quant]))

    assert all(quantiles.values.squeeze() == calculated_quantiles)


def test_geometric_mean():
    x_array = [1, 1, 2, 4, 4]
    res = stats.geometric_mean(x_array)
    expected = (2 * 4 * 4) ** (1 / 5)
    assert abs(res - expected) < 1e-15
    res2 = stats.geometric_mean(np.array(x_array))
    assert res2 == res


def test_geometric_mean_breaks():
    x_array = [1, 1, 2, 4, np.nan]
    res = stats.geometric_mean(x_array)
    assert np.isnan(res)


def test_geometric_mean_overflow():
    xs = [1000] * 1000000
    res = stats.geometric_mean(xs)
    assert abs(res - 1000) < 1e-10
