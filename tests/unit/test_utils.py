import src.silicone.utils as utils
import numpy as np
import pandas as pd

def test_aggregate_and_find_quantiles():
    xs = np.array([0, 0, 1])
    ys = np.array([0, 1, 1])
    desired_quantiles = [0.2, 0.5, 0.8]
    quantiles = utils.aggregate_and_find_quantiles(xs, ys, desired_quantiles)
    assert all(quantiles.iloc[0] == desired_quantiles)
    assert all(pd.isna(quantiles.iloc[1]))
    assert quantiles.iloc[-1, -1] == 1

def test_rolling_window_find_quantiles():

    xs = np.array([0, 0, 1, 1])
    ys = np.array([0, 1, 0, 1])
    desired_quantiles = [0.4, 0.5, 0.6]
    quantiles = utils.rolling_window_find_quantiles(xs, ys, desired_quantiles, 9, 2*9)
    assert all(quantiles.iloc[0] == [0, 0.5, 1])
    assert all(quantiles.iloc[1] == [0, 0.5, 1])

    xs = np.array([0, 0, 1, 1])
    ys = np.array([0, 0, 1, 1])
    quantiles = utils.rolling_window_find_quantiles(xs, ys, desired_quantiles, 9, 2*9)
    assert all(quantiles.iloc[0, :] == 0)
    assert all(quantiles.iloc[-1, :] == 1)
    assert all(quantiles.iloc[5, :] == [0, 0.5, 1])


test_aggregate_and_find_quantiles()
test_rolling_window_find_quantiles()
