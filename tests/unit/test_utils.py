import os
import silicone.utils as utils
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
    assert all(quantiles.iloc[0] == [0, 0, 1])
    assert all(quantiles.iloc[1] == [0, 0, 1])

    xs = np.array([0, 0, 1, 1])
    ys = np.array([0, 0, 1, 1])
    quantiles = utils.rolling_window_find_quantiles(xs, ys, desired_quantiles, 9, 2*9)
    assert all(quantiles.iloc[0, :] == 0)
    assert all(quantiles.iloc[-1, :] == 1)
    assert all(quantiles.iloc[5, :] == [0, 0, 1])

    desired_quantiles = [0, 0.5, 1]
    quantiles = utils.rolling_window_find_quantiles(np.array([1]), np.array([1]), desired_quantiles, 9, 2 * 9)
    assert all(quantiles.iloc[0, :] == [1, 1, 1])

    desired_quantiles = [0, 0.5, 1]
    quantiles = utils.rolling_window_find_quantiles(np.array([1, 1]), np.array([1, 1]), desired_quantiles, 9, 2 * 9)
    assert all(quantiles.iloc[0, :] == [1, 1, 1])

def test_ensure_savepath_folder():
    new_folder = './folder_should_not_exist/'
    if os.path.exists(new_folder):
        raise Exception('Folder that should have been deleted from a previous test still exists')
    utils.ensure_savepath(new_folder)
    if os.path.exists(new_folder):
        os.rmdir(os.path.dirname(new_folder))
    else:
        raise Exception('Folder not created properly')

def test_ensure_savepath_file():
    new_file = './folder_should_not_exist/file_should_not_exist.txt'
    if os.path.exists(os.path.dirname(new_file)):
        raise Exception('Folder that should have been deleted from a previous test still exists')
    utils.ensure_savepath(new_file)
    if os.path.exists(os.path.dirname(new_file)):
        os.rmdir(os.path.dirname(new_file))
    else:
        raise Exception('Folder not created properly')
