import os

import pandas as pd
import numpy as np


# Divides the x-axis up into nboxes of equal length, assumes that the contents of this box are all positioned
# in the center and calculate quantiles from that.
# The x and y co-ordinates, xs and ys, should be numpy arrays of the same length.
# This function is currently unused and scheduled for deletion.
def aggregate_and_find_quantiles(xs, ys, quantiles, nboxes=10):
    assert xs.size == ys.size
    step = (max(xs) - min(xs)) / nboxes
    boxes = np.arange(min(xs) - 0.5 * step, max(xs) + step, step)
    box_centers = (boxes[1:] + boxes[0:-1]) / 2.0
    quantmatrix = pd.DataFrame(index=box_centers, columns=quantiles)
    for ind in range(box_centers.size):
        ins = (xs < boxes[ind + 1]) & (xs >= boxes[ind])
        if not any(ins):
            quantmatrix.iloc[ind, ] = np.nan
        else:
            quantmatrix.iloc[ind, ] = np.quantile(ys[ins], quantiles)
    return quantmatrix


"""
Divides the x-axis into nboxes of equal length and weights data by how close they are to the center
of these boxes. Then returns the quantiles of this weighted data. Quantiles are defined so that the value returned 
is never interpolated and always equal to a y-value in the data. Extremal points are given their full weighting, 
meaning this will not agree with the np.quantiles under uniform weighting (which effectively gives 0 weight to min and 
max values)

Parameters
----------
xs/ ys: Numpy arrays of the same length, pandas series will also work. 
    The x and y co-ordinates. 
quantiles: list
    The quantiles to calculate
nboxes: positive int
    How many points to evaluate between x_max and x_min. 
decay_length_factor: float
    gives the distance over which the weighting of the values falls to 1/4, 
    relative to half the distance between boxes. Defaults to 1. Formula is
    w = 1/(1+(distance/(box_length*decay_length_factor))^2).
"""
def rolling_window_find_quantiles(xs, ys, quantiles, nboxes=10, decay_length_factor=1):
    assert xs.size == ys.size
    if xs.size == 1:
        return pd.DataFrame(index=[xs[0]] * nboxes, columns=quantiles, data=ys[0])
    step = (max(xs) - min(xs)) / (nboxes + 1)
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
        box_centers = np.arange(min(xs), max(xs) + step*0.99, step)
    quantmatrix = pd.DataFrame(index=box_centers, columns=quantiles)
    for ind in range(box_centers.size):
        weights = 1.0 / (1.0 + ((xs - box_centers[ind]) / decay_length) ** 2)
        weights /= sum(weights)
        # We want to calculate the weights at the midpoint of step corresponding to the y-value.
        cumsum_weights = np.cumsum(weights)
        for i_quantile in range(quantiles.__len__()):
            quantmatrix.iloc[ind, i_quantile] = min(ys[cumsum_weights >= quantiles[i_quantile]])
    return quantmatrix


"""
Checks if a folder already exists for the filepath entered and creates it if it does not. 
Parameters
----------
save_path: string
     The directory or a file in the directory that we wish to ensure exists. If a directory, it must include a / 
     at the end. 
"""
def ensure_savepath(save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))


"""
Finds what quantile a time-series corresponds to given a time series of scattered data. Uses the equal-weighting 
aggregate_and_find_quantiles definition of what a quantile is, with the box width determined by the spacing of the 
new data
----------
orig_xs :  
    the x position of the larger data source. This will usually be the time axis. It must be sorted in ascending order
"""
def which_quantile(orig_xs, orig_ys, new_xs, new_ys):
    if len(np.unique(new_xs)) != len(new_xs):
        raise ValueError(
            "Some x-values are repeated in the quantile dataset: {}".format(new_xs)
        )
    if len(orig_ys) != len(orig_xs) or len(new_xs) != len(new_ys):
        raise ValueError(
            "Unequal data inserted into quantile function"
        )

    sort_xs = np.argsort(new_xs)
    new_xs = new_xs[sort_xs]
    new_ys = new_ys[sort_xs]


    if len(new_xs) > 1:
        # Divide the x-axis into boxes centered around the new_xs
        boxes = 0.5 * (new_xs[1:] + new_xs[:-1])
        boxes = np.insert(boxes, 0, new_xs[0] - 0.5 * (new_xs[1] - new_xs[0]))
        boxes = np.append(boxes, new_xs[-1]+0.5*(new_xs[-1]-new_xs[-2]))
    elif len(new_xs) == 1:
        # There is no natural lengthscale, so we just use +/- 1
        boxes = np.insert(new_xs-1, 1, new_xs+1)

    quantile_of_xs = np.array(np.nan * new_xs)
    for ind in range(quantile_of_xs.size):
        ins = (orig_xs < boxes[ind + 1]) & (orig_xs >= boxes[ind])
        if not any(ins):
            quantile_of_xs[ind] = np.nan
        else:
            quantile_of_xs[ind] = sum(new_ys[ind] > orig_ys[ins]) / len(orig_ys[ins])
    return quantile_of_xs
