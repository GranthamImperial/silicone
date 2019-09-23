import pandas as pd
import numpy as np

# Divides the x-axis up into nboxes of equal length, assumes that the contents of this box are all positioned
# in the center and calculate quantiles from that.
# The x and y co-ordinates, xs and ys, should be numpy arrays of the same length.
def aggregate_and_find_quantiles(xs, ys, quantiles=[0.2, 0.33, 0.5, 0.66, 0.8], nboxes=10):
    assert xs.size == ys.size
    step = (max(xs)-min(xs))/nboxes
    boxes = np.arange(min(xs)-0.5*step, max(xs)+step, step)
    box_centers = (boxes[1:]+boxes[0:-1])/2.0
    quantmatrix = pd.DataFrame(index=box_centers, columns=quantiles)
    for ind in range(box_centers.size):
        ins = (xs < boxes[ind+1]) & (xs >= boxes[ind])
        if not any(ins):
            quantmatrix.iloc[ind, ] = np.nan
        else:
            quantmatrix.iloc[ind, ] = np.quantile(ys[ins], quantiles)
    return quantmatrix


# Divides the x-axis into nboxes of equal length and weights data by how close they are to the center
# of these boxes. Then finds quantiles of this weighted data.
# The x and y co-ordinates, xs and ys, should be numpy arrays of the same length.
# decay_length gives the distance over which the weighting of the values falls to 1/2, given by equation
# w = 1/(1+(distance/decay_length)^2). This defaults to the interbox distance if no argument is given.
# Note that this uses a different definition of quantiles to the np.quantile function, where the uppermost and
# lowermost points are 0/1. Here we identify halfway the quantile as equalling the value halfway through a weight
# cumulative sum, or linear interpolation between these values if there is none.
def rolling_window_find_quantiles(xs, ys, quantiles=[0.2, 0.33, 0.5, 0.66, 0.8], nboxes=10, decay_length=None):
    assert xs.size == ys.size
    step = (max(xs) - min(xs)) / (nboxes+1)
    if decay_length is None:
        decay_length = step
    sort_order = np.argsort(ys)
    ys = ys[sort_order]
    xs = xs[sort_order]
    box_centers = np.arange(min(xs), max(xs)+step, step)
    quantmatrix = pd.DataFrame(index=box_centers, columns=quantiles)
    for ind in range(box_centers.size):
        weights = 1.0/(1.0+((xs-box_centers[ind])/decay_length)**2)
        # We want to calculate the weights at the midpoint of step corresponding to the y-value.
        cumsum_weights = np.cumsum(weights)-0.5*weights-0.5*weights[0]
        cumsum_weights /= cumsum_weights[-1]
        quantmatrix.iloc[ind, ] = np.interp(quantiles, cumsum_weights, ys)
    return quantmatrix

