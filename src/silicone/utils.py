import pandas as pd
import numpy as np
from skmisc.loess import loess

# A function to return the quantiles of the data that fit into different boxes.
# Overshoot: the distance beyond the real limits of x to include in our plots, assumed 1
def linearfit_and_find_quantiles(xs, ys, quantiles=[0.2, 0.33, 0.5, 0.66, 0.8], overshoot=0.1):
    newxs = range(min(xs) - overshoot * (max(xs) - min(xs)), max(xs) + overshoot * (max(xs) - min(xs)))
    quantmatrix = pd.DataFrame(index=newxs, columns=quantiles)
    for x in newxs:
        y=x
        #TODO: write this function after clarification of the problem with R code

    return quantmatrix

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