import numpy as np
from math import floor

def systematicSubset(n, order_by):
    """
    Take a systematic subset of indices,
    stratified by a ranking variable
    """
    ntotal = len(order_by)
    sampling_ratio = floor(ntotal / n)
    if sampling_ratio <= 1:
        return np.arange(ntotal)
    i1 = floor(sampling_ratio / 2)
    i = np.arange(i1, ntotal, step=sampling_ratio)
    o = np.argsort(order_by)
    return o[i]
