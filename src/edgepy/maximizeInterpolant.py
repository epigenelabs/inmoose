import numpy as np

from edgepy_cpp import cxx_maximize_interpolant

def maximizeInterpolant(x,y):
    """
    This function takes an ordered set of spline points and a likelihood matrix where each row corresponds to a tag and each column corresponds to a spline point.
    It then calculates the position at which the maximum interpolated likelihood occurs for each by solving the derivative of the spline function.
    """
    x = np.asarray(x, order='F', dtype='double')
    y = np.asarray(y, order='F', dtype='double')
    if len(y.shape) != 2:
        raise ValueError("y is not a matrix: cannot perform interpolation")
    if len(x) != y.shape[1]:
        raise ValueError("number of columns must equal number of spline points")
    if not np.array_equal(np.unique(x), x):
        raise ValueError("spline points must be unique and sorted")

    # Performing some type checking
    out = cxx_maximize_interpolant(x, y)
    return out
