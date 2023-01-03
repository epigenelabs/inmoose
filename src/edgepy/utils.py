import numpy as np

def _isAllZero(y):
    """
    Check whether all counts are zero.
    Also check and stop with an informative message if negative, NaN or infinite counts are present.
    """
    if len(y) == 0:
        return False
    check_range = (np.amin(y), np.nanmax(y))
    if np.isnan(check_range[0]):
        raise ValueError("NaN counts are not allowed")
    if check_range[0] < 0:
        raise ValueError("negative counts are not allowed")
    if np.isinf(check_range[1]):
        raise ValueError("infinite counts are not allowed")
    return check_range[1] == 0

