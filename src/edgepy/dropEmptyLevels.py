from .factor import Factor

def dropEmptyLevels(x):
    """
    Drop levels of a factor that does not occur
    """
    if isinstance(x, Factor):
        return Factor(x.arr)
    else:
        return Factor(x)
