import numpy as np
from .makeCompressedMatrix import _compressOffsets, _compressPrior, makeCompressedMatrix
from edgepy_cpp import cxx_add_prior_count

def addPriorCount(y, lib_size=None, offset=None, prior_count=1):
    """
    Add library size-adjusted prior counts to values of y.
    Also add twice the adjusted prior to th library sizes,
    which are provided as log-transformed values in `offset`.
    """
    # Check y
    y = np.asarray(y, order='F')
    if not np.issubdtype(y.dtype, np.number):
        raise ValueError("count matrix must be numeric")

    # Check prior_count
    prior_count = _compressPrior(y, prior_count)

    # Check lib_size and offset
    # If offsets are provided, they must have a similar average to log(lib_size)
    # for the results to be meaningful as logCPM values
    offset = _compressOffsets(y, lib_size=lib_size, offset=offset)

    # Adding the prior count
    (out_y, out_offset) = cxx_add_prior_count(y, offset, prior_count)
    out_offset = makeCompressedMatrix(out_offset, y.shape, byrow=True)
    return (out_y, out_offset)

