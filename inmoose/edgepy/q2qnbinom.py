# -----------------------------------------------------------------------------
# Copyright (C) 2008-2022 Yunshun Chen, Aaron TL Lun, Davis J McCarthy, Matthew E Ritchie, Belinda Phipson, Yifang Hu, Xiaobei Zhou, Mark D Robinson, Gordon K Smyth
# Copyright (C) 2024 Maximilien Colange

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------

# This file is based on the file 'R/q2qnbinom.R' of the Bioconductor edgeR package (version 3.38.4).

from .edgepy_cpp import _q2qnbinom


# the actual implementation in done in Cython
# this Python wrapper allows to use optional and named arguments
def q2qnbinom(x, input_mean, output_mean, dispersion=0):
    """
    Interpolated quantile to quantile mapping between negative-binomial distributions with the same dispersion but different means.

    This function finds the quantile with the same left and right tail
    probabilities relative to the output mean as :code:`x` has relative to the
    input mean. In principle, :func:`q2qnbinom` gives similar results to calling
    :code:`pnbinom` followed by :code:`qnbinom` as in the example below.
    However this function avoids infinite values arising from rounding errors
    and does appropriate interpolation to return continuous values.

    It is called by :func:`equalizeLibSizes` to perform quantile-to-quantile
    normalization.

    See also
    --------
    equalizeLibSizes

    Arguments
    ---------
    x : array_like
        matrix of counts
    input_mean : array_like
        matrix of population means for :code:`x`. If 1-D, then of the same
        length as :code:`x.shape[0]`
    output_mean : array_like
        matrix of population means for the output values. If 1-D, then of the
        same length as :code:`x.shape[0]`
    dispersion : array_like
        scalar, vector or matrix giving negative binomial dispersion values

    Returns
    -------
    ndarray
        matrix of same dimensions as :code:`x`, with :code:`output_mean` as the
        new nominal population mean
    """

    return _q2qnbinom(x, input_mean, output_mean, dispersion)
