# -----------------------------------------------------------------------------
# Copyright (C) 2008-2022 Yunshun Chen, Aaron TL Lun, Davis J McCarthy, Matthew E Ritchie, Belinda Phipson, Yifang Hu, Xiaobei Zhou, Mark D Robinson, Gordon K Smyth
# Copyright (C) 2022-2023 Maximilien Colange

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

# This file is a Python port of the original C++ code from the files
# 'src/nbdev.cpp' and 'src/R_compute_nbdev.cpp' of the Bioconductor edgeR
# package (version 3.38.4).

import numpy as np

from .edgepy_cpp import compute_unit_nb_deviance


def nb_deviance(y, mu, w, phi):
    """
    Calculate the genewise deviance of a negative binomial fit

    See also
    --------
    compute_unit_nb_deviance

    Arguments
    ---------
    y : array_like
        counts matrix
    mu : array_like
        expected means matrix (broadcastable to the shape of :code:`y`)
    w : array_like
        observation weights matrix (broadcastable to the shape of :code:`y`)
    phi : array_like
        dispersion matrix (broadcastable to the shape of :code:`y`)

    Returns
    -------
    ndarray
        matrix of deviances, with as many elements as rows in :code:`y`
    """
    return np.sum(w * compute_unit_nb_deviance(y, mu, phi), axis=1)
