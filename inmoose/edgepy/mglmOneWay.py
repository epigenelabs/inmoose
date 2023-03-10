#-----------------------------------------------------------------------------
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
#-----------------------------------------------------------------------------

# This file is based on the file 'R/mglmOneWay.R' of the Bioconductor edgeR package (version 3.38.4).


import numpy as np
from ..utils import Factor, asfactor
from .makeCompressedMatrix import _compressOffsets, _compressDispersions, _compressWeights
from .mglmOneGroup import mglmOneGroup
from .edgepy_cpp import cxx_get_one_way_fitted
from scipy.linalg import solve

def designAsFactor(design):
    """
    Construct a factor from the unique rows of a matrix
    """
    design = np.asarray(design, order='F')
    z = (np.e + np.pi) / 5
    col = np.full(design.shape, np.arange(design.shape[1]), order='F')
    means = (design * z**col).mean(axis=1)
    uniq = np.unique(means)
    fact = np.zeros(means.shape, dtype='int64', order='F')
    for i in range(len(uniq)):
        fact[means == uniq[i]] = i+1
    return Factor(fact)

def mglmOneWay(y, design=None, group=None, dispersion=0, offset=0, weights=None, coef_start=None, maxit=50, tol=1e-10):
    """
    Fit multiple negative binomial GLMs with log link by Fisher scoring with a single explanatory factor in the model
    """
    y = np.asarray(y, order='F')
    (ngenes, nlibs) = y.shape

    offset = _compressOffsets(y, offset=offset)
    dispersion = _compressDispersions(y, dispersion)
    weights = _compressWeights(y, weights)

    # If necessary, the group factor is computed from the design matrix.
    # However, if group is supplied, we can avoid creating a design matrix altogether.
    if group is None:
        if design is None:
            group = Factor(np.ones((nlibs,)))
        else:
            design = np.asarray(design, order='F')
            group = designAsFactor(design)
    else:
        group = asfactor(group)

    # Convert factor to integer levels for efficiency
    levg = group.categories
    ngroups = len(levg)
    i = group.__array__()

    if design is not None:
        if design.shape[1] != ngroups:
            raise ValueError("design matrix is not equivalent to a oneway layout")

    # Reduce to representative design matrix, based on the column in which each group appears first
    firstjofgroup = [(i == x).nonzero()[0][0] for x in levg]
    if design is not None:
        designunique = design[firstjofgroup,:]
    else:
        designunique = None

    # Is it just a group indicator matrix?
    if np.sum(designunique == 1) == ngroups and np.sum(designunique == 0) == (ngroups-1)*ngroups:
        design = None

    # If necessary, convert starting values to group fitted values
    if design is not None and coef_start is not None:
        coef_start = coef_start @ designunique.T

    # Cycle through groups
    beta = np.zeros((ngenes, ngroups), order='F', dtype='double')
    for g in range(ngroups):
        j = np.nonzero(i == (g+1))[0]
        beta[:, g] = mglmOneGroup(y[:,j], dispersion=dispersion[:,j], offset=offset[:,j], weights=weights[:,j] if weights is not None else None, coef_start=coef_start[:,g] if coef_start is not None else None, maxit=maxit, tol=tol)

    # Reset -inf values to finite values to simplify calculations downstream
    beta = np.where(beta > -1e8, beta, -1e8)

    # Fitted values from group-wise beta's
    mu = cxx_get_one_way_fitted(beta, offset, i-1)

    # If necessary, reformat the beta's to reflect the original design.
    if design is not None:
        beta = solve(designunique, beta.T).T

    return (beta, mu)

