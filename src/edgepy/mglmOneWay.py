import numpy as np
from .factor import Factor, asfactor
from .makeCompressedMatrix import _compressOffsets, _compressDispersions, _compressWeights
from .mglmOneGroup import mglmOneGroup
from edgepy_cpp import cxx_get_one_way_fitted
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

