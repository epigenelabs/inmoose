#-----------------------------------------------------------------------------
# Copyright (C) 2019-2023 A. Behdenna, A. Nordor, J. Haziza, A. Gema and M. Colange

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

import logging
import numpy as np
import pandas as pd
from patsy import dmatrix

from ..utils import asfactor

def make_design_matrix(counts, batch, group, covar_mod, full_mod, ref_batch):
    """Make design matrix for batch effect correction. Handles covariates as
    well as reference batch.

    Arguments
    ---------
    counts : matrix
        raw count matrix from genomic studies (dimensions gene x sample)
    batch : array or list or :obj:`inmoose.utils.factor.Factor`
        batch indices
    group : array or list or :obj:`inmoose.utils.factor.Factor`
        vector/factor for biological condition of interest
    covar_mod : matrix
        model matrix for multiple covariates to include in linear model (signal
        from these variables are kept in data after adjustment)
    full_mod : bool
        if True, include condition of interest in model
    ref_batch : any
        batch id of the batch to use as reference. Must be one of the element of
        `batch`

    Returns
    -------
    matrix
        the design matrix
    matrix
        the batch-only design matrix (i.e. without covariates)
    matrix
        the covariate-only design matrix (i.e. without batches)
    list of list of int
        for each batch, the indices of the samples of this batch
    list of int
        the size of each batch (in number of samples)
    int
        the number of batches
    int
        the number of samples
    int or None
        the index of the reference batch if any, otherwise None
    """

    # preparation
    batch = asfactor(batch)

    # number of batches
    n_batch = batch.nlevels()
    # list of samples in each batch
    batches_ind = [(batch == batch.categories[i]).nonzero()[0]
                   for i in range(n_batch)]
    n_batches = [len(i) for i in batches_ind]
    n_sample = np.sum(n_batches)
    logging.info(f"Found {n_batch} batches")

    if 1 in n_batches:
        logging.warnings.warn("Single-sample batch detected!")

    # batch
    batchmod = dmatrix("~0 + C(batch)")
    # reference batch
    if ref_batch is not None:
        if ref_batch not in batch.categories:
            raise ValueError("Reference batch must identify one of the batches")
        logging.info(f"Using batch {ref_batch} as reference batch")
        # ref_batch_idx is the index of the reference batch in batch.categories
        ref_batch_idx = np.where(batch.categories == ref_batch)[0][0]
        # update batchmod with reference
        batchmod[:,ref_batch_idx] = 1
    else:
        ref_batch_idx = None

    # covariate
    group = asfactor([] if group is None else group)
    # handle missing covariates, by creating a distinct covariate per batch
    # where a missing covariate appears
    nan_group = group.isna()
    if nan_group.any():
        logging.warnings.warn(f"{nan_group.sum()} missing covariates in group. Creating a distinct covariate per batch for the missing values. You may want to double check your covariates.")
        nan_batch_group = [f"nan_batch_{batch[i]}"
                           for i in range(len(group))
                           if nan_group[i]]
        group = group.add_categories(np.unique(nan_batch_group))
        for i,j in enumerate(np.where(nan_group)[0]):
            group[j] = nan_batch_group[i]

    if full_mod and group.nlevels() > 1:
        logging.info("Using full model")
        mod = dmatrix("~group")
    else:
        logging.info("Using null model")
        mod = dmatrix("~1", pd.DataFrame(counts.T))
    # drop intercept in covariate model
    if covar_mod is not None:
        check = [(covar_mod[:,i] == 1).all() for i in range(covar_mod.shape[1])]
        covar_mod = covar_mod[:, np.logical_not(check)]
        # bind with biological condition of interest
        mod = np.concatenate((mod, covar_mod), axis=1)
    # combine
    design = dmatrix("~ 0 + batchmod + mod")

    # Check for intercept in covariates, and drop if present
    check = [(design[:,i] == 1).all() for i in range(design.shape[1])]
    if ref_batch_idx is not None:
        # the reference batch is not considered as a covariate
        check[ref_batch_idx] = False
    design = design[:, np.logical_not(check)]

    logging.info(f"Adjusting for {design.shape[1] - batchmod.shape[1]} covariate(s) or covariate level(s)")

    # Check if the desigin is confounded
    check_confounded_covariates(design, n_batch)

    return design, batchmod, mod, batches_ind, n_batches, n_batch, n_sample, ref_batch_idx


class ConfoundingVariablesError(Exception):
    """Exception raised when confounding variables are detected.

    Arguments
    ---------
    message : str
        explanation of the error
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def check_confounded_covariates(design, n_batch):
    """Detect confounded covariates.
    This function returns nothing, but raises exception if confounded covariates are detected.

    Arguments
    ---------
    design : matrix
        the design matrix
    n_batch : int
        the number of batches
    """

    # if matrix is not invertible, different cases
    if np.linalg.matrix_rank(design) < design.shape[1]:
        if design.shape[1] == n_batch+1: # case 1: covariate confounded with a batch
            raise ConfoundingVariablesError("Covariate is confounded with batch. Try removing the covariates.")
        if design.shape[1] > n_batch+1: # case 2: multiple covariates confounded with a batch
            if np.linalg.matrix_rank(design.T[:n_batch]) < design.shape[1]:
                raise ConfoundingVariablesError("Confounded design. Try removing one or more covariates.")
            else: # case 3: at least one covariate confounded with a batch
                raise ConfoundingVariablesError("At least one covariate is confounded with batch. Try removing confounded covariates.")
