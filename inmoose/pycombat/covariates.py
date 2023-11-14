# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------

import logging
import numpy as np
import pandas as pd
from patsy import dmatrix, DesignMatrix

from ..utils import asfactor


def make_design_matrix(
    counts, batch, covar_mod=None, ref_batch=None, cov_missing_value=None
):
    """Make design matrix for batch effect correction. Handles covariates as
    well as reference batch.

    Arguments
    ---------
    counts : matrix
        raw count matrix from genomic studies (dimensions gene x sample)
    batch : array or list or :obj:`inmoose.utils.factor.Factor`
        batch indices
    covar_mod : list or matrix, optional
        model matrix (dataframe, list or numpy array) for one or multiple covariates to include in linear model (signal
        from these variables are kept in data after adjustment). Covariates have to be categorial,
        they can not be continious values (default: `None`).
    ref_batch : any
        batch id of the batch to use as reference. Must be one of the element of
        `batch` (default: `None`).
    cov_missing_value : str
        Option to choose the way to handle missing covariates
        `None` raise an error if missing covariates and stop the code
        `remove` remove samples with missing covariates and raise a warning
        `fill` handle missing covariates, by creating a distinct covariate per batch
        (default: `None`)

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
    batch : array or list or :obj:`inmoose.utils.factor.Factor`
        batch indices
    remove_sample : list
        sample indices to remove from the analysis
    """
    # covariate
    remove_sample = []
    if covar_mod is not None:
        # if needed, format covariate DesignMatrix
        if type(covar_mod) != DesignMatrix:
            remove_sample, covar_mod = format_covar_mod(
                covar_mod, batch, cov_missing_value
            )
            batch = [el for i, el in enumerate(batch) if i not in remove_sample]
        # bind with biological condition of interest
        mod = covar_mod
    else:
        mod = dmatrix("~1", pd.DataFrame(counts.T))

    # preparation
    batch = asfactor(batch)
    # batch
    batchmod = dmatrix("~0 + C(batch)")
    # combine
    design = dmatrix("~ 0 + batchmod + mod")

    # number of batches
    n_batch = batch.nlevels()
    # list of samples in each batch
    batches_ind = [(batch == batch.categories[i]).nonzero()[0] for i in range(n_batch)]
    n_batches = [len(i) for i in batches_ind]
    n_sample = np.sum(n_batches)
    logging.info(f"Found {n_batch} batches")

    if 1 in n_batches:
        logging.warnings.warn("Single-sample batch detected!")

    # reference batch
    if ref_batch is not None:
        if ref_batch not in batch.categories:
            raise ValueError("Reference batch must identify one of the batches")
        logging.info(f"Using batch {ref_batch} as reference batch")
        # ref_batch_idx is the index of the reference batch in batch.categories
        ref_batch_idx = np.where(batch.categories == ref_batch)[0][0]
        # update batchmod with reference
        batchmod[:, ref_batch_idx] = 1
    else:
        ref_batch_idx = None

    # Check for intercept in covariates, and drop if present
    check = [(design[:, i] == 1).all() for i in range(design.shape[1])]
    if ref_batch_idx is not None:
        # the reference batch is not considered as a covariate
        check[ref_batch_idx] = False
    design = design[:, np.logical_not(check)]

    logging.info(
        f"Adjusting for {design.shape[1] - batchmod.shape[1]} covariate(s) or covariate level(s)"
    )

    # Check if the desigin is confounded
    check_confounded_covariates(design, n_batch)

    return (
        design,
        batchmod,
        mod,
        batches_ind,
        n_batches,
        n_batch,
        n_sample,
        ref_batch_idx,
        batch,
        remove_sample,
    )


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


def format_covar_mod(covar_mod, batch, cov_missing_value=None):
    """Format the covariate table in model matrix.

    Arguments
    ---------
    covar_mod : list or matrix
        model matrix (dataframe, list or numpy array) contaning one or multiple covariates
    batch : array or list or :obj:`inmoose.utils.factor.Factor`
        batch indices
    cov_missing_value : str
        Option to choose the way to handle missing covariates
        `None` raise an error if missing covariates and stop the code
        "remove" remove samples with missing covariates and raise a warning
        "fill" handle missing covariates, by creating a distinct covariate per batch
        (default: "raise")

    Returns
    -------
    remove_sample : list
        sample indices to remove from the analysis
    covar_mod : dataframe
        model matrix (dataframe) contaning covariates in interger format
    """
    if (type(covar_mod) == list) | (type(covar_mod) == np.ndarray):
        covar_mod = pd.DataFrame(covar_mod)
        if covar_mod.shape[0] != len(batch):
            if covar_mod.shape[1] == len(batch):
                covar_mod = covar_mod.T
            else:
                raise ValueError(
                    "The covar_mod matrix parameter doesn't match the number of sample"
                )
        covar_mod.columns = ["col_" + str(col_nb) for col_nb in covar_mod.columns]
    elif type(covar_mod) != pd.DataFrame:
        raise ValueError(
            f"The covar_mod parameter type {type(covar_mod)} is not accepted, it must be list, numpy.array or pandas.dataframe."
        )

    # check for numeric type (excluding nan) to identify potential continuous_variable
    continuous_variable = []
    potential_continuous_variable = []
    for col in covar_mod.columns:
        # Return True if numeric type with decimal
        # Return False if numeric type without decimal
        col_data_type = [
            (ele % 1) > 0
            for ele in set(covar_mod[col].dropna())
            if (type(ele) == float) | (type(ele) == int) | (type(ele) == complex)
        ]
        if True in col_data_type:
            continuous_variable.append(col)
        elif False in col_data_type:
            potential_continuous_variable.append(col)

    if len(continuous_variable) > 0:
        raise ValueError(
            f"Found numerical covariates with decimal {', '.join(continuous_variable)} in covar_mod parameters. Numerical covariates are not accepted. Please remove them before proceeding with pycombat."
        )
    elif len(potential_continuous_variable) > 0:
        logging.warnings.warn(
            f"Found intereger covariates {', '.join(potential_continuous_variable)} in covar_mod parameters. Numerical covariates are not accepted, these covariates will be process as categorial. You may want to double check your covariates."
        )

    # check for nan in categorial covariates
    remove_sample = []
    nan_covar_mod = covar_mod.isna()
    if nan_covar_mod.any().any():
        if cov_missing_value is None:
            name_nan_covar_mod = nan_covar_mod.loc[
                :, nan_covar_mod.any() == True
            ].columns
            raise ValueError(
                f"{nan_covar_mod.sum().sum()} values are missing in covariates {', '.join(name_nan_covar_mod)}. Correct your covariates or use the cov_missing_value parameters"
            )
        elif cov_missing_value == "remove":
            logging.warnings.warn(
                f"{(nan_covar_mod.sum(axis=1)>0).sum()} samples with missing covariates in covar_mod. They are removed from the data. You may want to double check your covariates."
            )
            remove_sample = [
                i for i, x in enumerate(nan_covar_mod.sum(axis=1) > 0) if x == True
            ]
        elif cov_missing_value == "fill":
            logging.warnings.warn(
                f"{nan_covar_mod.sum().sum()} missing covariates in covar_mod. Creating a distinct covariate per batch for the missing values. You may want to double check your covariates."
            )
            # handle missing covariates, by creating a distinct covariate per batch
            # where a missing covariate appears
            for col in covar_mod.columns:
                nan_cov_col = covar_mod[col].isna()
                nan_batch_group = [
                    f"nan_batch_{batch[i]}"
                    for i in range(len(covar_mod[col]))
                    if nan_cov_col[i]
                ]
                for i, j in enumerate(np.where(nan_cov_col)[0]):
                    covar_mod.loc[j, col] = nan_batch_group[i]
        else:
            raise ValueError(
                f"cov_missing_value parameter doesn't accept {cov_missing_value} value, only `None`, `fill` or `remove` are valid"
            )

    covar_mod = dmatrix(
        "+".join([f"C({cv})" for cv in covar_mod.columns]), data=covar_mod
    )
    return remove_sample, covar_mod


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
        if design.shape[1] == n_batch + 1:  # case 1: covariate confounded with a batch
            raise ConfoundingVariablesError(
                "Covariate is confounded with batch. Try removing the covariates."
            )
        if (
            design.shape[1] > n_batch + 1
        ):  # case 2: multiple covariates confounded with a batch
            if np.linalg.matrix_rank(design.T[:n_batch]) < design.shape[1]:
                raise ConfoundingVariablesError(
                    "Confounded design. Try removing one or more covariates."
                )
            else:  # case 3: at least one covariate confounded with a batch
                raise ConfoundingVariablesError(
                    "At least one covariate is confounded with batch. Try removing confounded covariates."
                )
