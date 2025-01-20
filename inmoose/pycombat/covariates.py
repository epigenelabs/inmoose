# -----------------------------------------------------------------------------
# Copyright (C) 2019-2024 A. Behdenna, A. Nordor, J. Haziza, A. Gema, M. Colange, L. Meunier

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

import numpy as np
import pandas as pd
from anndata import AnnData
from patsy import DesignMatrix, dmatrix

from ..utils import LOGGER, asfactor


class VirtualCohortInput:
    """
    Attributes
    ----------
    counts : matrix
        raw count matrix from genomic studies (dimensions gene x sample)
    batch : :class:`~inmoose.utils.factor.Factor`
        batch identifiers
    nan_batch : bool
        whether :code:`batch` contains :code:`NA` values
    nan_genes : int
        number of genes with a :code:`NA` value in the :code:`counts` matrix
    nan_samples : int
        number of samples with a :code:`NA` value in the :code:`counts` matrix
    batch_composition : dict[batch_id, list[int]]
        for each batch, the indices of the samples of this batch
    n_batch : int
        number of batches
    ref_batch_idx : int or None
        index (in :code:`batch.categories`) of the reference batch
    nan_cov : int
        number of covariates containing a :code:`NA` value
    nan_cov_samples : dict[batch_id, int]
        number of samples with a :code:`NA` covariate per batch
    confounded_cov : Option[list]
        list of indices of confounded variables
        :code:`None` if some covariates contain :code:`NA` values, because
        confounded covariates cannot be determined
    batch_mod : :class:`patsy.DesignMatrix`
        the batch-only design matrix (i.e. without covariates)
        :code:`None` if some batch is :code:`NA`
    covar_mod : :class:`patsy.DesignMatrix` or :class:`pd.DataFrame`
        the covariate-only design matrix (i.e. without batches)
        as a :class:`pd.DataFrame` only if some covariates contain :code:`NA` values
    design : :class:`patsy.DesignMatrix`
        the design matrix (combining batches and covariates)
        :code:`None` if some covariates contain :code:`NA` values or if some
        batch is :code:`NA`
    """

    def __init__(self, counts, batch, covar_mod=None, ref_batch=None):
        """
        Arguments
        ---------
        counts : ndarray or pd.DataFrame or AnnData
            raw count matrix from genomic studies (dimensions gene x sample)
        batch : array or list or :class:`~inmoose.utils.factor.Factor` or str
            batch identifiers
            if :code:`counts` is an AnnData, then :code:`batch` can be the name
            of the obs column containing the batch information
        covar_mod : list or matrix, optional
            model matrix (dataframe, list or numpy array) for one or multiple
            covariates to include in linear model (signal from these variables
            are kept in data after adjustment). Covariates have to be
            categorial, they can not be continious values (default: :code:`None`).
        ref_batch : any
            batch id of the batch to use as reference. Must be one of the element of
            :code:`batch` (default: :code:`None`).
        """
        if isinstance(counts, pd.DataFrame):
            list_samples = counts.columns
            list_genes = counts.index
            counts = counts.values
            input_type = "dataframe"
        elif isinstance(counts, np.ndarray):
            list_samples = None
            list_genes = None
            input_type = "ndarray"
        elif isinstance(counts, AnnData):
            list_samples = counts.obs_names
            list_genes = counts.var_names
            self.input_ad = counts
            counts = counts.X.T
            input_type = "anndata"
            if isinstance(batch, str):
                try:
                    batch = self.input_ad.obs[batch]
                except KeyError:
                    raise ValueError(
                        f"the batch column \"{batch}\" must appear in 'counts.obs'"
                    )
        else:
            raise ValueError("counts must be a pandas DataFrame or a numpy nd array")

        self.input_type = input_type
        self.list_samples = list_samples
        self.list_genes = list_genes
        self.counts = counts
        self.batch = asfactor(batch)
        self.nan_batch = self.batch.isna().any()

        nancounts = np.isnan(counts)
        self.nan_genes = nancounts.any(axis=1).sum()
        self.nan_samples = nancounts.any(axis=0).sum()

        # number of batches
        self.n_batch = self.batch.nlevels()
        # number of samples in each batch
        self.batch_composition = {
            b: np.where(self.batch == b)[0] for b in self.batch.categories
        }

        if self.nan_batch:
            self.batch_mod = None
        else:
            # batch design matrix
            self.batch_mod = dmatrix("~0 + C(self.batch)")

        # reference batch
        if ref_batch is not None:
            if ref_batch not in self.batch.categories:
                raise ValueError("Reference batch must identify one of the batches")
            # ref_batch_idx is the index of the reference batch in batch.categories
            self.ref_batch_idx = np.where(self.batch.categories == ref_batch)[0][0]
            # update batchmod with reference
            self.batch_mod[:, self.ref_batch_idx] = 1
        else:
            self.ref_batch_idx = None

        if covar_mod is not None and not isinstance(covar_mod, DesignMatrix):
            # convert to dataframe
            if isinstance(covar_mod, (list, np.ndarray)):
                covar_mod = pd.DataFrame(covar_mod)
            elif type(covar_mod) is not pd.DataFrame:
                raise ValueError(
                    f"The covar_mod parameter type {type(covar_mod)} is not accepted, it must be list, numpy.array or pandas.dataframe."
                )

            if covar_mod.shape[0] != len(self.batch):
                if covar_mod.shape[1] == len(self.batch):
                    LOGGER.warning(
                        "The covariate matrix seems to be transposed. The computation will proceed with the input covariate matrix transposed, but you should double-check the covariate matrix."
                    )
                    covar_mod = covar_mod.T
                else:
                    raise ValueError(
                        "The covar_mod matrix parameter doesn't match the number of sample"
                    )

            if covar_mod.columns.values.dtype == "int64":
                covar_mod.columns = [
                    "cov_" + str(col_nb) for col_nb in covar_mod.columns
                ]

            nan_covar_mod = covar_mod.isna()
            self.nan_cov = nan_covar_mod.any(axis=0).sum()
            self.nan_cov_samples = {
                b: nan_covar_mod[self.batch == b].any(axis=0).sum()
                for b in self.batch.categories
            }
        else:
            self.nan_cov = 0
            self.nan_cov_samples = {b: 0 for b in self.batch.categories}

        if self.nan_cov == 0:
            if covar_mod is not None:
                if not isinstance(covar_mod, DesignMatrix):
                    covar_mod = dmatrix(
                        "+".join([f"{cv}" for cv in covar_mod.columns]),
                        data=covar_mod,
                    )
            else:
                covar_mod = dmatrix("~1", pd.DataFrame(counts.T))

            self.confounded_cov = []
            for c in range(covar_mod.shape[1]):
                # skip the intercept
                if (covar_mod[:, c] == 1).all():
                    continue
                if (
                    np.linalg.matrix_rank(
                        np.hstack([self.batch_mod, covar_mod[:, c][:, None]])
                    )
                    <= self.batch_mod.shape[1]
                ):
                    self.confounded_cov.append(c)

        else:
            self.confounded_cov = None

        self.covar_mod = covar_mod
        if self.nan_cov == 0 and not self.nan_batch:
            self.design = dmatrix("~ 0 + self.batch_mod + covar_mod")
            # Check for intercept in covariates, and drop if present
            check = (self.design == 1).all(axis=0)
            if self.ref_batch_idx is not None:
                # the reference batch is not considered as a covariate
                check[self.ref_batch_idx] = False
            self.design = self.design[:, np.logical_not(check)]
        else:
            self.design = None

    def run_check(self, na_cov_action="raise"):
        """
        Helper function to control the quality of the input virtual cohort.

        Arguments
        ---------
        na_cov_action : {"raise", "remove", "fill"}, optional
            the action to take to handle :code:`NA` values in covariates:
            - :code:`"raise"` do nothing and raise a :code:`ValueError`
            - :code:`"remove"` remove the samples impacted by the :code:`NA` covariates
            - :code:`"fill"` replace the :code:`NA` by a new covariate category per batch.
              not applicable to non-categorical covariates (an error will be raised)
            defaults to "raise"

        Returns
        -------
        bool
            whether the input was modified
        VirtualCohortInput
            the resulting input after modification (if any)

        Examples
        --------
        Typical usage is as follows:
        >>> while True:
        ...     (modified, vc_input) = VirtualCohortInput(counts, batch, covar)
        ...     if not modified:
        ...         break
        """
        if self.nan_batch:
            raise ValueError(
                "NaN batches detected! Please review your batches before proceeding with batch effect correction."
            )

        single_sample_batches = [
            b for b, v in self.batch_composition.items() if len(v) == 1
        ]
        if len(single_sample_batches) > 0:
            raise ValueError(
                f"Batches {', '.join([str(b) for b in single_sample_batches])} contain a single sample, which is not supported for batch effect correction. Please review your inputs."
            )

        if self.nan_genes > 0 or self.nan_samples > 0:
            raise ValueError(
                f"The count matrix contains NaN values impacting {self.nan_genes} genes and {self.nan_samples} samples. Please review your inputs."
            )

        if self.nan_cov > 0:
            counts, batch, covar, list_samples, list_genes = self.fix_na_cov(
                na_cov_action
            )
            if self.input_type == "dataframe":
                counts = pd.DataFrame(counts, index=list_genes, columns=list_samples)
            elif self.input_type == "anndata":
                tmp = self.input_ad.copy()
                tmp.X = counts
                counts = tmp
            if self.ref_batch_idx is not None:
                ref_batch = self.batch[self.ref_batch_idx]
            else:
                ref_batch = None
            return (
                True,
                VirtualCohortInput(counts, batch, covar_mod=covar, ref_batch=ref_batch),
            )

        if self.confounded_cov is not None and len(self.confounded_cov) > 0:
            raise ValueError(
                f"Covariates {', '.join([str(self.covar_mod.design_info.column_names[c]) for c in self.confounded_cov])} are confounded with the batches. Please review your covariates before proceeding with batch effect correction."
            )
        return (False, self)

    def fix_na_cov(self, na_cov_action="raise"):
        """
        A helper function to handle :code:`NA` in covariates

        Arguments
        ---------
        counts : matrix
            raw count matrix from genomic studies (dimensions gene x sample)
        batch : :class:`~inmoose.utils.factor.Factor`
            batch identifiers
        covar_mod : :class:`patsy.DesignMatrix`
            covariate matrix
        na_cov_action : {"raise", "remove", "fill"}, optional
            the action to take:
            - :code:`"raise"` do nothing and raise a :code:`ValueError`
            - :code:`"remove"` remove the samples impacted by the :code:`NA` covariates
            - :code:`"fill"` replace the :code:`NA` by a new covariate category per batch.
              not applicable to non-categorical covariates (an error will be raised)
            defaults to "raise"

        Returns
        -------
        matrix
            resulting count matrix
        :class:`~inmoose.utils.factor.Factor`
            resulting batch identifiers
        :class:`patsy.DesignMatrix`
            resulting covariate Matrix
        """
        nan_covar_mod = self.covar_mod.isna()
        if na_cov_action == "raise":
            name_nan_covar_mod = nan_covar_mod.loc[:, nan_covar_mod.any()].columns
            raise ValueError(
                f"{nan_covar_mod.sum().sum()} values are missing in covariates {', '.join(name_nan_covar_mod)}. Correct your covariates or use the cov_missing_value parameters"
            )
        elif na_cov_action == "remove":
            LOGGER.warning(
                f"{(nan_covar_mod.sum(axis=1) > 0).sum()} samples with missing covariates in covar_mod. They are removed from the data. You may want to double check your covariates."
            )
            keep = nan_covar_mod.sum(axis=1) == 0
            if self.input_type in ["dataframe", "anndata"]:
                list_samples = self.list_samples[keep]
            else:
                list_samples = None
            return (
                self.counts[:, keep],
                self.batch[keep],
                self.covar_mod[keep],
                list_samples,
                self.list_genes,
            )
        elif na_cov_action == "fill":
            LOGGER.warning(
                f"{nan_covar_mod.sum().sum()} missing covariates in covar_mod. Creating a distinct covariate per batch for the missing values. You may want to double check your covariates."
            )

            # check for numeric type (excluding nan) to identify potential continuous_variable
            continuous_variable = []
            potential_continuous_variable = []
            for col in self.covar_mod.columns:
                # Return True if numeric type with decimal
                # Return False if numeric type without decimal
                col_data_type = [
                    (ele % 1) > 0
                    for ele in set(self.covar_mod[col].dropna())
                    if isinstance(ele, (float, int, complex))
                ]
                if True in col_data_type:
                    continuous_variable.append(col)
                elif False in col_data_type:
                    potential_continuous_variable.append(col)

            nan_cols = self.covar_mod.columns[self.covar_mod.isna().any(axis=0)]
            continuous_nan_cols = np.intersect1d(nan_cols, continuous_variable)
            maybe_continuous_nan_cols = np.intersect1d(
                nan_cols, potential_continuous_variable
            )
            if len(continuous_nan_cols) > 0:
                raise ValueError(
                    f"Cannot create new categories for numerical covariates {', '.join(continuous_nan_cols)}. Please fix the NA in those covariates manually."
                )
            if len(maybe_continuous_nan_cols) > 0:
                LOGGER.warning(
                    f"Creating new categories for integer covariates {', '.join(maybe_continuous_nan_cols)}. These are treated as categorical covariates, but you may want to double-check those."
                )

            # handle missing covariates, by creating a distinct covariate per batch
            # where a missing covariate appears
            covar_mod = self.covar_mod
            for col in self.covar_mod.columns:
                nan_cov_col = self.covar_mod[col].isna()
                nan_batch_group = [
                    f"nan_batch_{self.batch[i]}"
                    for i in range(len(self.covar_mod[col]))
                    if nan_cov_col[i]
                ]
                if nan_cov_col.any() and not hasattr(covar_mod[col], "str"):
                    covar_mod[col] = covar_mod[col].astype(str)
                for i, j in enumerate(np.where(nan_cov_col)[0]):
                    covar_mod.loc[j, col] = nan_batch_group[i]
            return (
                self.counts,
                self.batch,
                covar_mod,
                self.list_samples,
                self.list_genes,
            )
        else:
            raise ValueError(
                f"unknown value {na_cov_action} for argument 'na_cov_action': must be one of 'raise', 'remove' or 'fill'"
            )


def make_design_matrix(
    counts, batch, covar_mod=None, ref_batch=None, na_cov_action="raise"
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
        model matrix (dataframe, list or numpy array) for one or multiple
        covariates to include in linear model (signal from these variables are
        kept in data after adjustment). Covariates have to be categorial, they
        can not be continious values (default: `None`).
    ref_batch : any
        batch id of the batch to use as reference. Must be one of the element of
        `batch` (default: `None`).
    na_cov_action : {"raise", "remove", "fill"}, optional
        the action to take to handle :code:`NA` values in covariates:
        - :code:`"raise"` do nothing and raise a :code:`ValueError`
        - :code:`"remove"` remove the samples impacted by the :code:`NA` covariates
        - :code:`"fill"` replace the :code:`NA` by a new covariate category per batch.
          not applicable to non-categorical covariates (an error will be raised)
        defaults to "raise"

    Returns
    -------
    VirtualCohortInput
        a class containing all the information required for batch effect correction
    """

    vc_input = VirtualCohortInput(counts, batch, covar_mod, ref_batch)
    while True:
        (modified, vc_input) = vc_input.run_check(na_cov_action)
        if not modified:
            break

    LOGGER.info(f"Found {vc_input.n_batch} batches")

    if ref_batch is not None:
        LOGGER.info(f"Using batch {ref_batch} as reference batch")

    LOGGER.info(
        f"Adjusting for {vc_input.design.shape[1] - vc_input.batch_mod.shape[1]} covariate(s) or covariate level(s)"
    )

    return vc_input
