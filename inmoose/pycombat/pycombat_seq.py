# -----------------------------------------------------------------------------
# Copyright (C) 2019-2020 Yuqing Zhang
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

# This file is based on the file 'R/ComBat_seq.R' of the Bioconductor sva package (version 3.44.0).

import logging
import numpy as np
import pandas as pd

from ..edgepy import DGEList, estimateGLMCommonDisp, estimateGLMTagwiseDisp, glmFit
from ..utils import asfactor
from .covariates import make_design_matrix
from .helper_seq import vec2mat, match_quantiles


def pycombat_seq(
    data,
    batch,
    covar_mod=None,
    shrink=False,
    shrink_disp=False,
    gene_subset_n=None,
    ref_batch=None,
    cov_missing_value=None,
):
    """pycombat_seq is an improved model from ComBat using negative binomial regression, which specifically targets RNA-Seq count data.

    Arguments
    ---------
    data : matrix
        raw count matrix (dataframe or numpy array) from genomic studies (dimensions gene x sample)
    batch : array or list or :obj:`inmoose.utils.factor.Factor`
        Batch indices. Must have as many elements as the number of columns in the expression matrix.
    covar_mod : list or matrix, optional
        model matrix (dataframe, list or numpy array) for one or multiple covariates to include in linear model (signal
        from these variables are kept in data after adjustment). Covariates have to be categorial,
        they can not be continious values (default: `None`).
    shrink : bool, optional
        whether to apply shrinkage on parameter estimation
    shrink_disp : bool, optional
        whether to apply shrinkage on dispersion
    gene_subset_n : int, optional
        number of genes to use in emprirical Bayes estimation, only useful when shrink = True
    ref_batch : any, optional
        batch id of the batch to use as reference (default: `None`)
    cov_missing_value : str
        Option to choose the way to handle missing covariates
        `None` raise an error if missing covariates and stop the code
        `remove` remove samples with missing covariates and raise a warning
        `fill` handle missing covariates, by creating a distinct covariate per batch
        (default: `None`)

    Returns
    -------
    matrix
        the input expression matrix adjusted for batch effects.
        same type as the input `data`
    """

    if isinstance(data, pd.DataFrame):
        list_samples = data.columns
        list_genes = data.index
        counts = data.values
    else:
        counts = data

    ####### Preparation #######
    # make sure batch is a factor
    batch = asfactor(batch)

    # Handle batches, covariates and prepare design matrix
    (
        design,
        batchmod,
        mod,
        batches_ind,
        n_batches,
        n_batch,
        n_sample,
        ref_batch_index,
        batch,
        remove_sample,
    ) = make_design_matrix(counts, batch, covar_mod, ref_batch, cov_missing_value)
    # Remove samples
    counts = np.delete(counts, (remove_sample), axis=1)

    # Raise error if single-sample batch, code does not support 1 sample per batch
    if 1 in n_batches:
        raise ValueError("pycombat_seq doesn't support 1 sample per batch")

    # Remove genes with only 0 counts in any batch
    keep = np.full((counts.shape[0],), True)
    for b in batch.categories:
        keep &= counts[:, batch == b].sum(axis=1) > 0
    rm = np.logical_not(keep).nonzero()[0]
    keep = keep.nonzero()[0]
    countsOri = counts.copy()
    counts = counts[keep, :]

    dge_obj = DGEList(counts=counts)

    # Check for missing values in count matrix
    if np.isnan(counts).any():
        raise ValueError(
            f"Found {np.isnan(counts).sum()} missing values (NaN) in count matrix. NaN values are not accepted. Please remove them before proceeding with pycombat_seq."
        )

    ####### Estimate gene-wise dispersions within each batch #######
    logging.info("Estimating dispersions")

    # Estimate common dispersion within each batch as an initial value
    def disp_common_helper(i):
        if (
            n_batches[i] <= design.shape[1] - batchmod.shape[1] + 1
            or np.linalg.matrix_rank(mod[batches_ind[i]]) < mod.shape[1]
        ):
            # not enough residual degree of freedom

            return estimateGLMCommonDisp(
                counts[:, batches_ind[i]], design=None, subset=counts.shape[0]
            )
        else:
            return estimateGLMCommonDisp(
                counts[:, batches_ind[i]],
                design=mod[batches_ind[i]],
                subset=counts.shape[0],
            )

    disp_common = [disp_common_helper(i) for i in range(n_batch)]

    # Estimate gene-wise dispersion within each batch
    def genewise_disp_helper(i):
        if (
            n_batches[i] <= design.shape[1] - batchmod.shape[1] + 1
            or np.linalg.matrix_rank(mod[batches_ind[i]]) < mod.shape[1]
        ):
            return [disp_common[i] for j in range(counts.shape[0])]
        else:
            return estimateGLMTagwiseDisp(
                counts[:, batches_ind[i]],
                design=mod[batches_ind[i]],
                dispersion=disp_common[i],
                prior_df=0,
            )

    genewise_disp_lst = [genewise_disp_helper(i) for i in range(n_batch)]
    # construction dispersion matrix
    phi_matrix = np.full(counts.shape, np.nan)
    for k in range(n_batch):
        phi_matrix[:, batches_ind[k]] = vec2mat(genewise_disp_lst[k], n_batches[k])

    ####### Estimate parameters from NB GLM #######
    logging.info("Fitting the GLM model")
    # no intercept - nonEstimable; compute offset (library sizes) within function
    glm_f = dge_obj.glmFit(design=design, dispersion=phi_matrix, prior_count=1e-4)
    # compute intercept as batch-size-weighted average from batches
    alpha_g = glm_f.coefficients[:, range(n_batch)] @ (n_batches / n_sample)
    # original offset - sample (library size)
    new_offset = vec2mat(dge_obj.getOffset(), counts.shape[0]).T
    # new offset - gene background expression (dge_obj.getOffset() is the same as log(dge_obj.samples.lib_size)
    new_offset += vec2mat(alpha_g, counts.shape[1])

    glm_f2 = glmFit(
        dge_obj.counts,
        design=design,
        dispersion=phi_matrix,
        offset=new_offset,
        prior_count=1e-4,
    )

    gamma_hat = glm_f2.coefficients[:, range(n_batch)]
    mu_hat = glm_f2.fitted_values
    phi_hat = np.column_stack(genewise_disp_lst)

    ####### In each batch, compute posterior estimation through Monte-Carlo integration #######
    if shrink:
        logging.info("Apply shrinkage - computing posterior estimates for parameters")
        raise NotImplementedError
    else:
        logging.info("shrinkage off - using GLM estimates for parameters")
        gamma_star_mat = gamma_hat
        phi_star_mat = phi_hat

    ####### Obtain adjusted batch-free distribution #######
    mu_star = np.full(counts.shape, np.nan)
    for jj in range(n_batch):
        mu_star[:, batches_ind[jj]] = np.exp(
            np.log(mu_hat[:, batches_ind[jj]])
            - vec2mat(gamma_star_mat[:, jj], n_batches[jj])
        )
    phi_star = phi_star_mat.mean(axis=1)

    ####### Ajust the data #######
    logging.info("Adjusting the data")
    adjust_counts = np.full(counts.shape, np.nan)
    for kk in range(n_batch):
        counts_sub = counts[:, batches_ind[kk]]
        if kk == ref_batch_index:
            adjust_counts[:, batches_ind[kk]] = counts_sub
            continue
        old_mu = mu_hat[:, batches_ind[kk]]
        old_phi = phi_hat[:, kk]
        new_mu = mu_star[:, batches_ind[kk]]
        new_phi = phi_star
        adjust_counts[:, batches_ind[kk]] = match_quantiles(
            counts_sub=counts_sub,
            old_mu=old_mu,
            old_phi=old_phi,
            new_mu=new_mu,
            new_phi=new_phi,
        )

    # Add back genes with only 0 counts in any batch (so that dimensions do not change)
    adjust_counts_whole = np.full(countsOri.shape, np.nan)
    adjust_counts_whole[keep, :] = adjust_counts
    adjust_counts_whole[rm, :] = countsOri[rm, :]

    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(adjust_counts_whole, columns=list_samples, index=list_genes)
    else:
        return adjust_counts_whole
