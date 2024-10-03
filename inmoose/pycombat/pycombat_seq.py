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

import numpy as np
import pandas as pd

from ..edgepy import DGEList, estimateGLMCommonDisp, estimateGLMTagwiseDisp, glmFit
from ..utils import LOGGER
from .covariates import make_design_matrix
from .helper_seq import match_quantiles, vec2mat


def pycombat_seq(
    counts,
    batch,
    covar_mod=None,
    shrink=False,
    shrink_disp=False,
    gene_subset_n=None,
    ref_batch=None,
    na_cov_action="raise",
):
    """pycombat_seq is an improved model from ComBat using negative binomial regression, which specifically targets RNA-Seq count data.

    Arguments
    ---------
    counts : matrix
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
    na_cov_action : str
        Option to choose the way to handle missing covariates

        - :code:`"raise"` raise an error if missing covariates and stop the code
        - :code:`"remove"` remove samples with missing covariates and raise a warning
        - :code:`"fill"` handle missing covariates, by creating a distinct
          covariate per batch
        (default: :code:`"raise"`)

    Returns
    -------
    matrix
        the input expression matrix adjusted for batch effects.
        same type as the input `data`
    """

    ####### Preparation #######
    # Handle batches, covariates and prepare design matrix
    vci = make_design_matrix(
        counts,
        batch,
        covar_mod=covar_mod,
        ref_batch=ref_batch,
        na_cov_action=na_cov_action,
    )
    counts = vci.counts
    list_samples = vci.list_samples
    list_genes = vci.list_genes
    batch = vci.batch
    design = vci.design
    batchmod = vci.batch_mod
    mod = vci.covar_mod
    ref_batch_index = vci.ref_batch_idx

    n_batch = vci.n_batch
    batches_ind = vci.batch_composition
    batch_sizes = {b: len(v) for b, v in batches_ind.items()}

    n_sample = counts.shape[1]

    # Remove genes with only 0 counts in any batch
    keep = np.full((counts.shape[0],), True)
    for b in batch.categories:
        # force ndarray instead of matrices (if any), and squeeze to remove dims of length 1
        keep &= np.asarray(counts[:, batch == b].sum(axis=1) > 0).squeeze()
    rm = np.logical_not(keep).nonzero()[0]
    keep = keep.nonzero()[0]
    countsOri = counts.copy()
    counts = counts[keep, :]

    dge_obj = DGEList(counts=counts)

    ####### Estimate gene-wise dispersions within each batch #######
    LOGGER.info("Estimating dispersions")

    # Estimate common dispersion within each batch as an initial value
    def disp_common_helper(i):
        if (
            batch_sizes[i] <= design.shape[1] - batchmod.shape[1] + 1
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

    disp_common = {b: disp_common_helper(b) for b in batch.categories}

    # Estimate gene-wise dispersion within each batch
    def genewise_disp_helper(i):
        if (
            batch_sizes[i] <= design.shape[1] - batchmod.shape[1] + 1
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

    genewise_disp_lst = {b: genewise_disp_helper(b) for b in batch.categories}
    # construction dispersion matrix
    phi_matrix = np.full(counts.shape, np.nan)
    for b in batch.categories:
        phi_matrix[:, batches_ind[b]] = vec2mat(genewise_disp_lst[b], batch_sizes[b])

    ####### Estimate parameters from NB GLM #######
    LOGGER.info("Fitting the GLM model")
    # no intercept - nonEstimable; compute offset (library sizes) within function
    glm_f = dge_obj.glmFit(design=design, dispersion=phi_matrix, prior_count=1e-4)
    # compute intercept as batch-size-weighted average from batches
    alpha_g = glm_f.coefficients[:, range(n_batch)] @ (
        np.array([batch_sizes[batch.categories[i]] for i in range(n_batch)]) / n_sample
    )
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
    phi_hat = np.column_stack(
        [genewise_disp_lst[batch.categories[i]] for i in range(n_batch)]
    )

    ####### In each batch, compute posterior estimation through Monte-Carlo integration #######
    if shrink:
        LOGGER.info("Apply shrinkage - computing posterior estimates for parameters")
        raise NotImplementedError
    else:
        LOGGER.info("shrinkage off - using GLM estimates for parameters")
        gamma_star_mat = gamma_hat
        phi_star_mat = phi_hat

    ####### Obtain adjusted batch-free distribution #######
    mu_star = np.full(counts.shape, np.nan)
    for jj in range(n_batch):
        b = batch.categories[jj]
        mu_star[:, batches_ind[b]] = np.exp(
            np.log(mu_hat[:, batches_ind[b]])
            - vec2mat(gamma_star_mat[:, jj], batch_sizes[b])
        )
    phi_star = phi_star_mat.mean(axis=1)

    ####### Ajust the data #######
    LOGGER.info("Adjusting the data")
    adjust_counts = np.full(counts.shape, np.nan)
    for kk in range(n_batch):
        b = batch.categories[kk]
        counts_sub = counts[:, batches_ind[b]]
        if kk == ref_batch_index:
            adjust_counts[:, batches_ind[b]] = counts_sub
            continue
        old_mu = mu_hat[:, batches_ind[b]]
        old_phi = phi_hat[:, kk]
        new_mu = mu_star[:, batches_ind[b]]
        new_phi = phi_star
        adjust_counts[:, batches_ind[b]] = match_quantiles(
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

    if vci.input_type == "anndata":
        res = vci.input_ad.copy()
        res.X = adjust_counts_whole.T
        return res
    elif vci.input_type == "dataframe":
        return pd.DataFrame(adjust_counts_whole, columns=list_samples, index=list_genes)
    else:
        return adjust_counts_whole
