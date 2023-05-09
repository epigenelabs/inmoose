#-----------------------------------------------------------------------------
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
#-----------------------------------------------------------------------------

# This file is based on the file 'R/ComBat_seq.R' of the Bioconductor sva package (version 3.44.0).

import numpy as np
from pandas import DataFrame
from patsy import dmatrix
from warnings import warn

from ..edgepy import DGEList, estimateGLMCommonDisp, estimateGLMTagwiseDisp, glmFit
from ..utils import asfactor
from .covariates import check_confounded_covariates
from .helper_seq import vec2mat, match_quantiles

def pycombat_seq(counts, batch, group=None, covar_mod=None, full_mod=True, shrink=False, shrink_disp=False, gene_subset_n=None, ref_batch=None):
    """pycombat_seq is an improved model from ComBat using negative binomial regression, which specifically targets RNA-Seq count data.

    Arguments
    ---------
    counts : matrix
        raw count matrix from genomic studies (dimensions gene x sample)
    batch : vector or list or :obj:`inmoose.utils.factor.Factor`
        Batch indices. Must have as many elements as the number of columns in the expression matrix.
    group : vector or list or :obj:`inmoose.utils.factor.Factor`, optional
        vector/factor for biological condition of interest (default: `None`)
    covar_mod : matrix, optional
        model matrix for multiple covariates to include in linear model (signal from these variables are kept in data after adjustment)
    full_mod : bool, optional
        if True, include condition of interest in model
    shrink : bool, optional
        whether to apply shrinkage on parameter estimation
    shrink_disp : bool, optional
        whether to apply shrinkage on dispersion
    gene_subset_n : int, optional
        number of genes to use in emprirical Bayes estimation, only useful when shrink = True
    ref_batch, optional
        batch id of the batch to use as reference (default: `None`)

    Returns
    -------
    matrix
        the input expression matrix adjusted for batch effects
    """

    ####### Preparation #######
    batch = asfactor(batch)
    # TODO No support for single-sample batch yet

    # Remove genes with only 0 counts in any batch
    keep = np.full((counts.shape[0],), True)
    for b in batch.categories:
        keep &= counts[:, batch == b].sum(axis=1) > 0
    rm = np.logical_not(keep).nonzero()[0]
    keep = keep.nonzero()[0]
    countsOri = counts.copy()
    counts = counts[keep,:]

    dge_obj = DGEList(counts=counts)

    # Prepare characteristics on batches
    n_batch = batch.nlevels()
    # list of samples in each batch
    batches_ind = [(batch == batch.categories[i]).nonzero()[0] for i in range(n_batch)]
    n_batches = [len(i) for i in batches_ind]
    n_sample = np.sum(n_batches)
    print("Found", n_batch, "batches")

    # Make design matrix
    # batch
    batchmod = dmatrix("~-1 + C(batch)")
    # reference batch
    if ref_batch is not None:
        if ref_batch not in batch.categories:
            raise ValueError("Reference batch must identify one of the batches")
        print("Using batch", ref_batch, "as reference batch")
        # ref_batch_index is the index of the reference batch in batch.categories
        ref_batch_index = np.where(batch.categories == ref_batch)[0][0]
        # update batchmod with reference
        batchmod[:,ref_batch_index] = 1
    else:
        ref_batch_index = None

    # covariate
    group = asfactor([] if group is None else group)
    # handle missing covariates, by creating a distinct covariate per batch where a missing covariate appears
    nan_group = group.isna()
    if nan_group.any():
        warn(f"{nan_group.sum()} missing covariates in group. Creating a distinct covariate per batch for the missing values. You may want to double check your covariates.")
        nan_batch_group = [f"nan_batch_{batch[i]}" for i in range(len(group)) if nan_group[i]]
        group = group.add_categories(np.unique(nan_batch_group))
        for i,j in enumerate(np.where(nan_group)[0]):
            group[j] = nan_batch_group[i]

    if full_mod and group.nlevels() > 1:
        print("Using full model in pycombat_seq")
        mod = dmatrix("~group")
    else:
        print("Using null model in pycombat_seq")
        mod = dmatrix("~1", DataFrame(counts.T))
    # drop intercept in covariate model
    if covar_mod is not None:
        check = [(covar_mod[:,i] == 1).all() for i in range(covar_mod.shape[1])]
        covar_mod = covar_mod[:, np.logical_not(check)]
        # bind with biological condition of interest
        mod = np.concatenate((mod, covar_mod), axis=1)
    # combine
    design = dmatrix("~-1 + batchmod + mod")

    # Check for intercept in covariates, and drop if present
    check = [(design[:,i] == 1).all() for i in range(design.shape[1])]
    if ref_batch_index is not None:
        # the reference batch is not considered as a covariate
        check[ref_batch_index] = False
    design = design[:, np.logical_not(check)]
    print("Adjusting for", design.shape[1]-batchmod.shape[1], "covariate(s) or covariate level(s)")

    # Check if the design is confounded
    check_confounded_covariates(design, n_batch)

    # Check for missing values in count matrix
    nas = np.isnan(counts).any()
    if nas:
        print("Found", np.isnan(counts).sum(), "missing data values")
        raise RuntimeError("missing values in count matrix")

    ####### Estimate gene-wise dispersions within each batch #######
    print("Estimating dispersions")
    # Estimate common dispersion within each batch as an initial value
    def disp_common_helper(i):
        if n_batches[i] <= design.shape[1]-batchmod.shape[1]+1 or np.linalg.matrix_rank(mod[batches_ind[i]]) < mod.shape[1]:
            # not enough residual degree of freedom
            return estimateGLMCommonDisp(counts[:, batches_ind[i]], design=None, subset=counts.shape[0])
        else:
            return estimateGLMCommonDisp(counts[:, batches_ind[i]], design=mod[batches_ind[i]], subset=counts.shape[0])

    disp_common = [disp_common_helper(i) for i in range(n_batch)]

    # Estimate gene-wise dispersion within each batch
    def genewise_disp_helper(i):
        if n_batches[i] <= design.shape[1]-batchmod.shape[1]+1 or np.linalg.matrix_rank(mod[batches_ind[i]]) < mod.shape[1]:
            return [disp_common[i] for j in range(counts.shape[0])]
        else:
            return estimateGLMTagwiseDisp(counts[:, batches_ind[i]], design=mod[batches_ind[i]], dispersion=disp_common[i], prior_df=0)

    genewise_disp_lst = [genewise_disp_helper(i) for i in range(n_batch)]

    # construction dispersion matrix
    phi_matrix = np.full(counts.shape, np.nan)
    for k in range(n_batch):
        phi_matrix[:, batches_ind[k]] = vec2mat(genewise_disp_lst[k], n_batches[k])

    ####### Estimate parameters from NB GLM #######
    print("Fitting the GLM model")
    # no intercept - nonEstimable; compute offset (library sizes) within function
    glm_f = dge_obj.glmFit(design=design, dispersion=phi_matrix, prior_count=1e-4)
    # compute intercept as batch-size-weighted average from batches
    alpha_g = glm_f.coefficients[:, range(n_batch)] @ (n_batches/n_sample)
    # original offset - sample (library size)
    new_offset = vec2mat(dge_obj.getOffset(), counts.shape[0]).T
    # new offset - gene background expression (dge_obj.getOffset() is the same as log(dge_obj.samples.lib_size)
    new_offset += vec2mat(alpha_g, counts.shape[1])

    glm_f2 = glmFit(dge_obj.counts, design=design, dispersion=phi_matrix, offset=new_offset, prior_count=1e-4)

    gamma_hat = glm_f2.coefficients[:, range(n_batch)]
    mu_hat = glm_f2.fitted_values
    phi_hat = np.column_stack(genewise_disp_lst)

    ####### In each batch, compute posterior estimation through Monte-Carlo integration #######
    if shrink:
        print("Apply shrinkage - computing posterior estimates for parameters")
        raise NotImplementedError
    else:
        print("shrinkage off - using GLM estimates for parameters")
        gamma_star_mat = gamma_hat
        phi_star_mat = phi_hat

    ####### Obtain adjusted batch-free distribution #######
    mu_star = np.full(counts.shape, np.nan)
    for jj in range(n_batch):
        mu_star[:, batches_ind[jj]] = np.exp(np.log(mu_hat[:, batches_ind[jj]]) - vec2mat(gamma_star_mat[:, jj], n_batches[jj]))
    phi_star = phi_star_mat.mean(axis=1)

    ####### Ajust the data #######
    print("Adjusting the data")
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
        adjust_counts[:, batches_ind[kk]] = match_quantiles(counts_sub=counts_sub, old_mu=old_mu, old_phi=old_phi, new_mu=new_mu, new_phi=new_phi)

    # Add back genes with only 0 counts in any batch (so that dimensions do not change)
    adjust_counts_whole = np.full(countsOri.shape, np.nan)
    adjust_counts_whole[keep,:] = adjust_counts
    adjust_counts_whole[rm,:] = countsOri[rm,:]
    return adjust_counts_whole
