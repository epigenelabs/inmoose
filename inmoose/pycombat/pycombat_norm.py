# -----------------------------------------------------------------------------
# Copyright (C) 2019-2024 A. Behdenna, A. Nordor, J. Haziza, A. Gema and M. Colange

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

from functools import partial

import mpmath as mp
import numpy as np
import pandas as pd

from ..utils.logging import LOGGER
from .covariates import make_design_matrix

# aprior and bprior are useful to compute "hyper-prior values"
# -> prior parameters used to estimate the prior gamma distribution for multiplicative batch effect
# aprior - calculates empirical hyper-prior values


def compute_prior(prior, gamma_hat, mean_only):
    """[summary]

    Arguments:
        prior {char} -- 'a' or 'b' depending of the prior to be calculated
        gamma_hat {matrix} -- matrix of additive batch effect
        mean_only {bool} -- True iff mean_only selected

    Returns:
        float -- [the prior calculated (aprior or bprior)
    """
    if mean_only:
        return 1
    m = np.mean(gamma_hat)
    s2 = np.var(gamma_hat)
    if prior == "a":
        return (2 * s2 + m * m) / s2
    elif prior == "b":
        return (m * s2 + m * m * m) / s2


def postmean(g_bar, d_star, t2_n, t2_n_g_hat):
    """estimates additive batch effect

    Arguments:
        g_bar {matrix} -- additive batch effect
        d_star {matrix} -- multiplicative batch effect
        t2_n {matrix} --
        t2_n_g_hat {matrix} --

    Returns:
        matrix -- estimated additive batch effect
    """
    return np.divide(t2_n_g_hat + d_star * g_bar, np.asarray(t2_n + d_star))


def postvar(sum2, n, a, b):
    """estimates multiplicative batch effect

    Arguments:
        sum2 {vector} --
        n {[type]} --
        a {float} -- aprior
        b {float} -- bprior

    Returns:
        matrix -- estimated multiplicative batch effect
    """
    return np.divide((np.multiply(0.5, sum2) + b), (np.multiply(0.5, n) + a - 1))


def it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, conv=0.0001, exit_iteration=10e5):
    """iterative solution for Empirical Bayesian method

    Arguments:
        sdat {matrix} --
        g_hat {matrix} -- average additive batch effect
        d_hat {matrix} -- average multiplicative batch effect
        g_bar {matrix} -- additive batch effect
        t2 {matrix} --
        a {float} -- aprior
        b {float} -- bprior

    Keyword Arguments:
        conv {float} -- convergence criterion (default: {0.0001})
        exit_iteration {float} -- maximum number of iterations before exit (default: {10e5})

    Returns:
        array list -- estimated additive and multiplicative batch effect
    """

    n = [len(i) for i in np.asarray(sdat)]
    t2_n = np.multiply(t2, n)
    t2_n_g_hat = np.multiply(t2_n, g_hat)
    g_old = np.ndarray.copy(g_hat)
    d_old = np.ndarray.copy(d_hat)
    change = 1
    count = 0  # number of steps needed (for diagnostic only)
    # convergence criteria, if new-old < conv, then stop
    while (change > conv) and (count < exit_iteration):
        g_new = postmean(
            g_bar, d_old, t2_n, t2_n_g_hat
        )  # updated additive batch effect
        sum2 = np.sum(
            np.asarray(
                np.square(
                    sdat - np.outer(g_new[0][0], np.ones(np.ma.size(sdat, axis=1)))
                )
            ),
            axis=1,
        )
        d_new = postvar(sum2, n, a, b)  # updated multiplicative batch effect
        change = max(
            np.amax(np.absolute(g_new - np.asarray(g_old)) / np.asarray(g_old)),
            np.amax(np.absolute(d_new - d_old) / d_old),
        )  # maximum difference between new and old estimate
        g_old = np.ndarray.copy(g_new)  # save value for g
        d_old = np.ndarray.copy(d_new)  # save value for d
        count += 1
    return (g_new, d_new)


# int_eprior - Monte Carlo integration function to find nonparametric adjustments
# Johnson et al (Biostatistics 2007, supp.mat.) show that we can estimate the multiplicative and additive batch effects with an integral
# This integral is numerically computed through Monte Carlo inegration (iterative method)


def int_eprior(sdat, g_hat, d_hat, precision):
    """int_eprior - Monte Carlo integration function to find nonparametric adjustments
        Johnson et al (Biostatistics 2007, supp.mat.) show that we can estimate the multiplicative and additive batch effects with an integral
        This integral is numerically computed through Monte Carlo inegration (iterative method)

    Arguments:
        sdat {matrix} -- data matrix
        g_hat {matrix} -- average additive batch effect
        d_hat {matrix} -- average multiplicative batch effect
        precision {float} -- level of precision for precision computing

    Returns:
        array list -- estimated additive and multiplicative batch effect
    """
    g_star = []
    d_star = []
    # use this variable to only print error message once if approximation used
    test_approximation = 0
    for i in range(len(sdat)):
        # additive batch effect
        g = np.asarray(np.delete(np.transpose(g_hat), i))
        # multiplicative batch effect
        d = np.asarray(np.delete(np.transpose(d_hat), i))
        x = np.asarray(np.transpose(sdat[i]))
        n = len(x)
        j = [1] * n
        dat = np.repeat(x, len(np.transpose(g)), axis=1)
        resid2 = np.square(dat - g)
        sum2 = np.dot(np.transpose(resid2), j)
        # /begin{handling high precision computing}
        temp_2d = 2 * d
        if precision is None:
            LH = np.power(1 / (np.pi * temp_2d), n / 2) * np.exp(
                np.negative(sum2) / (temp_2d)
            )

        else:  # only if precision parameter informed
            # increase the precision of the computing (if negative exponential too close to 0)
            mp.dps = precision
            buf_exp = np.array(list(map(mp.exp, np.negative(sum2) / (temp_2d))))
            buf_pow = np.array(
                list(map(partial(mp.power, y=n / 2), 1 / (np.pi * temp_2d)))
            )
            # print(buf_exp.dtype, buf_pow.dtype)
            LH = buf_pow * buf_exp  # likelihood
        # /end{handling high precision computing}
        LH = np.nan_to_num(LH)  # corrects NaNs in likelihood
        if np.sum(LH) == 0 and test_approximation == 0:
            test_approximation = 1  # this message won't appear again
            LOGGER.info(
                "###\nValues too small, approximation applied to avoid division by 0.\nPrecision mode can correct this problem, but increases computation time.\n###"
            )

        if np.sum(LH) == 0:  # correction for LH full of 0.0
            LH[LH == 0] = np.exp(-745)
            g_star.append(np.sum(g * LH) / np.sum(LH))
            d_star.append(np.sum(d * LH) / np.sum(LH))
        else:
            g_star.append(np.sum(g * LH) / np.sum(LH))
            d_star.append(np.sum(d * LH) / np.sum(LH))
    adjust = np.asarray([np.asarray(g_star), np.asarray(d_star)])
    return adjust


def param_fun(
    i, s_data, batches, mean_only, gamma_hat, gamma_bar, delta_hat, t2, a_prior, b_prior
):
    """parametric estimation of batch effects

    Arguments:
        i {int} -- column index
        s_data {matrix} --
        batches {list list} -- list of list of batches' elements
        mean_only {bool} -- True iff mean_only selected
        gamma_hat {matrix} -- average additive batch effect
        gamma_bar {matrix} -- estimated additive batch effect
        delta_hat {matrix} -- average multiplicative batch effect
        t2 {matrix} --
        a_prior {float} -- aprior
        b_prior {float} -- bprior

    Returns:
        array list -- estimated adjusted additive and multiplicative batch effect
    """
    if mean_only:  # if mean_only, no need for complex method: batch effect is immediately calculated
        t2_n = np.multiply(t2[i], 1)
        t2_n_g_hat = np.multiply(t2_n, gamma_hat[i])
        gamma_star = postmean(
            gamma_bar[i], 1, t2_n, t2_n_g_hat
        )  # additive batch effect
        delta_star = [1] * len(s_data)  # multiplicative batch effect
    else:  # if not(mean_only) then use it_solve
        # additive and multiplicative batch effects
        gamma_star, delta_star = it_sol(
            np.transpose(np.transpose(s_data)[batches[i]]),
            gamma_hat[i],
            delta_hat[i],
            gamma_bar[i],
            t2[i],
            a_prior[i],
            b_prior[i],
        )
    return [gamma_star, delta_star]


def nonparam_fun(i, mean_only, delta_hat, s_data, batches, gamma_hat, precision):
    """non-parametric estimation

    Arguments:
        i {int} -- column index
        mean_only {bool} -- True iff mean_only selected
        delta_hat {matrix} -- estimated multiplicative batch effect
        s_data {matrix} --
        batches {list list} -- list of list of batches' elements
        gamma_hat {matrix} -- estimated additive batch effect
        precision {float} -- level of precision for precision computing

    Returns:
        array list -- estimated adjusted additive and multiplicative batch effect
    """
    if mean_only:  # if mean only, change delta_hat to vector of 1s
        delta_hat[i] = [1] * len(delta_hat[i])
    # use int_eprior for non-parametric estimation
    temp = int_eprior(
        np.transpose(np.transpose(s_data)[batches[i]]),
        gamma_hat[i],
        delta_hat[i],
        precision,
    )
    return [temp[0], temp[1]]


############
# pyComBat #
############


def check_mean_only(mean_only):
    """checks mean_only option

    Arguments:
        mean_only {boolean} -- user's choice about mean_only

    Returns:
        ()
    """
    if mean_only:
        LOGGER.info("Using mean only version")


def calculate_mean_var(design, batches, ref, dat, n_batches, n_batch, n_array):
    """calculates the Normalisation factors

    Arguments:
        design {matrix} -- model matrix for all covariates
        batches {int list} -- list of unique batches
        ref {int} -- reference batch index
        dat {matrix} -- data matrix
        n_batches {int list} -- list of batches lengths
        n_array {int} -- total size of dataset

    Returns:
        B_hat {matrix} -- regression coefficients corresponding to the design matrix
        grand_mean {matrix} -- Mean for each gene and each batch
        var_pooled {matrix} -- Variance for each gene and each batch
    """
    LOGGER.info("Standardizing Data across genes.")
    # B_hat is the vector of regression coefficients corresponding to the design matrix
    B_hat = np.linalg.solve(
        np.dot(design, np.transpose(design)), np.dot(design, np.transpose(dat))
    )

    # Calculates the general mean
    if ref is not None:
        grand_mean = np.transpose(B_hat[ref])
    else:
        grand_mean = np.dot(
            np.transpose([i / n_array for i in n_batches]), B_hat[0:n_batch]
        )
    # Calculates the general variance
    if ref is not None:  # depending on ref batch
        ref_dat = np.transpose(np.transpose(dat)[batches[ref]])
        var_pooled = np.dot(
            np.square(
                ref_dat
                - np.transpose(np.dot(np.transpose(design)[batches[ref]], B_hat))
            ),
            [1 / n_batches[ref]] * n_batches[ref],
        )
    else:
        var_pooled = np.dot(
            np.square(dat - np.transpose(np.dot(np.transpose(design), B_hat))),
            [1 / n_array] * n_array,
        )

    return (B_hat, grand_mean, var_pooled)


def calculate_stand_mean(grand_mean, n_array, design, n_batch, B_hat):
    """transform the format of the mean for substraction

    Arguments:
        grand_mean {matrix} -- Mean for each gene and each batch
        n_array {int} -- total size of dataset
        design {[type]} -- design matrix for all covariates including batch
        n_batch {int} -- number of batches
        B_hat {matrix} -- regression coefficients corresponding to the design matrix

    Returns:
        stand_mean {matrix} -- standardised mean
    """
    stand_mean = np.dot(
        np.transpose(np.asmatrix(grand_mean)), np.asmatrix([1] * n_array)
    )
    # corrects the mean with design matrix information
    if design is not None:
        tmp = np.ndarray.copy(design)
        tmp[0:n_batch] = 0
        stand_mean = stand_mean + np.transpose(np.dot(np.transpose(tmp), B_hat))
    return stand_mean


def standardise_data(dat, stand_mean, var_pooled, n_array):
    """standardise the data: substract mean and divide by variance

    Arguments:
        dat {matrix} -- data matrix
        stand_mean {matrix} -- standardised mean
        var_pooled {matrix} -- Variance for each gene and each batch
        n_array {int} -- total size of dataset

    Returns:
        s_data {matrix} -- standardised data matrix
    """
    s_data = (dat - stand_mean) / np.dot(
        np.transpose(np.asmatrix(np.sqrt(var_pooled))), np.asmatrix([1] * n_array)
    )
    return s_data


def fit_model(design, n_batch, s_data, batches, mean_only, par_prior, precision, ref):
    LOGGER.info("Fitting L/S model and finding priors.")

    # fraction of design matrix related to batches
    batch_design = design[0:n_batch]

    # gamma_hat is the vector of additive batch effect
    gamma_hat = np.linalg.solve(
        np.dot(batch_design, np.transpose(batch_design)),
        np.dot(batch_design, np.transpose(s_data)),
    )

    delta_hat = []  # delta_hat is the vector of estimated multiplicative batch effect

    if mean_only:
        # no variance if mean_only == True
        delta_hat = [np.asarray([1] * len(s_data))] * len(batches)
    else:
        for i in batches:  # feed incrementally delta_hat
            list_map = np.transpose(np.transpose(s_data)[i]).var(
                axis=1
            )  # variance for each row
            delta_hat.append(np.squeeze(np.asarray(list_map)))

    gamma_bar = list(map(np.mean, gamma_hat))  # vector of means for gamma_hat
    t2 = list(map(np.var, gamma_hat))  # vector of variances for gamma_hat

    # calculates hyper priors for gamma (additive batch effect)
    a_prior = list(map(partial(compute_prior, "a", mean_only=mean_only), delta_hat))
    b_prior = list(map(partial(compute_prior, "b", mean_only=mean_only), delta_hat))

    # initialise gamma and delta for parameters estimation
    gamma_star = np.empty((n_batch, len(s_data)))
    delta_star = np.empty((n_batch, len(s_data)))

    if par_prior:
        # use param_fun function for parametric adjustments (cf. function definition)
        LOGGER.info("Finding parametric adjustments.")
        results = list(
            map(
                partial(
                    param_fun,
                    s_data=s_data,
                    batches=batches,
                    mean_only=mean_only,
                    gamma_hat=gamma_hat,
                    gamma_bar=gamma_bar,
                    delta_hat=delta_hat,
                    t2=t2,
                    a_prior=a_prior,
                    b_prior=b_prior,
                ),
                range(n_batch),
            )
        )
    else:
        # use nonparam_fun for non-parametric adjustments (cf. function definition)
        LOGGER.info("Finding nonparametric adjustments")
        results = list(
            map(
                partial(
                    nonparam_fun,
                    mean_only=mean_only,
                    delta_hat=delta_hat,
                    s_data=s_data,
                    batches=batches,
                    gamma_hat=gamma_hat,
                    precision=precision,
                ),
                range(n_batch),
            )
        )

    for i in range(n_batch):  # store the results in gamma/delta_star
        results_i = results[i]
        gamma_star[i], delta_star[i] = results_i[0], results_i[1]

    # update if reference batch (the reference batch is not supposed to be modified)
    if ref is not None:
        len_gamma_star_ref = len(gamma_star[ref])
        gamma_star[ref] = [0] * len_gamma_star_ref
        delta_star[ref] = [1] * len_gamma_star_ref

    return (gamma_star, delta_star, batch_design)


def adjust_data(
    s_data,
    gamma_star,
    delta_star,
    batch_design,
    n_batches,
    var_pooled,
    stand_mean,
    n_array,
    ref,
    batches,
    dat,
):
    """Adjust the data -- corrects for estimated batch effects

    Arguments:
        s_data {matrix} -- standardised data matrix
        gamma_star {matrix} -- estimated additive batch effect
        delta_star {matrix} -- estimated multiplicative batch effect
        batch_design {matrix} -- information about batches in design matrix
        n_batches {int list} -- list of batches lengths
        stand_mean {matrix} -- standardised mean
        var_pooled {matrix} -- Variance for each gene and each batch
        n_array {int} -- total size of dataset
        ref {int} -- the index of the reference batch in the batch list
        batches {int list} -- list of unique batches
        dat

    Returns:
        bayes_data [matrix] -- data adjusted for correction of batch effects
    """
    # Now we adjust the data:
    # 1. substract additive batch effect (gamma_star)
    # 2. divide by multiplicative batch effect (delta_star)
    LOGGER.info("Adjusting the Data")
    bayes_data = np.transpose(s_data)
    j = 0
    for i in batches:  # for each batch, specific correction
        bayes_data[i] = (
            bayes_data[i] - np.dot(np.transpose(batch_design)[i], gamma_star)
        ) / np.transpose(
            np.outer(np.sqrt(delta_star[j]), np.asarray([1] * n_batches[j]))
        )
        j += 1

    # renormalise the data after correction:
    # 1. multiply by variance
    # 2. add mean
    bayes_data = (
        np.multiply(
            np.transpose(bayes_data),
            np.outer(np.sqrt(var_pooled), np.asarray([1] * n_array)),
        )
        + stand_mean
    )

    # correction for reference batch
    if ref is not None:
        bayes_data[batches[ref]] = dat[batches[ref]]

    # returns the data corrected for batch effects
    return bayes_data


def pycombat_norm(
    counts,
    batch,
    covar_mod=None,
    par_prior=True,
    prior_plots=False,
    mean_only=False,
    ref_batch=None,
    precision=None,
    na_cov_action="raise",
    **kwargs,
):
    """Corrects batch effect in microarray expression data. Takes an gene expression file and a list of known batches corresponding to each sample.

    Arguments
    ---------
    counts : np.ndarray or pd.DataFrame or ad.AnnData
        expression matrix. It contains the information about the gene expression (rows) for each sample (columns).
    batch : list or str
        batch indices. Must have as many elements as the number of columns in the expression matrix. If :code:`counts` is an AnnData or a DataFrame, :code:`batch` can be the name of the column containing the batch data.
    covar_mod : list or matrix, optional
        model matrix (dataframe, list or numpy array) for one or multiple covariates to include in linear model (signal
        from these variables are kept in data after adjustment). Covariates have to be categorial,
        they can not be continious values (default: `None`).
    par_prior : bool, optional
        False for non-parametric estimation of batch effects (default: `True`).
    prior_plots : bool, optional
        True if requires to plot the priors (default: `False`). -- Not implemented yet!
    mean_only : bool, optional
        True iff just adjusting the means and not individual batch effects (default: `False`).
    ref_batch, optional
        batch id of the batch to use as reference (default: `None`)
    precision : float, optional
        level of precision for precision computing (default: `None`).
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
    check_mean_only(mean_only)

    # Handle batches, covariates and prepare design matrix
    vci = make_design_matrix(
        counts, batch, covar_mod, ref_batch, na_cov_action=na_cov_action
    )

    dat = vci.counts
    list_samples = vci.list_samples
    list_genes = vci.list_genes
    batch = vci.batch
    design = np.transpose(vci.design)
    ref = vci.ref_batch_idx

    n_batch = vci.n_batch
    batches_ind = [vci.batch_composition[b] for b in batch.categories]
    batch_sizes = [len(v) for v in batches_ind]

    n_sample = dat.shape[1]

    B_hat, grand_mean, var_pooled = calculate_mean_var(
        design, batches_ind, ref, dat, batch_sizes, n_batch, n_sample
    )
    stand_mean = calculate_stand_mean(grand_mean, n_sample, design, n_batch, B_hat)
    s_data = standardise_data(dat, stand_mean, var_pooled, n_sample)
    gamma_star, delta_star, batch_design = fit_model(
        design, n_batch, s_data, batches_ind, mean_only, par_prior, precision, ref
    )
    bayes_data = adjust_data(
        s_data,
        gamma_star,
        delta_star,
        batch_design,
        batch_sizes,
        var_pooled,
        stand_mean,
        n_sample,
        ref,
        batches_ind,
        dat,
    )

    if vci.input_type == "anndata":
        res = vci.input_ad.copy()
        res.X = bayes_data.T
        return res
    elif vci.input_type == "dataframe":
        return pd.DataFrame(bayes_data, columns=list_samples, index=list_genes)
    else:
        return bayes_data
