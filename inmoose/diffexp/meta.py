# -----------------------------------------------------------------------------
# Copyright (C) 2024 M. Colange

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
from scipy.stats import combine_pvalues, false_discovery_control
from statsmodels.stats.meta_analysis import combine_effects

from .DEResults import DEResults


def meta_de(de_results, alpha=0.05, min_common_genes=None):
    """
    Combine logFC and *p*-values of differential expression analyses

    log-fold-changes are combined using a random-effect model, where the random
    effects variance is iteratively estimated with the Paule-Mandel method.
    Confidence intervals for the combined log-fold-change values are computed
    assuming a normal distribution and without scaling.

    *p*-values are combined using Fisher's combined probability test, then
    adjusted for multiple testing with Benjamini-Hochberg procedure.

    Arguments
    ---------
    de_results : list of DEResults
        the list of differential expression results to combine. Depending on
        the use-case, it can be results obtained with different tools on the
        same dataset, results obtained with the same tool on different
        datasets, or any combination thereof
    alpha : float between 0 and 1
        significance level for the confidence intervals. Defaults to 0.05.
    min_common_genes : int or None
        minimal number of genes all the elements of :code:`de_results` need to
        have in common. Below this threshold, an error will be raised. If
        :code:`None`, then all elements of :code:`de_results` must have the
        same set of genes.

    Returns
    -------
    pd.DataFrame
        a dataframe indexed by genes with the following columns:

        - :code:`"combined logFC"`: the combined log-fold-change
        - :code:`"combined logFC (CI_L)"`: the lower bound of the confidence
          interval for the combined log-fold-change
        - :code:`"combined logFC (CI_R)"`: the lower bound of the confidence
          interval for the combined log-fold-change
        - :code:`"adjusted combined pval"`: the combined *p*-value, adjusted
          for multiple testing
    """
    if len(de_results) < 2:
        raise ValueError("metaanalysis requires at least two analyses to aggregate")
    if not all(isinstance(d, DEResults) for d in de_results):
        raise ValueError(
            "diffexp results fed to metanalysis must inherit from DEResults"
        )

    idx = list(set.intersection(*[set(d.index) for d in de_results]))
    if min_common_genes is None:
        max_n = np.max([d.shape[0] for d in de_results])
        if len(idx) < max_n:
            raise ValueError(
                "diffexp results fed to metaanalysis must have the same set of genes"
            )
        idx = de_results[0].index
    elif len(idx) < min_common_genes:
        raise ValueError(
            f"diffexp results fed to metaanalysis must have at least {min_common_genes} in common"
        )

    de_results = [d.loc[idx, :] for d in de_results]

    # combining logFC
    logFC = {g: np.array([d.loc[g, "log2FoldChange"] for d in de_results]) for g in idx}
    lfcSE = {g: np.array([d.loc[g, "lfcSE"] for d in de_results]) for g in idx}

    ce = {g: combine_effects(logFC[g], lfcSE[g] ** 2) for g in idx}
    # NB: using random effect model with no scaling
    res = pd.DataFrame(
        {g: c.conf_int(alpha=alpha)[1] for g, c in ce.items()},
        index=["combined logFC (CI_L)", "combined logFC (CI_R)"],
    ).T
    res["combined logFC"] = [ce[g].mean_effect_re for g in res.index]

    # combining p-values
    pvals = {g: [d.loc[g, "pvalue"] for d in de_results] for g in idx}
    meta_pvals = np.array(
        [combine_pvalues(pvals[g], nan_policy="omit").pvalue for g in idx]
    )

    # multiple testing
    nan_meta_pvals = np.isnan(meta_pvals)
    meta_adj_pvals = np.zeros(meta_pvals.shape)
    meta_adj_pvals[nan_meta_pvals] = np.nan
    meta_adj_pvals[~nan_meta_pvals] = false_discovery_control(
        meta_pvals[~nan_meta_pvals]
    )
    res["adjusted combined pval"] = meta_adj_pvals

    return res
