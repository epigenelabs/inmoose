# -----------------------------------------------------------------------------
# Copyright (C) 2004-2022 Gordon Smyth, Yifang Hu, Matthew Ritchie, Jeremy Silver, James Wettenhall, Davis McCarthy, Di Wu, Wei Shi, Belinda Phipson, Aaron Lun, Natalie Thorne, Alicia Oshlack, Carolyn de Graaf, Yunshun Chen, Mette Langaas, Egil Ferkingstad, Marcus Davy, Francois Pepin, Dongseok Choi
# Copyright (C) 2024-2025 Maximilien Colange

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

# This file is based on the file 'R/toptable.R' of the Bioconductor limma package (version 3.55.1).

import logging

import numpy as np
import pandas as pd
import scipy
from statsmodels.stats.multitest import multipletests

from ..diffexp import DEResults
from .marraylm import MArrayLM


def topTable(
    fit,
    coef=None,
    number=10,
    genelist=None,
    adjust_method="fdr_bh",
    sort_by="B",
    resort_by=None,
    p_value=1,
    fc=None,
    lfc=None,
    confint=False,
):
    """
    Extract a table of the top-ranked genes from a linear model fit

    This function summarizes the linear model fit object produced by
    :func:`lmFit`, :func:lm_series`, :func:`mrlm` by selecting the top-ranked
    genes for any given contrast, or for a set of contrasts. It assumes that
    the linear model fit has already been processed by :func:`eBayes`.

    If :code:`coef` has a single value, then the moderated *t*-statistics and
    *p*-values for that coefficient or contrast are used. If :code:`coef` takes
    two or more values, the moderated *F*-statistics for that set of
    coefficients or contrasts are used. If :code:`coef=None`, then all the
    coefficients or contrasts in the fitted model are used, except that any
    coefficient named :code:`Intercept` will be removed.

    The *p*-values for the coefficient/contrast of interest are adjusted for
    multiple testing by a call to
    :func:`statsmodels.stats.multitest.multipletests`. The :code:`"fdr_bh"`
    method, which controls the expected false discovery rate (FDR) below the
    specified value, is the default adjustment method because it is the most
    likely to be appropriate for microarray studies. Note that the adjusted
    *p*-values from this method are bounds on the FDR rather than *p*-values in
    the usual sense. Because they relate to FDRs rather than rejection
    probabilities, they are sometimes called *q*-values.

    Note, if there is no good evidence for differential expression in the
    experiment, that it is quite possible for all the adjusted *p*-values to be
    large, even for all of them to be equal to one. It is quite posssible for
    all the adjusted *p*-values to be equal to one if the smallest *p*-values
    is no smaller than :code:`1/ngenes` where :code:`ngenes` is the number of
    genes with non-missing *p*-values.

    The :code:`sort_by` argument specifies the criterion used to select the top
    genes. The choices are :code:`"logFC"` to sort by the (absolute)
    coefficient representing the log-fold-change; :code:`"A"` to sort by
    average expression level (over all arrays) in descending order, :code:`"T"`
    or :code:`"t"` for absolute *t*-statistic; :code:`"P"` or :code:`"p"` for
    *p*-values; or :code:`"B"` for the :code:`lods` or *B*-statistic.

    Normally the genes appear in order of selection in the output table. If a
    different order is wanted, then the :code:`resort_by` argument may be
    useful. For example, :code:`topTable(fit, sort_by="B", resort_by="logFC")`
    selects the top genes according to log-odds of differential expression and
    then orders the selected genes by log-ratio in decreasing order. Or
    :code:`topTable(fit, sort_by="logFC", resort_y="logFC")` would select the
    genes by absolute log-fold-change and then sort them from most positive to
    most negative.

    Toptable output for all probes in original (unsorted) order can be obtained
    by :code:`topTable(fit, sort="none", number=np.inf)`.

    By default :code:`number` probes are listed. Alternatively, by specifying
    :code:`p_values` and :code:`number=np.inf`, all genes with adjusted
    *p*-values below a specified value can be listed.

    The arguments :code:`fc` and :code:`lfc` give the ability to filter genes
    by log-fold change, but see the Notes below.

    Notes
    -----
    Although this function enables users to set both *p*-values and fold-change
    cutoffs, the use of fold-change cutoffs is not generally recommended. If
    the fold changes and *p*-values are not highly correlated, then the use of
    a fold change cutoff can increase the false discovery rate above the
    nominal level. Users wanting to use fold change thresholding are usually
    recommended to use :func:`treat` and :func:`topTreat` instead.

    In general, the adjusted *p*-values returned by
    :code:`adjust_method="fdr_bh"` remain valid as FDR bounds only when the
    genes remain sorted by *p*-value. Resorting the table by log-fold-change
    can increase the false discovery rate above the nominal level for genes at
    the top of resorted table.

    Arguments
    ---------
    fit : MArrayLM
        linear model fit produced by :func:`lmFit`, :func:`lm_series`,
        :func:`gls_series` or :func:`mrlm`
    coef : int or str
        column number or column name specifying which coefficient or contrast
        of the linear model is of interest. Can also be a vector of column
        subscripts, in which case the gene ranking is by *F*-statistic for that
        set of contrasts.
    number : int
        maximum number of genes to list
    genelist : array_like
        data frame or array containing gene information. Defaults to :code:`fit.genes`
    adjust_method : {"none", "fdr_bh", "fdr_by", "holm"}
        method used to adjust the *p*-values for multiple testing. See
        :func:`statsmodels.stats.multitest.multipletests` for the complete list
        of options. A :code:`None` value will result in the default adjustment
        method, which is :code:`"fdr_bh"`.
    sort_by : { "logFC", "log2FoldChange", "AveExpr", "t", "P", "p", "B", "none" }
        string specifying which statistic to rank the genes by
    resort_by : { "logFC", "AveExpr", "t", "P", "p", "B", "none" }
        string specifying statistic to sort the selected genes by in the output
        data frame
    p_value : float
        cutoff value for adjusted *p*-values. Only genes with lower *p*-values
        are listed
    fc : float, optional
        minimum fold-change required
    lfc : float, optional
        optional minimum log2-fold-change required, equal to :code:`log2(fc)`.
        :code:`fc` and :code:`lfc` are alternative ways to specify a
        fold-change cutoff and, if both are specified, then :code:`fc` takes
        precedence. If specified, then the results will include only genes with
        (at least one) absolute log-fold-change greater than :code:`lfc`
    confint : bool or float
        whether the confidence 95% intervals should be output for
        :code:`logFC`. Alternatively, can be a value between 0 and 1 specifying
        the required confidence level.

    Returns
    -------
    DEResults or pd.DataFrame
        :class:`~DEResults` if :code:`coef` has a single value, otherwise
        :class:`pd.DataFrame`. A dataframe with a row for each of the
        :code:`number` top genes and the following columns:

        - genelist: one or more columns of probe annotation, if
          :code:`genelist` was included as input
        - log2FoldChange: estimate of the log2-fold-change corresponding to the
          effect or contrast (:class:`DEResults` only)
        - CI_L: left limit of confidence interval for :code:`logFC`, if
          :code:`confint=True` or :code:`confint` is a numeric value
        - CI_R: right limit of confidence interval for :code:`logFC`, if
          :code:`confint=True` or :code:`confint` is a numeric value
        - AveExpr: average log2-expression for the probe over all arrays and
          channels, same as :code:`Amean` in the :class:`MArrayLM` object
        - stat: moderated *t*-statistic (:class:`DEResults` only)
        - F: moderated *F*-statistic (:class:`pd.DataFrame` only)
        - pvalue: raw *p*-value
        - adj_pvalue: adjusted *p*-value or *q*-value
        - B: log-odds that the gene is differentially expressed
    """
    if not isinstance(fit, MArrayLM):
        raise ValueError("fit must be a MArrayLM object")
    if (fit.t is None) and (fit.F is None):
        raise ValueError("Need to run eBayes or treat first")
    if fit.coefficients is None:
        raise ValueError("coefficients not found in fit object")
    if confint and (fit.stdev_unscaled is None):
        raise ValueError("stdev_unscaled not found in fit object")

    if genelist is None:
        genelist = fit.genes

    # Check coef
    if coef is None:
        if (not hasattr(fit, "treat_lfc")) or (fit.treat_lfc is None):
            coef = np.arange(fit.coefficients.shape[1])
        else:
            coef = fit.coefficients.shape[1]

    # Set log2-fold-change cutoff
    if fc is None:
        if lfc is None:
            lfc = 0
    else:
        if fc < 1:
            raise ValueError("fc must be greater than or equal to 1")
        lfc = np.log2(fc)

    # If testing for multiple coefficients, call low-level topTable function for F-statistics
    if not isinstance(coef, (int, str)):
        if len(coef) == 1:
            coef = coef[0]
    if not isinstance(coef, (int, str)):
        if hasattr(fit, "treat_lfc") and (fit.treat_lfc is not None):
            raise ValueError(
                "Treat p-values can only be displayed for single coefficient"
            )
        coef = np.unique(coef)
        if len(coef) < fit.coefficients.shape[1]:
            fit = fit[:, coef]
        if sort_by == "B":
            sort_by = "F"
        return _topTableF(
            fit,
            number=number,
            genelist=genelist,
            adjust_method=adjust_method,
            sort_by=sort_by,
            p_value=p_value,
            lfc=lfc,
        )

    # Call low-level topTabel function for t-statistics
    return _topTableT(
        fit,
        coef=coef,
        number=number,
        genelist=genelist,
        A=fit.Amean,
        eb=fit,
        adjust_method=adjust_method,
        sort_by=sort_by,
        resort_by=resort_by,
        p_value=p_value,
        lfc=lfc,
        confint=confint,
    )


def _topTableF(
    fit, number=10, genelist=None, adjust_method="fdr_bh", sort_by="F", p_value=1, lfc=0
):
    if genelist is None:
        genelist = fit.genes
    # Check fit
    if fit.coefficients is None:
        raise ValueError("coefficients not found in fit")
    M = fit.coefficients
    Amean = fit.Amean
    Fstat = fit.F
    Fp = fit.F_p_value
    if Fstat is None:
        raise ValueError("F-statistics not found in fit")

    # Ensure genelist is a data frame
    if (genelist is not None) and not isinstance(genelist, pd.DataFrame):
        genelist = pd.DataFrame({"ProbeID": genelist})

    # Check row names
    rn = M.index
    if len(np.unique(rn)) != len(rn):
        if genelist is None:
            genelist = pd.DataFrame({"ID": rn})
        else:
            if "ID" in genelist.columns:
                genelist["ID0"] = rn
            else:
                genelist["ID"] = rn

    # Check sort_by
    if sort_by not in ["F", "none"]:
        raise ValueError(f"invalid value {sort_by} for argument sort_by")

    # Apply multiple testing adjustment
    adj_pvalue = multipletests(Fp, method=adjust_method)[1]

    # Thin out fit by lfc and p_value thresholds
    if lfc > 0 or p_value < 1:
        if lfc > 0:
            big = np.nansum(np.abs(M) > lfc, axis=1) > 0
        else:
            big = True
        if p_value < 1:
            sig = adj_pvalue <= p_value
            sig[np.isnan(sig)] = False
        else:
            sig = True
        keep = big & sig
        if not np.all(keep):
            M = M[keep, :]
            rn = rn[keep]
            Amean = Amean[keep]
            Fstat = Fstat[keep]
            Fp = Fp[keep]
            genelist = genelist[keep, :]
            adj_pvalue = adj_pvalue[keep]

    # Enough rows left?
    if M.shape[0] < number:
        number = M.shape[0]
    if number < 1:
        return pd.DataFrame()

    # Find rows of top genes
    if sort_by == "F":
        o = np.argsort(Fp)[:number]
    else:
        o = np.arange(number)

    # Assemble data frame
    if genelist is None:
        tab = pd.DataFrame(M.iloc[o, :])
    else:
        tab = pd.DataFrame(genelist[o, :], M[o, :])
    tab["AveExpr"] = Amean.iloc[o]
    tab["F"] = Fstat[o]
    tab["pvalue"] = Fp[o]
    tab["adj_pvalue"] = adj_pvalue[o]
    tab.index = rn[o]
    return tab


def _topTableT(
    fit,
    coef=1,
    number=10,
    genelist=None,
    A=None,
    eb=None,
    adjust_method="fdr_bh",
    sort_by="B",
    resort_by=None,
    p_value=1,
    lfc=0,
    confint=False,
):
    # Check fit
    rn = fit.coefficients.index

    if not isinstance(coef, (int, str)):
        coef = coef[0]
        logging.warnings.warn(
            "Treat is for single coefficients: only first value of coef being used"
        )

    # Ensure genelist is a data frame
    if (genelist is not None) and not isinstance(genelist, pd.DataFrame):
        genelist = pd.DataFrame({"ID": genelist})

    # Check rownames
    if len(np.unique(rn)) != len(rn):
        if genelist is None:
            genelist = pd.DataFrame({"ID": rn})
        else:
            if "ID" in genelist.columns:
                genelist["ID0"] = rn
            else:
                genelist["ID"] = rn
        rn = np.arange(fit.coefficients.shape[0])

    # Check sort_by
    if sort_by not in [
        "logFC",
        "log2FoldChange",
        "M",
        "A",
        "Amean",
        "AveExpr",
        "P",
        "p",
        "T",
        "t",
        "B",
    ]:
        raise ValueError(f"invalid value {sort_by} for sort_by")
    if sort_by == "M" or sort_by == "log2FoldChange":
        sort_by = "logFC"
    if sort_by == "A" or sort_by == "Amean":
        sort_by = "AveExpr"
    if sort_by == "T":
        sort_by = "t"
    if sort_by == "p":
        sort_by = "P"

    # Check resort_by
    if resort_by is not None:
        if resort_by not in [
            "logFC",
            "log2FoldChange",
            "M",
            "A",
            "Amean",
            "AveExpr",
            "P",
            "p",
            "T",
            "t",
            "B",
        ]:
            raise ValueError(f"invalid value {sort_by} for sort_by")
        if resort_by == "M" or resort_by == "log2FoldChange":
            resort_by = "logFC"
        if resort_by == "A" or resort_by == "Amean":
            resort_by = "AveExpr"
        if resort_by == "T":
            resort_by = "t"
        if resort_by == "p":
            resort_by = "P"

    # Check A
    if A is None:
        if sort_by == "A":
            raise ValueError("Cannot sort by A-values as these have not been given")
    else:
        if A.ndim == 2:
            A = np.nanmean(A, axis=1)

    # Check for lods component
    if eb.lods is None:
        if sort_by == "B":
            raise ValueError(
                "Trying to sort_by B, but B-statistic (lods) not found in MArrayLM object"
            )
        if resort_by == "B":
            raise ValueError(
                "Trying to resort_by B, but B-statistic (lods) not found in MArrayLM object"
            )
        include_B = False
    else:
        include_B = True

    # Extract statistics for table
    M = fit.coefficients.loc[:, coef]
    lfcSE = fit.stdev_unscaled.loc[:, coef]
    tstat = eb.t.loc[:, coef]
    P_Value = eb.p_value.loc[:, coef]
    if include_B:
        B = eb.lods.loc[:, coef]

    # Apply multiple testing adjustment
    adj_pvalue = multipletests(P_Value, method=adjust_method)[1]

    # Thin out fit by p_value and lfc thresholds
    if p_value < 1 or lfc > 0:
        sig = (adj_pvalue <= p_value) & (np.abs(M) >= lfc)
        if np.any(np.isnan(sig)):
            sig[np.isnan(sig)] = False
        if not np.any(sig):
            return pd.DataFrame()
        genelist = genelist[sig, :]
        M = M[sig]
        lfcSE = lfcSE[sig]
        A = A[sig]
        tstat = tstat[sig]
        P_Value = P_Value[sig]
        adj_pvalue = adj_pvalue[sig]
        if include_B:
            B = B[sig]
        rn = rn[sig]

    # Are enough rows left?
    if len(M) < number:
        number = len(M)
    if number < 1:
        return pd.DataFrame()

    # Select top rows
    if sort_by == "logFC":
        ord = np.flip(np.argsort(np.abs(M)))
    elif sort_by == "AveExpr":
        ord = np.flip(np.argsort(A))
    elif sort_by == "P":
        ord = np.argsort(P_Value)
    elif sort_by == "t":
        ord = np.flip(np.argsort(np.abs(tstat)))
    elif sort_by == "B":
        ord = np.flip(np.argsort(B))
    elif sort_by == "none":
        ord = np.arange(len(M))
    top = ord[:number]

    # Assemble output data frame
    if genelist is None:
        tab = pd.DataFrame({"log2FoldChange": M.iloc[top], "lfcSE": lfcSE.iloc[top]})
    else:
        tab = pd.DataFrame(genelist.loc[top, :])
        tab["log2FoldChange"] = M[top]
        tab["lfcSE"] = lfcSE[top]

    if confint is not False:
        if isinstance(confint, (int, float)):
            alpha = (1 + confint) / 2
        else:
            alpha = 0.975
        margin_error = (
            np.sqrt(eb.s2_post.iloc[top])
            * fit.stdev_unscaled[coef].iloc[top]
            * scipy.stats.t.ppf(alpha, df=eb.df_total[top])
        )
        tab["CI_L"] = M.iloc[top] - margin_error
        tab["CI_R"] = M.iloc[top] + margin_error

    if A is not None:
        tab["AveExpr"] = A.iloc[top]
    tab["stat"] = tstat.iloc[top]
    tab["pvalue"] = P_Value.iloc[top]
    tab["adj_pvalue"] = adj_pvalue[top]

    if include_B:
        tab["B"] = B.iloc[top]
    tab.index = rn[top]

    # Resort table
    if resort_by is not None:
        if resort_by == "logFC":
            ord = np.flip(np.argsort(tab["log2FoldChange"]))
        elif resort_by == "AveExpr":
            ord = np.flip(np.argsort(tab["AveExpr"]))
        elif resort_by == "P":
            ord = np.argsort(tab["pvalue"])
        elif resort_by == "t":
            ord = np.flip(np.argsort(tab["stat"]))
        elif resort_by == "B":
            ord = np.flip(np.argsort(tab["B"]))
        tab = tab[ord, :]

    return DEResults(tab)
