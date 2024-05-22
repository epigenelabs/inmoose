# -----------------------------------------------------------------------------
# Copyright (C) 2008-2022 Yunshun Chen, Aaron TL Lun, Davis J McCarthy, Matthew E Ritchie, Belinda Phipson, Yifang Hu, Xiaobei Zhou, Mark D Robinson, Gordon K Smyth
# Copyright (C) 2024 Maximilien Colange

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

# This file is based on the files 'R/topTags.R' of the Bioconductor edgeR package (version 3.38.4).

import logging

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .DGEExact import DGEExact


class TopTags:
    def __init__(self, table, adjust_method, comparison, test):
        self.table = table
        self.adjust_method = adjust_method
        self.comparison = comparison
        self.test = test

    def __repr__(self):
        return f"Coefficient: {self.comparison}\n{self.table}"


def topTags(self, n=10, adjust_method="fdr_bh", sort_by="PValue", p_value=1):
    """
    Extract the most differentially expressed genes (or sequence tags) from a test object, ranked either by *p*-value or by absolute log-fold-change.

    This function accepts a test statistic object created by any of the
    functions :func:`exactTest`, :func:`glmLRT`, :func:`glmTreat` or
    :func:`glmQLFTest` and extracts a readable dataframe of the most
    differentially expressed genes.  The dataframe collates the annotation and
    differential expression statistics for the top genes. The dataframe is
    wrapped in a :class:`TopTags` object that records the test statistic used
    and the multiple testing adjustment method.

    :func:`topTags` permits ranking by fold-change but the authors do not
    recommend fold-change ranking or fold-change cutoffs for routine RNA-Seq
    analysis. The *p*-value ranking is intended to be more biologically
    meaningful, especially if the *p*-values were computed using
    :func:`glmTreat`.

    Arguments
    ---------
    self : DGEExact or DGELRT
        object containing test statistics and *p*-values
    n : int
        maximum number of genes/tags to return
    adjust_method : str
        specify the method used to adjust *p*-values for multiple testing. See
        :func:`statsmodels.stats.multitest` for possible values
    sort_by : {"PValue, "logFC", "none"}
        specify the sort method

        - :code:`"PValue"` to sort by *p*-value
        - :code:`"logFC"` to sort by absolute log-fold-change
        - :code:`"none"` for no sorting

    p_value : float
        cutoff value for adjusted *p*-values. Only tags with adjusted
        *p*-values equal or lower than specified are returned.

    Returns
    -------
    TopTags
        an object with the following components:

        - :code:`table`, a dataframe containing differential expression results
          for the top genes in a sorted order. The number of rows is the
          smaller of :code:`n` and the number of genes with adjusted *p*-value
          less than or equal to :code:`p_value`. The dataframe includes all the
          annotation columns from :code:`self.genes` and all statistic columns
          from :code:`self.table` plus one of:

            - :code:`"FDR"`, false discovery rate (only when
              :code:`adjust_method` is :code:`"fdr_bh"`, :code:`"fdr_by"`))
            - :code:`"FWER"`, family-wise error rate (only when
              :code:`adjust_method` is :code:`"holm"`, :code:`"simes-hochberg"`,
              :code:`"hommel"` or :code:`"bonferroni"`)

        - :code:`adjust_method`, string specifying the method used to adjust
          *p*-values for multiple testing, same as input argument
        - :code:`comparison` the names of the two groups being compared (for
          :class:`DGEExact` objects) or the glm contrast being tested (for
          :class:`DGELRT` objects).
        - :code:`test`, string stating the name of the test
    """
    if self.table is None:
        raise ValueError("Need to run exactTest or glmLRT first")
    if isinstance(self, DGEExact):
        test = "exact"
    else:
        test = "glm"
    MultipleContrasts = test == "glm" and self.table.shape[1] > 4

    # Check n
    n = np.min([n, self.table.shape[0]])
    if n < 1:
        raise ValueError("No rows to output")

    # Check adjust_method
    FWER_methods = ["holm", "simes-hochberg", "hommel", "bonferroni"]
    FDR_methods = ["fdr_bh", "fdr_by"]
    if adjust_method not in FWER_methods and adjust_method not in FDR_methods:
        raise ValueError(f"invalid argument {adjust_method} for 'adjust_method'")

    # Check sort_by
    if sort_by == "p_value":
        sort_by = "PValue"
    if sort_by not in ["none", "PValue", "logFC"]:
        raise ValueError(f"invalid value {sort_by} for 'sort_by'")

    # Absolute log fold change
    if MultipleContrasts:
        if sort_by == "logFC":
            logging.warnings.warn(
                "Two or more logFC columns in DGELRT object. First logFC column used to rank by logFC"
            )
        alfc = np.abs(self.table["logFC"].iloc[:, 0])
    else:
        alfc = np.abs(self.table["logFC"])

    # Choose top genes
    if sort_by == "logFC":
        o = np.argsort(-alfc)
    elif sort_by == "PValue":
        o = np.argsort(
            np.array(
                [(a, b) for a, b in zip(self.table["PValue"], -alfc)],
                dtype=[("x", self.table["PValue"].dtype), ("y", alfc.dtype)],
            ),
            order=("x", "y"),
        )
    elif sort_by == "none":
        o = np.arange(self.table.shape[0])
    else:
        raise ValueError(f"invalid value {sort_by} for 'sort_by'")
    tab = self.table.iloc[o, :]

    # Add adjusted p-values if appropriate
    adj_p_val = multipletests(self.table["PValue"], method=adjust_method)[1]
    if adjust_method != "none":
        if adjust_method in FWER_methods:
            adjustment = "FWER"
        if adjust_method in FDR_methods:
            adjustment = "FDR"
        self.table.loc[o, adjustment] = adj_p_val[o]

    # Add gene annotation if appropriate
    if self.genes is not None:
        rn = self.genes.index
        for c in self.genes.columns:
            tab[c] = self.genes[c]
        tab.index = rn

    # Thin out fit p_value threshold
    if p_value < 1:
        sig = adj_p_val[o] <= p_value
        sig[np.isnan(sig)] = False
        tab = tab[sig, :]

    # Enough rows left?
    if tab.shape[0] < n:
        n = tab.shape[0]
    if n < 1:
        return pd.DataFrame()

    # Output object
    return TopTags(
        table=tab.iloc[:n, :],
        adjust_method=adjust_method,
        comparison=self.comparison,
        test=test,
    )
