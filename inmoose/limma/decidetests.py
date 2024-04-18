# -----------------------------------------------------------------------------
# Copyright (C) 2004-2022 Gordon Smyth, Yifang Hu, Matthew Ritchie, Jeremy Silver, James Wettenhall, Davis McCarthy, Di Wu, Wei Shi, Belinda Phipson, Aaron Lun, Natalie Thorne, Alicia Oshlack, Carolyn de Graaf, Yunshun Chen, Mette Langaas, Egil Ferkingstad, Marcus Davy, Francois Pepin, Dongseok Choi
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

# This file is based on the file 'R/decidetests.R' of the Bioconductor limma package (version 3.55.1).

import numpy as np
import pandas as pd
import scipy

from ..utils import cov2cor


class TestResults(pd.DataFrame):
    """
    A matrix-based class for storing the results of simultaneous tests.

    Instances of this class are usually created by :func:`decideTests`.

    A :class:`TestResults` object is essentially a matrix with elements equal
    to :math:`0`, :math:`1` or :math:`-1`. Zero represents acceptance of the
    null hypothesis, :math:`1` indicates rejection in favor of the right tail
    alternative and :math:`-1` indicates rejection in favor of the left tail
    alternative.

    :class:`TestResults` objects can be created by :code:`TestResults(results)`
    where :code:`results` is a matrix.
    """

    _metadata = [
        "levels",
        "labels",
    ]

    @property
    def _constructor(self):
        return TestResults

    @property
    def _constructor_sliced(self):
        return pd.Series

    def __init__(self, df, *args, **kwargs):
        super().__init__(df, *args, **kwargs)
        self.levels = None
        self.labels = None

    def summary(self):
        """
        Summary of the test results

        Returns
        -------
        pd.DataFrame
            a summary table
        """
        levels = self.levels
        if levels is None:
            levels = [-1, 0, 1]
        nlevels = len(levels)
        tab = pd.DataFrame(np.zeros((nlevels, self.shape[1]), dtype=int))
        labels = self.labels
        if labels is None:
            labels = levels
        tab.index = labels
        tab.columns = self.columns
        for i in range(nlevels):
            tab.iloc[i, :] = np.nan_sum(self == levels[i], axis=1)
        return tab


class FStat(np.ndarray):
    """
    Helper class to store F-statistics in a numpy array along with two degrees of freedom attributes

    Subclassing ndarray is necessarily to add attributes to it.

    Attributes
    ----------
    df1 : int
        first degrees-of-freedom parameter of the F-statistics
    df2 : int
        second degrees-of-freedom parameter of the F-statistics
    """

    def __new__(cls, input_array, df1=None, df2=None):
        obj = np.asarray(input_array).view(cls)
        obj.df1 = df1
        obj.df2 = df2
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.df1 = getattr(obj, "df1", None)
        self.df2 = getattr(obj, "df2", None)


def classifyTestsF(self, cor_matrix=None, df=np.inf, p_value=0.01, fstat_only=False):
    """
    For each gene, classify a series of related *t*-statistics as significantly up or down using nested *F*-tests.

    This function implements the :code:`nestedF` multiple testing option
    offered by :func:`decideTests`. Users should generally use
    :func:`decideTests` rather than calling :func:`classifyTestsF` directly
    because, by itself, :func:`classifyTestsF` does not incorporate any
    multiple testing adjustment across genes. Instead, it simply tests across
    contrasts for each gene individually.

    :func:`classifyTestsF` used a nested *F*-test approach giving particular
    attention to correctly classifying genes that have two or more significant
    *t*-statistics, *i.e.* which are differentially expressed in two or more
    conditions.  For each row of :code:`tstat`, the overall *F*-statistics is
    constructed from the *t*-statistics as for :code:`FStat`.
    At least one contrast will be classified as significant if and only if the
    overall *F*-statistic is significant.
    If the overall *F*-statistic is significant, then the function makes a best
    choice as to which *t*-statistics contributed to this result.
    The methodology is based on the principle that any *t*-statistic should be
    called significant if the *F*-test is still significant for that row when
    all the larger *t*-statistics are set to the same absolute size as the
    *t*-statistic in question.

    Compared to conventional multiple testing methods, the nested *F*-test
    approach achieves better consistency between related contrasts. (For
    example, if B is judged to be different from C, then at least one of B or C
    should be different to A.) the approach was first used by [Michaud2008]_.
    The nested *F*-test approach provides *weak* control of the family-wise
    error rate, *i.e.* it correctly controls the type I error rate of calling
    any contrast as significant if all the null hypotheses are true. In other
    words, it provides error rate control at the overall *F*-test level but
    does not provide strict error rate control at the individual contrast
    level.

    Usually, :code:`self` is a limma linear model fitted object, from which a
    matrix of *t*-statistics can be extracted, but it can also be a numeric
    matrix of *t*-statistics. In either case, rows correspond to genes and
    columns to coefficients or contrasts. The :code:`cor_matrix` is the same as
    the correlation matrix of the coefficients from which the *t*-statistics
    were calculated and :code:`df` is the degrees of freedom of the
    *t*-statistics. All statistics for the same gene must have the same degrees
    of freedom.

    If :code:`fstat_only=True`, this function just returns the vector of
    overall *F*-statistics for each gene.

    Arguments
    ---------
    self : MArrayLM or ndarray
        matrix of *t*-statistics, or a :class:`MArrayLM` object from which the
        *t*-statistics may be extracted
    cor_matrix : ndarray
        covariance matrix of each of *t*-statistics. Will be extracted
        automatically from the :class:`MArrayLM` object, but otherwise defaults
        to the identity matrix.
    df : array_like
        array of degrees of freedom for the *t*-statistics. Should be
        broadcastable to the shape of :code:`tstats`. Will be extract
        automatically from the :class:`MArrayLM` object but otherwise defaults
        to :code:`np.inf`.
    p_value : float
        value between 0 and 1 giving the desired size of the test
    fstat_only : bool
        if :code:`True` then return the overall *F*-statistic as for
        :code:`FStat` instead of classifying the test results.

    Returns
    -------
    TestResults or ndarray
        if :code:`fstats_only=False`, then an object of class
        :class:`TestResults`, which is essentially a matrix with elements -1, 0
        or 1 depending on whether each *t*-statistics is classified as
        significantly negative, not significant or significantly positive
        respectively.
        if :code:`fstats_only=True`, then an array of *F*-statistics is
        returned with attributes :code:`df1` and :code:`df2` giving the
        corresponding degrees of freedom.
    """
    if isinstance(self, np.ndarray):
        tstat = self.copy()
    else:
        if self.t is None:
            raise ValueError("tstat cannot be extracted from object")
        if (cor_matrix is None) and (self.cov_coefficients is not None):
            # Check for an adjust any coefficient variances exactly zero (usually
            # caused by an all zero contrast)
            n = self.cov_coefficients.shape[0]
            if isinstance(self.cov_coefficients, pd.DataFrame):
                tmp = self.cov_coefficients.values
            else:
                tmp = self.cov_coefficients
            for i in range(n):
                if tmp[i, i] == 0:
                    tmp[i, i] = 1
            cor_matrix = cov2cor(self.cov_coefficients)
        if (
            (df is None)
            and (self.df_prior is not None)
            and (self.df_residual is not None)
        ):
            df = self.df_prior + self.df_residual
        tstat = self.t.copy()

    ngenes, ntests = tstat.shape
    if ntests == 1:
        if fstat_only:
            fstat = FStat(tstat**2)
            fstat.df1 = 1
            fstat.df2 = df
            return fstat
        else:
            p = 2 * scipy.stats.t.sf(np.abs(tstat), df)
            return TestResults(np.sign(tstat) * (p < p_value))

    # cor_matrix is estimated correlation matrix of the coefficients and also
    # the estimated covariance matrix of the t-statistics
    if cor_matrix is None:
        r = ntests
        Q = np.eye(r) / np.sqrt(r)
    else:
        evalues, evectors = np.linalg.eigh(cor_matrix)
        r = np.sum((evalues / evalues[0]) > 1e-8)
        Q = evectors[:, :r] * (1.0 / np.sqrt(evalues[:r]) / np.sqrt(r))

    # return overall moderated F-statistic only
    if fstat_only:
        fstat = FStat(np.squeeze((tstat @ Q) ** 2 @ np.ones((r, 1))))
        fstat.df1 = r
        fstat.df2 = df
        return fstat

    # return TestResults matrix
    if df > 1e10:
        # if qF ~ F(r, df), then r qF ~ chi2(r) when df -> infinity
        qF = scipy.stats.chi2.isf(p_value, r) / r
    else:
        qF = scipy.stats.f.isf(p_value, dfn=r, dfd=df)
    qF = np.broadcast_to(qF, ngenes)
    result = np.zeros((ngenes, ntests), dtype=int)
    for i in range(ngenes):
        x = tstat[i, :]
        if np.isnan(x).any():
            result[i, :] = np.nan
        else:
            if ((Q.T @ x).T @ (Q.T @ x)) > qF[i]:
                ord = np.flip(np.argsort(np.abs(x)))
                result[i, ord[0]] = np.sign(x[ord[0]])
                for j in range(1, ntests):
                    bigger = ord[:j]
                    x[bigger] = np.sign(x[bigger]) * np.abs(x[ord[j]])
                    if ((Q.T @ x).T @ (Q.T @ x)) > qF[i]:
                        result[i, ord[j]] = np.sign(x[ord[j]])
                    else:
                        break
    return TestResults(result)
