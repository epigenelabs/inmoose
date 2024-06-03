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

# This file is based on the file 'R/exactTest.R' of the Bioconductor edgeR package (version 3.38.4).

import numpy as np
import pandas as pd

from ..utils import asfactor
from .DGEExact import DGEExact
from .exactTestByDeviance import exactTestByDeviance
from .exactTestBySmallP import exactTestBySmallP
from .exactTestDoubleTail import exactTestDoubleTail
from .mglmOneGroup import mglmOneGroup
from .q2qnbinom import q2qnbinom


def exactTest(
    self,
    pair=(0, 1),
    dispersion="auto",
    rejection_region="doubletail",
    big_count=900,
    prior_count=0.125,
):
    """
    Compute genewise exact tests for differences in the means between two groups of negative-binomially distributed counts.

    This function tests for differential expression between two groups of count
    libraries. It implements the exact test proposed by [Robinson2008]_ for a
    difference in mean between two groups of negative binomial random
    variables.  The functions accept two groups of count libraries, and a test
    is performed for each row of data. For each row, the test is conditional on
    the sum of counts for that row. The test can be viewed as a generalization
    of the well-known exact binomial test (implemented in :func:`binomTest`) but
    generalized to overdispersed counts.

    This function is the main user-level function, and produces an object
    containing all the necessary components for downstream analysis. It calls
    one of the low-level functions :func:`exactTestDoubleTail`,
    :func:`exactTestBetaApprox`, :func:`exactTestBySmallP` or
    :func:`exactTestByDeviance` to do the *p*-value computation. The low-level
    functions all assume that the libraries have been normalized to have the
    same size, *i.e.* to have the same expected column sum under the null
    hypothesis. :func:`exactTest` equalizes the library sizes using
    :code:`equalizeLibSizes` before calling the low-level functions.

    The functions :func:`exactTestDoubleTail`, :func:`exactTestBySmallP` and
    :func:`exactTestByDeviance` correspond to different ways to define the
    two-sided rejection region when the two groups have different numbers of
    samples.  :func:`exactTestBySmallP` implements the method of small
    probabilities as proposed by [Robinson2008]_. This method corresponds
    exactly to :func:`binomTest` as the dispersion approaches zero, but gives
    poor results when the dispersion is very large.  :func:`exactTestDoubleTail`
    computed two-sided *p*-values by doubling the smaller tail probability.
    :func:`exactTestByDeviance` uses the deviance goodness of fit statistics to
    define the rejection region, and is therefore equivalent to a conditional
    likelihood ratio test.

    Note that :code:`rejection_region="smallp"` is no longer recommended. It is
    preserved as an option only for backward compatibility with earlier
    versions of edgeR. :code:`rejection_region="deviance"` has good theoretical
    statistical properties but is relatively slow to compute.
    :code:`rejection_region="doubletail"` is just slightly more conservative
    than :code:`rejection_region="deviance"`, but is recommended because of its
    much greater speed. For general remarks on different types of rejection
    regions for exact tests, see [Gibbons1975]_.

    :func:`exactTestBetaApprox` implements an asymptotic beta distribution
    approximation to the conditional count distribution. It is called by the
    other functions for rows with both group counts greater than
    :code:`big_count`.

    Arguments
    ---------
    pair : pair of ints or of strings
        the pair of groups to be compared. If strings, then should be the names
        of two groups (*e.g.* two levels of :code:`self.samples["group"]`). If
        integers, then groups to be compared are chosen by finding the levels
        :code:`self.samples["group"]` corresponding to those indices and using
        those levels as the groups to be compared. If :code:`None`, then first
        two levels of :code:`self.samples["group"]` (a factor) are used. Note that
        the first group listed in the pair is the baseline for the comparision,
        so if the pair is :code:`("A","B")` then the comparison is :code:`B -
        A`, so genes with positive log-fold changes are up-regulated in group B
        compared with group A (and vice versa for genes with negative log-fold
        change)
    dispersion : array_like of floats, or {"auto", "common", "trended", "tagwise"}
        an array of dispersions or a string indicating that dispersions should
        be taken from the data object. If floats, then can be either of length
        one or of length equal to the number of genes. Defaults to
        :code:`"auto"` to use the most complex dispersions found in data
        object.
    rejection_region : {"doubletail", "smallp", "deviance"}
        type of rejection region for two-sided exact test.
    big_count : int
        count size above which asymptotic beta approximation will be used.
    prior_count : float
        average prior count used to shrink log-fold-changes. Larger values
        produce more shrinkage.

    Returns
    -------
    DGEExact
        dataframe with two additional components:

        - :code:`comparison`, string giving the names of the two groups being compared.
        - :code:`genes`, dataframe containing annotation for each gene; taken
          from :code:`self`

        The dataframe columns has the same rows as :code:`self` and contains
        the following columns:

        - :code:`"log2FoldChange"`, log2-fold-change of expression between
          conditions being tested.
        - :code:`"lfcSE"`, standard error of log2-fold-change.
        - :code:`"logCPM"`, average log2-counts per million.
        - :code:`"pvalue"`, the two-sided *p*-values.
    """
    # Check input
    if len(pair) != 2:
        raise ValueError("Pair must be a pair!")
    if rejection_region not in ["doubletail", "deviance", "smallp"]:
        raise ValueError(f"Invalid value {rejection_region} for 'rejection_region'")

    # Get group names
    group = asfactor(self.samples["group"])
    levs_group = group.categories
    if isinstance(pair[0], int) and isinstance(pair[1], int):
        pair = levs_group[list(pair)]

    if not all(np.isin(pair, levs_group)):
        raise ValueError(
            f"At least one element of given pair is not a group.\nGroups are {' '.join(levs_group)}"
        )

    # Get dispersion vector
    if dispersion is None:
        dispersion = "auto"
    if isinstance(dispersion, str):
        if dispersion == "common":
            dispersion = self.common_dispersion
        elif dispersion == "trended":
            dispersion = self.trended_dispersion
        elif dispersion == "tagwise":
            dispersion = self.tagwise_dispersion
        elif dispersion == "auto":
            dispersion = self.getDispersion()
        else:
            raise ValueError(f"Invalid value {dispersion} for 'dispersion'")

        if dispersion is None:
            raise ValueError("specified dispersion not found in object")
        if np.isnan(dispersion).any():
            raise ValueError("dispersion is NA")
    ntags = self.counts.shape[0]
    try:
        dispersion = np.broadcast_to(dispersion, (ntags,))
    except ValueError:
        raise ValueError(
            "Dispersion provided by user must have length either 1 or the number of tags in the DGEList object"
        )

    # Reduce to two groups
    j = np.isin(group, pair)
    y = self.counts.loc[:, j]
    lib_size = self.samples["lib_size"][j]
    norm_factors = self.samples["norm_factors"][j]
    group = group[j]
    # TODO row names

    # Normalized library sizes
    lib_size = lib_size * norm_factors
    offset = np.log(lib_size)
    lib_size_average = np.exp(offset.mean())

    # logFC
    prior_count = prior_count * lib_size / lib_size.mean()
    offset_aug = np.log(lib_size + 2 * prior_count)
    j1 = group == pair[0]
    n1 = sum(j1)
    if n1 == 0:
        raise ValueError(f"No libraries for {pair[0]}")
    y1 = y.loc[:, j1]
    abundance1 = mglmOneGroup(
        y1.to_numpy() + prior_count[j1].to_numpy(),
        offset=offset_aug[j1],
        dispersion=dispersion,
    )
    j2 = group == pair[1]
    n2 = sum(j2)
    if n2 == 0:
        raise ValueError(f"No libraries for {pair[1]}")
    y2 = y.loc[:, j2]
    abundance2 = mglmOneGroup(
        y2.to_numpy() + prior_count[j2].to_numpy(),
        offset=offset_aug[j2],
        dispersion=dispersion,
    )
    logFC = (np.asarray(abundance2) - np.asarray(abundance1)) / np.log(2)

    # following parts is inspired from DESeq2 function fitBeta
    # x is the design matrix -- here, 2 groups, so a matrix of shape (n1+n2,2)
    # fitted1 and fitted2 are the fitted values in groups 1 and 2, respectively
    # contrast is the contrast matrix between groups 1 and 2
    x = np.zeros((n1 + n2, 2))
    x[:n1, 0] = 1
    x[n1:, 1] = 1
    contrast = np.array([[-1], [1]])

    fitted1 = np.exp(abundance1[:, None] + np.asarray(offset_aug[j1]))
    wvec1 = fitted1 / (1.0 + dispersion[:, None] * fitted1)
    fitted2 = np.exp(abundance2[:, None] + np.asarray(offset_aug[j2]))
    wvec2 = fitted2 / (1.0 + dispersion[:, None] * fitted2)
    w_vec = np.hstack([wvec1, wvec2])

    ridge = np.diag(np.repeat(1e-6 / (np.log(2) ** 2), x.shape[1]))
    xtwxr_inv = np.linalg.inv(x.T @ (x * w_vec[:, :, None]) + ridge)
    assert xtwxr_inv.shape == (y.shape[0], 2, 2)
    # sigma is the covariance matrix for the logFC
    sigma = xtwxr_inv @ x.T @ (x * w_vec[:, :, None]) @ xtwxr_inv
    assert sigma.shape == (y.shape[0], 2, 2)
    lfcSE = np.sqrt(contrast.T @ sigma @ contrast) / np.log(2)
    lfcSE = lfcSE.squeeze()
    assert lfcSE.shape == logFC.shape, f"{lfcSE.shape} vs {logFC.shape}"

    # Equalize library sizes
    abundance = mglmOneGroup(y.to_numpy(), dispersion=dispersion, offset=offset)
    e = np.exp(abundance)
    input_mean = np.broadcast_to(e, (n1, ntags))
    output_mean = input_mean * lib_size_average
    input_mean = (input_mean.T * np.broadcast_to(lib_size[j1], input_mean.T.shape)).T
    y1 = q2qnbinom(
        y1.T, input_mean=input_mean, output_mean=output_mean, dispersion=dispersion
    ).T
    input_mean = np.broadcast_to(e, (n2, ntags))
    output_mean = input_mean * lib_size_average
    input_mean = (input_mean.T * np.broadcast_to(lib_size[j2], input_mean.T.shape)).T
    y2 = q2qnbinom(
        y2.T, input_mean=input_mean, output_mean=output_mean, dispersion=dispersion
    ).T

    if rejection_region == "doubletail":
        exact_pvals = exactTestDoubleTail(
            y1, y2, dispersion=dispersion, big_count=big_count
        )
    elif rejection_region == "deviance":
        exact_pvals = exactTestByDeviance(y1, y2, dispersion=dispersion)
    elif rejection_region == "smallp":
        exact_pvals = exactTestBySmallP(y1, y2, dispersion=dispersion)

    AveLogCPM = self.AveLogCPM
    if AveLogCPM is None:
        AveLogCPM = self.aveLogCPM()
    de_out = pd.DataFrame(
        {
            "log2FoldChange": logFC,
            "lfcSE": lfcSE,
            "logCPM": AveLogCPM,
            "pvalue": exact_pvals,
        },
        index=self.counts.index,
    )
    return DGEExact(table=de_out, comparison=pair, genes=self.genes)
