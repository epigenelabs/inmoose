# -----------------------------------------------------------------------------
# Copyright (C) 2008-2022 Yunshun Chen, Aaron TL Lun, Davis J McCarthy, Matthew E Ritchie, Belinda Phipson, Yifang Hu, Xiaobei Zhou, Mark D Robinson, Gordon K Smyth
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

# This file is based on the file 'R/DGEList.R' of the Bioconductor edgeR package (version 3.38.4).

import numpy as np
import pandas as pd

from ..utils import LOGGER, Factor
from .utils import _isAllZero


class DGEList(object):
    """
    A class for storing read counts and associated information from digital
    gene expression or sequencing technologies.

    Attributes
    ----------
    counts : pd.DataFrame
        matrix of read counts, one row per gene and one column per sample
    samples : pd.DataFrame
        dataframe with a row for each sample and columns :code:`group`,
        :code:`lib_size` and :code:`norm_factors` containing the group labels,
        library sizes and normalization factors.
        Other columns can be optionally added to give more detailed sample
        information.
    common_dispersion : float, optional
        the overall dispersion estimate
    tagwise_dispersion : ndarray, optional
        genewise dispersion estimates for each gene ("tag" and "gene" are
        synonymous here)
    trended_dispersion : ndarray, optional
        trended dispersion estimates for each gene
    offset : array_like, optional
        matrix of same shape as :code:`counts` giving offsets for log-linear
        models
    genes : DataFrame, optional
        annotation information for each gene. Same number of rows as
        :code:`counts`
    AveLogCPM : ndarray, optional
        average log2 counts per million for each gene
    """

    from .aveLogCPM import aveLogCPM_DGEList as aveLogCPM
    from .estimateGLMCommonDisp import (
        estimateGLMCommonDisp_DGEList as estimateGLMCommonDisp,
    )
    from .estimateGLMTagwiseDisp import (
        estimateGLMTagwiseDisp_DGEList as estimateGLMTagwiseDisp,
    )
    from .glmFit import glmFit_DGEList as glmFit
    from .glmQLFit import glmQLFit_DGEList as glmQLFit
    from .predFC import predFC_DGEList as predFC
    from .splitIntoGroups import splitIntoGroups_DGEList as splitIntoGroups

    def __init__(
        self,
        counts,
        lib_size=None,
        norm_factors=None,
        samples=None,
        group=None,
        group_col="group",
        genes=None,
        remove_zeroes=False,
    ):
        """
        Construct DGEList object from components with some checking

        Arguments
        ---------
        counts : array_like or pd.DataFrame
            matrix of counts
        lib_size : array_like, optional
            vector of total counts (sequence depth) for each library
        norm_factors : array_like, optional
            vector of normalization factors that modify the library sizes
        samples : pd.DataFrame, optional
            information for each sample
        group : array_like or Factor, optional
            vector or factor giving the experimental group/condition for each
            sample/library
        group_col : str
            the name of the column containing the group information in :code:`samples`.
            only used if :code:`group` is not :code:`None`
        genes : pd.DataFrame, optional
            annotation information for each gene
        remove_zeroes : bool
            whether to remove rows that have 0 total count
        """

        # Check counts
        if not isinstance(counts, (np.ndarray, pd.DataFrame)):
            counts = np.asarray(counts)
        try:
            counts = counts.astype(int)
        except:  # noqa: E722
            raise ValueError("non-numeric values found in 'counts'")
        if counts.ndim != 2:
            raise ValueError("'counts' is not a matrix!")

        (ntags, nlibs) = counts.shape
        if not isinstance(counts, pd.DataFrame):
            counts = pd.DataFrame(
                counts,
                index=[f"gene{i}" for i in range(ntags)],
                columns=[f"sample{i}" for i in range(nlibs)],
            )
        _isAllZero(
            counts
        )  # don't really care about all-zeroes, but do want to protect against NaN, infinite and negative values

        # Check lib_size
        if lib_size is None:
            lib_size = counts.sum(axis=0)
        lib_size = np.asarray(lib_size)
        # TODO check that lib_size is numeric
        if nlibs != len(lib_size):
            raise ValueError(
                "length of 'lib_size' must be equal to the number of columns in 'counts'"
            )
        minlibsize = np.min(lib_size)
        # TODO check lib_size for NaN
        if minlibsize < 0:
            raise ValueError("negative library size not permitted")
        if minlibsize == 0:
            if np.logical_and(lib_size == 0, (counts.sum(axis=0) > 0)).any():
                raise ValueError("library size set to zero but counts are nonzero")
            LOGGER.warning("library size of zero detected")

        # Check norm_factors
        if norm_factors is None:
            norm_factors = np.ones(nlibs)
        # TODO check that norm_factors is numeric
        if nlibs != len(norm_factors):
            raise ValueError(
                "Length of 'norm_factors' must be equal to the number of columns in 'counts'"
            )
        minnf = norm_factors.min()
        # TODO check norm_factors for NaN
        if minnf <= 0:
            raise ValueError("norm factors should be positive")
        # TODO check that norm factors multiply to 1 (more or less 1-e6)

        # Check samples
        if samples is not None:
            if samples.shape[0] != nlibs:
                raise ValueError(
                    "Number of rows in 'samples' must be equal to the number of columns in 'counts'"
                )

        # Get group from samples if appropriate
        if (
            group is None
            and samples is not None
            and (samples[group_col].values != None).all()  # noqa: E711
        ):
            group = samples[group_col].values
            samples = samples.drop(columns=[group_col])

        # Check group
        if group is None:
            group = np.ones(nlibs)
            group = Factor(group)
        if len(group) != nlibs:
            raise ValueError(
                "Length of 'group' must be equal to the number of columns in 'counts'"
            )

        # Make data frame of sample informations
        # in R, acts as a dictionnary of info for each row. R allows duplicated column names (with a warning message)
        sam = pd.DataFrame(
            {"group": group, "lib_size": lib_size, "norm_factors": norm_factors}
        )
        if samples is not None:
            sam.index = samples.index
            sam = pd.concat([sam, samples], axis=1)
        samples = sam
        samples.index = counts.columns

        # make object
        self.counts = counts
        self.samples = samples
        self.design = None
        self.weights = None
        self.common_dispersion = None
        self.tagwise_dispersion = None
        self.trended_dispersion = None
        self.offset = None
        self.genes = None
        self.prior_df = None
        self.AveLogCPM = None

        # TODO Add data frame of gene information (should set self.genes)
        if genes is not None:
            if len(genes) != ntags:
                raise ValueError("Counts and genes have different number of rows")
            # TODO check for duplicated gene names (?)
            self.genes = genes

        # TODO Remove rows with all zeroes
        if remove_zeroes:
            raise NotImplementedError

    def getOffset(self):
        """
        Extract offset vector or matrix from data object and optional arguments.
        By default, offset is constructed from the lib_size and norm_factors
        """
        if self.offset is not None:
            return self.offset

        lib_size = self.samples["lib_size"]
        norm_factors = self.samples["norm_factors"]
        if (norm_factors.values != None).all():  # noqa: E711
            lib_size = lib_size * norm_factors
        return np.log(lib_size)

    def getDispersion(self):
        """
        Get most complex dispersion values from DGEList object
        """
        if self.tagwise_dispersion is not None:
            dispersion = self.tagwise_dispersion
        elif self.trended_dispersion is not None:
            dispersion = self.trended_dispersion
        elif self.common_dispersion is not None:
            dispersion = self.common_dispersion
        else:
            dispersion = None

        return dispersion
