# -----------------------------------------------------------------------------
# Copyright (C) 2013-2022 Michael I. Love, Constantin Ahlmann-Eltze
# Copyright (C) 2023 Maximilien Colange

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

# This file is based on the file 'R/core.R' of the Bioconductor DESeq2 package
# (version 3.16).


import numpy as np

from ..utils import LOGGER


def getAndCheckWeights(obj, modelMatrix, weightThreshold=1e-2):
    """
    Check and retrive weights

    Weights checking consists in verifying that the model matrix remains full
    rank, even when considering weights below `weightThreshold` are considered
    to be zero.  Checking sets an attribute in `obj`, so that this verification
    is performed only once.

    Arguments
    ---------
    obj : DESeqDataSet
        the DESeqDataSet to retrieve weights from
    modelMatrix : matrix
        the design matrix
    weightThreshold : float
        the threshold below which weights may be considered zero

    Returns
    -------
    obj : DESeqDataSet
        the updated input `obj`
    weights : ndarray
        the matrix of weights (same shape as `obj`)
    useWeights : bool
        a flag indicating weights are used or not
    """
    if "weights" in obj.layers:
        useWeights = True
        weights = obj.layers["weights"]
        if not (weights >= 0).all():
            raise ValueError("weights must be positive")
        weights = weights / np.max(weights, 0)
        # some code for testing whether still full rank
        # only performed once per analysis, by setting obj attribute
        if obj.weightsOK is None:
            m = modelMatrix.shape[1]
            full_rank = np.linalg.matrix_rank(modelMatrix) == m
            weights_ok = np.repeat(False, weights.shape[1])
            # most designs are full rank with current version
            if full_rank:
                for i in range(weights.shape[1]):
                    # note: downweighting of samples very low will still be full rank
                    # so this test is king of minimally in play -- good for checking
                    # the user input however, e.g. all zero weights for a gene
                    test1 = (
                        np.linalg.matrix_rank(weights[:, i][:, None] * modelMatrix) == m
                    )
                    # we test that it will be possible to calculate the CR term
                    # following subsetting based on weightThreshold
                    mm_sub = modelMatrix[weights[:, i] > weightThreshold]
                    mm_sub = mm_sub[:, np.sum(np.abs(mm_sub), 0) > 0]
                    test2 = np.linalg.matrix_rank(mm_sub) == mm_sub.shape[1]
                    weights_ok[i] = test1 and test2
            else:
                # model matrix is not full rank (backwards compatibility for betaPrior=True)
                # just check for zero columns
                weights_ok = np.repeat(True, weights.shape[1])
                for j in range(modelMatrix.shape[1]):
                    num_zero = np.sum(weights.T * modelMatrix[:, j] == 0, axis=1)
                    weights_ok = weights_ok & (num_zero != modelMatrix.shape[0])

            # instead of giving an error, switch allZero to True for the problem cols
            if not weights_ok.all():
                obj.var.loc[~weights_ok, "allZero"] = True
                obj.var["weightsFail"] = ~weights_ok
                obj.var.type["weightsFail"] = "intermediate"
                obj.var.description["weightsFail"] = (
                    "weights fail to allow parameter estimation"
                )
                LOGGER.warning(
                    f"for {np.sum(~weights_ok)} genes, the weights as supplied won't allow parameter estimation, producing a degenerate design matrix. These columns have been flagged in dds.var['weightsFail'] and treated as if the column contained all zeros (dds.var['allZero'] set to True). If you are blocking for donors/organisms, consider design = ~0+donor+condition."
                )

        obj.weightsOK = True

    else:
        useWeights = False
        weights = np.ones(obj.shape)

    return (obj, weights, useWeights)
