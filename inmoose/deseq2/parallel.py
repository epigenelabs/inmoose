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

# This file is based on the file 'R/parallel.R' of the Bioconductor DESeq2
# package (version 3.16).


import numpy as np
import pandas as pd

from .fitNbinomGLMs import fitNbinomGLMs
from .misc import (
    buildDataFrameWithNACols,
    buildMatrixWithNACols,
    renameModelMatrixColumns,
)


def estimateMLEForBetaPriorVar(
    obj, maxit=100, useOptim=True, useQR=True, modelMatrixType=None
):
    """
    TODO
    """
    # this function copies code from other functions,
    # in order to allow parallelization
    objNZ = obj[:, ~obj.var["allZero"]]

    if modelMatrixType is None:
        # this code copied from nbinomWaldTest()
        termsOrder = np.array([len(t.factors) for t in obj.design.design_info.terms])
        blindDesign = np.array_equal(termsOrder, [0])
        mmTypeTest = not blindDesign
        if mmTypeTest:
            modelMatrixType = "expanded"
        else:
            modelMatrixType = "standard"

    obj.modelMatrixType = modelMatrixType

    # this code copied from fitGLMsWithPrior()
    fit = fitNbinomGLMs(
        objNZ,
        maxit=maxit,
        useOptim=useOptim,
        useQR=useQR,
        renameCols=(modelMatrixType == "standard"),
    )
    modelMatrix = fit["modelMatrix"]
    modelMatrixNames = modelMatrix.design_info.column_names
    H = fit["hat_diagonals"]

    convertNames = renameModelMatrixColumns(obj.obs, objNZ.design)
    modelMatrixNames = [
        convertNames[x] if x in convertNames else x for x in modelMatrixNames
    ]

    mleBetaMatrix = fit["betaMatrix"]
    mleBetaMatrix.columns = [f"MLE_{n}" for n in modelMatrixNames]
    mleBetaMatrix = buildDataFrameWithNACols(mleBetaMatrix, obj.var["allZero"])
    mleBetaMatrix.index = obj.var_names
    # remove any MLE columns if they exist
    obj.var = obj.var.drop(obj.var.filter(regex="MLE_").columns)
    obj.var = pd.concat([obj.var, mleBetaMatrix], axis=1)
    obj.layers["H"] = buildMatrixWithNACols(H, obj.var["allZero"])
    return obj
