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
import pandas as pd

from ..utils import Factor


def checkFullRank(modelMatrix):
    if np.linalg.matrix_rank(modelMatrix) < modelMatrix.shape[1]:
        if np.apply_along_axis(lambda col: np.all(col == 0), 0, modelMatrix).any():
            raise ValueError(
                "the model matrix is not full rank, so the model cannot be fit as specified. Levels or combinations of levels without any samples have resulted in column(s) of zeros in the model matrix."
            )
        else:
            raise ValueError(
                "the model matrix is not full rank, so the model cannot be fit as specified. One or more variables or interaction terms in the design formula are linear combinations of the others and must be removed."
            )


def buildMatrixWithNACols(m, NACols):
    mFull = np.full((m.shape[0], len(NACols)), np.nan)
    mFull[:, ~NACols] = m
    return mFull


def buildVectorWithNACols(v, NACols):
    vFull = np.full(len(NACols), np.nan)
    vFull[~NACols] = v
    return vFull


def buildDataFrameWithNACols(d, NACols):
    v = buildMatrixWithNACols(d.values.T, NACols).T
    return pd.DataFrame(v, columns=d.columns)


def nOrMoreInCell(modelMatrix, n):
    """for each sample in the model matrix,
    are there n or more replicates in the same cell (including that sample)
    """
    cells = Factor([tuple(modelMatrix[i]) for i in range(modelMatrix.shape[0])])
    return (cells.value_counts() >= n)[cells]


def renameModelMatrixColumns(data, design):
    """convenience function to make more descriptive names for factor variables"""
    factors = [
        info
        for f, info in design.design_info.factor_infos.items()
        if info.type == "categorical"
    ]
    return {
        f"{f.factor.name()}[T.{lvl}]": f"{f.factor.name()}_{lvl}_vs_{f.categories[0]}"
        for f in factors
        for lvl in f.categories[1:]
    }


def getFactorName(name):
    """convenience function to extract the name of a categorical factor"""
    if name.startswith("C("):
        end = name.find(",")
        if end < 0:
            end = name.find(")")
        return name[2:end]
    return name


def cleanCategoricalColumnName(name):
    """convenience function to remove the "C(" part of a categorical factor in a column name"""
    bracket = name.find("[")
    if bracket >= 0:
        return getFactorName(name[:bracket]) + name[bracket:]
    else:
        return name
