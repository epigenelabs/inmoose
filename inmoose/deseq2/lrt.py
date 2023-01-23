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


def nbinomLRT(
    obj,
    full=None,
    reduced=None,
    betaTol=1e-8,
    maxit=100,
    useOptim=True,
    quiet=False,
    useQR=True,
    minmu=None,
    type_="DESeq2",
):
    """
    Arguments
    ---------
    full
    reduced
    betaTol
    maxit : int
    useOptim : bool
    quiet : bool
    useQR : bool
    minmu : float
    type_ : "DESeq2" or "glmGamPoi"
    """

    if type_ not in ["DESeq2", "glmGamPoi"]:
        raise ValueError(f"invalid value for type_: {type_}")
    if "dispersion" not in obj.var:
        raise ValueError(
            "testing requires dispersion estimates, first call estimateDispersions()"
        )

    if reduced is None:
        raise ValueError(
            'provide a reduced formula for the LRT, e.g. nbinomLRT(obj, reduced="~1")'
        )

    # run check on the formula
    raise NotImplementedError()


def checkLRT(full, reduced):
    """check for LRT formulas, written as function to share code between DESeq and nbinomLRT"""
    reducedNotInFull = ~np.isin(reduced.design_info.terms, full.design_info.terms)
    if np.any(reducedNotInFull):
        reducedVars = [
            v for (v, x) in zip(reduced.design_info.term_names, reducedNotInFull) if x
        ]
        raise ValueError(
            f"the following variables in the reduced formula not in the full formula: {' '.join(reducedVars)}"
        )

    full.design_info.terms
