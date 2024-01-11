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

# This file is based on the file 'R/AllClasses.R' and 'R/methods.R' of the
# Bioconductor DESeq2 package (version 3.16).

from anndata import AnnData


class DESeqTransform(AnnData):
    def __init__(self, ad):
        """
        DESeqTransform constructor

        This constructor function would not typically be used by "end users". This simple class extends the :class:`~anndata.AnnData` class of the *anndata* package.
        It is used by :func:`rlog` and :func:`varianceStabilizingTransformation` to wrap up the results into a class for downstream methods, such as :func:`plotPCA`.

        Arguments
        ---------
        ad : AnnData
        """

        if not isinstance(ad, AnnData):
            raise ValueError("DESeqTransform constructor requires an AnnData as input.")
        super().__init__(X=ad)
