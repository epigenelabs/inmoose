# -----------------------------------------------------------------------------
# Copyright (C) 2024 M. Colange

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

import pandas as pd


class DEResults(pd.DataFrame):
    """
    A class to store the results of differential expression analysis

    It serves as a common class to store the results of differential
    expression analysis from DESeq2, edgeR, limma.
    """

    from .plot import plotMA_res as plotMA

    @property
    def _constructor(self):
        return DEResults

    @property
    def _constructor_sliced(self):
        return pd.Series

    def __init__(self, df, *args, **kwargs):
        super().__init__(df, *args, **kwargs)
        # TODO what about baseMean (mean of normalized counts)?
        # TODO what about adjusted p-values (would be adj_pvalue)?
        for col in ["log2FoldChange", "lfcSE", "pvalue"]:
            if col not in self.columns:
                raise ValueError(f"{col} missing from results table")
