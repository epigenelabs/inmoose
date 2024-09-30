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

import pandas as pd

from ..diffexp import DEResults


class DGELRT(DEResults):
    _metadata = ["fit", "comparison", "df_test", "df_total"]
    _internal_names = DEResults._internal_names + ["genes"]
    _internal_names_set = set(_internal_names)

    @property
    def _constructor(self):
        def f(*args, **kwargs):
            return DGELRT(*args, glmfit=self.fit, **kwargs)

        return f

    @property
    def _constructor_sliced(self):
        return pd.Series

    def __init__(self, df, glmfit, *args, **kwargs):
        super().__init__(df, *args, **kwargs)
        self.fit = glmfit

    @property
    def genes(self):
        return self.fit.genes
