# -----------------------------------------------------------------------------
# Copyright (C) 2022-2023 M. Colange

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


from pandas import Categorical


class Factor(Categorical):
    """
    A class to represent a factor, as in R.

    It is essentially a :obj:`pandas.Categorical` object, with additional
    methods to mimic R API.
    """

    def __init__(self, arr):
        """
        Constructs a Factor instance from an array

        Arguments
        ---------
        arr : array_like
            a list of factors
        """

        super().__init__(arr)

    def droplevels(self):
        """
        drop unused levels
        """

        return Factor(self.__array__())

    def nlevels(self):
        """
        the number of levels
        """

        return len(self.categories)


def asfactor(g):
    if type(g) is Factor:
        return g
    else:
        return Factor(g)


def gl(n, k):
    arr = []
    for i in range(1, n + 1):
        arr.extend([i for j in range(k)])
    return Factor(arr)
