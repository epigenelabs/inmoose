#-----------------------------------------------------------------------------
# Copyright (C) 2019-2023 A. Behdenna, A. Nordor, J. Haziza and A. Gema

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
#-----------------------------------------------------------------------------

import numpy as np

class ConfoundingVariablesError(Exception):
    """Exception raised when confounding variables are detected.

    :param message: explanation of the error
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def check_confounded_covariates(design, n_batch):
    """Detect confounded covariates.
    This function returns nothing, but raises exception if confounded covariates are detected.

    :param design: design matrix
    :type design: 2D ndarray
    :param n_batch: number of batches
    :type n_batch: int
    """

    # if matrix is not invertible, different cases
    if np.linalg.matrix_rank(design) < design.shape[1]:
        if design.shape[1] == n_batch+1: # case 1: covariate confounded with a batch
            raise ConfoundingVariablesError("Covariate is confounded with batch. Try removing the covariates.")
        if design.shape[1] > n_batch+1: # case 2: multiple covariates confounded with a batch
            if np.linalg.matrix_rank(design.T[:n_batch]) < design.shape[1]:
                raise ConfoundingVariablesError("Confounded design. Try removing one or more covariates.")
            else: # case 3: at least one covariate confounded with a batch
                raise ConfoundingVariablesError("At least one covariate is confounded with batch. Try removing confounded covariates.")
