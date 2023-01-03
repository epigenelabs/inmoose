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
