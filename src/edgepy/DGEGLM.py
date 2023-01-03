class DGEGLM(object):
    def __init__(self, fit):
        # fit is a 5-tuple
        (coefficients, fitted_values, deviance, iter, failed) = fit
        self.coefficients = coefficients
        self.fitted_values = fitted_values
        self.deviance = deviance
        self.iter = iter
        self.failed = failed

        self.counts = None
        self.design = None
        self.offset = None
        self.dispersion = None
        self.weights = None
        self.prior_count = None
        self.unshrunk_coefficients = None
        self.method = None
