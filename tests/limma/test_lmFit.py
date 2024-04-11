import unittest
import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy.stats import norm

from inmoose.utils import Factor
from inmoose.limma import lmFit, nonEstimable, contrasts_fit, MArrayLM


class Test(unittest.TestCase):
    def setUp(self):
        y = norm.rvs(size=(10, 6), scale=0.3, random_state=42)
        y[0, :2] += 2
        self.y = y
        group = Factor([1, 1, 1, 2, 2, 2])
        self.design = dmatrix("0+group")
        self.contrast_matrix = pd.DataFrame(
            {"First3": [1, 0], "Last3": [0, 1], "Last3-First3": [-1, 1]}
        )

    def test_nonEstimable(self):
        d = dmatrix("x", data={"x": ["A", "B"]})
        self.assertEqual(nonEstimable(d), None)
        d = dmatrix("x+y", data={"x": ["A", "B"], "y": ["B", "A"]})
        self.assertEqual(nonEstimable(d), ["x[T.B]"])

    def test_lmFit(self):
        with self.assertRaisesRegex(
            ValueError, expected_regex="the correlation must be set"
        ):
            lmFit(self.y, self.design, ndups=2)

        # fit = lmFit(self.y, self.design, ndups=2, correlation=0.9)
        # coef_ref = np.array(
        #    [
        #        [1.144165577, 0.03340760],
        #        [-0.212545740, -0.14219338],
        #        [-0.017368967, -0.09863220],
        #        [-0.248713327, 0.04916677],
        #        [0.001386212, 0.02736341],
        #    ]
        # )
        # self.assertTrue(np.allclose(fit.coefficients, coef_ref))

        fit = lmFit(self.y, self.design)
        coef_ref = np.array(
            [
                [1.4339472, 0.10547395],
                [0.1877173, -0.03865874],
                [-0.3396236, -0.12608713],
                [-0.0854679, -0.15829963],
                [-0.1584454, -0.05166344],
                [0.1237074, -0.14560097],
                [-0.3078993, 0.11066961],
                [-0.1895274, -0.01233607],
                [-0.1095338, -0.04503280],
                [0.1123062, 0.09975962],
            ]
        )
        self.assertTrue(np.allclose(fit.coefficients, coef_ref, rtol=1e-6))

    def test_contrasts_fit(self):
        fit = MArrayLM(
            np.arange(20, dtype=float).reshape((10, 2)),
            np.arange(10, 30, dtype=float).reshape((10, 2)) / 30.0,
            None,
            None,
            None,
        )
        fit2 = contrasts_fit(fit, self.contrast_matrix)
        coef_ref = pd.DataFrame(
            {
                "First3": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
                "Last3": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
                "Last3-First3": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            }
        )
        self.assertTrue(np.allclose(coef_ref, fit2.coefficients))
        std_ref = pd.DataFrame(
            {
                "First3": fit.stdev_unscaled[:, 0],
                "Last3": fit.stdev_unscaled[:, 1],
                "Last3-First3": [
                    0.4955356,
                    0.5897269,
                    0.6839428,
                    0.7781745,
                    0.8724168,
                    0.9666667,
                    1.0609220,
                    1.1551816,
                    1.2494443,
                    1.3437096,
                ],
            }
        )
        self.assertTrue(np.allclose(std_ref, fit2.stdev_unscaled))
        cov_ref = pd.DataFrame(
            {
                "First3": [0.4377778, 0.0, -0.4377778],
                "Last3": [0.0, 0.4811111, 0.4811111],
                "Last3-First3": [-0.4377778, 0.4811111, 0.9188889],
            }
        )
        self.assertTrue(np.allclose(cov_ref, fit2.cov_coefficients))

        fit.coefficients[0, 0] = np.nan
        fit3 = contrasts_fit(fit, self.contrast_matrix)
        coef_ref = pd.DataFrame(
            {
                "First3": [np.nan, 2, 4, 6, 8, 10, 12, 14, 16, 18],
                "Last3": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
                "Last3-First3": [np.nan, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            }
        )
        self.assertTrue(np.allclose(coef_ref, fit3.coefficients, equal_nan=True))

        fit.coefficients[0, 0] = 10
        fit.cov_coefficients = np.cov(fit.coefficients.T)
        fit4 = contrasts_fit(fit, self.contrast_matrix)
        cov_ref = pd.DataFrame(
            {
                "First3": [26.66667, 26.66667, 4.586534e-15],
                "Last3": [26.66667, 36.66667, 10.0],
                "Last3-First3": [4.586534e-15, 10.0, 10.0],
            }
        )
        self.assertTrue(np.allclose(cov_ref, fit4.cov_coefficients))
