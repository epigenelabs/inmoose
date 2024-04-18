import unittest
import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy.stats import norm

from inmoose.utils import Factor
from inmoose.limma import lmFit, nonEstimable, contrasts_fit, MArrayLM, eBayes


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
        coef_ref = [
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
        self.assertTrue(np.allclose(fit.coefficients, coef_ref, rtol=1e-6))

        self.assertTrue(np.array_equal(fit.df_residual, [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]))

        sigma_ref = [
            0.7919070,
            0.2512174,
            0.2908815,
            0.3666233,
            0.1706735,
            0.3631793,
            0.2461618,
            0.2569999,
            0.2941130,
            0.2615085,
        ]
        self.assertTrue(np.allclose(fit.sigma, sigma_ref))

        cov_ref = [[0.3333333, 0.0], [0.0, 0.3333333]]
        self.assertTrue(np.allclose(fit.cov_coefficients, cov_ref))

        self.assertTrue(np.allclose(fit.stdev_unscaled, 0.5773503))

        Amean_ref = [
            0.76971056,
            0.07452928,
            -0.23285536,
            -0.12188377,
            -0.10505440,
            -0.01094676,
            -0.09861482,
            -0.10093174,
            -0.07728329,
            0.10603292,
        ]
        self.assertTrue(np.allclose(fit.Amean, Amean_ref))

    def test_ebayes(self):
        fit = lmFit(self.y, self.design)
        fit2 = eBayes(fit)

        self.assertTrue(np.allclose(fit2.coefficients, fit.coefficients))
        self.assertTrue(np.allclose(fit2.sigma, fit.sigma))
        self.assertTrue(np.allclose(fit2.stdev_unscaled, fit.stdev_unscaled))
        self.assertTrue(np.allclose(fit2.Amean, fit.Amean))

        self.assertTrue(np.abs(fit2.df_prior - 71047.76) / 71047.76 < 1e-3)
        self.assertAlmostEqual(fit2.s2_prior, 0.1198481)
        self.assertTrue(np.allclose(fit2.var_prior, [36.68822975, 0.08343894]))
        self.assertTrue(
            np.allclose(
                fit2.s2_post,
                [
                    0.1198767,
                    0.1198449,
                    0.1198461,
                    0.1198489,
                    0.1198430,
                    0.1198488,
                    0.1198448,
                    0.1198451,
                    0.1198462,
                    0.1198452,
                ],
            )
        )
        self.assertTrue(
            np.allclose(
                fit2.t,
                [
                    [7.1734231, 0.52764098],
                    [0.9391936, -0.19341874],
                    [-1.6992077, -0.63084025],
                    [-0.4276087, -0.79199683],
                    [-0.7927456, -0.25848634],
                    [0.6189273, -0.72846394],
                    [-1.5404929, 0.55370627],
                    [-0.9482493, -0.06172022],
                    [-0.5480202, -0.22530838],
                    [0.5618936, 0.49912011],
                ],
            )
        )
        self.assertTrue(
            np.allclose(
                fit2.p_value,
                [
                    [1.076797e-08, 0.6006634],
                    [3.532684e-01, 0.8476099],
                    [9.704589e-02, 0.5317331],
                    [6.712294e-01, 0.4330345],
                    [4.326031e-01, 0.7973571],
                    [5.394734e-01, 0.4705731],
                    [1.313148e-01, 0.5828621],
                    [3.486966e-01, 0.9510930],
                    [5.867236e-01, 0.8228865],
                    [5.773237e-01, 0.6204289],
                ],
            )
        )
        self.assertTrue(
            np.allclose(
                fit2.lods,
                [
                    [9.767247, -4.678431],
                    [-6.507090, -4.702983],
                    [-5.534718, -4.666348],
                    [-6.857523, -4.643355],
                    [-6.633502, -4.699973],
                    [-6.756554, -4.653013],
                    [-5.779612, -4.675576],
                    [-6.498600, -4.706428],
                    [-6.798220, -4.701616],
                    [-6.790460, -4.681400],
                ],
            )
        )
        self.assertTrue(
            np.allclose(
                fit2.F,
                [
                    25.8682018,
                    0.4597478,
                    1.6426331,
                    0.4050541,
                    0.3476304,
                    0.4568653,
                    1.3398546,
                    0.4514930,
                    0.1755450,
                    0.2824226,
                ],
            )
        )
        F_p_value_ref = [
            5.883976e-12,
            6.314448e-01,
            1.934773e-01,
            6.669423e-01,
            7.063611e-01,
            6.332675e-01,
            2.618904e-01,
            6.366787e-01,
            8.390000e-01,
            7.539558e-01,
        ]
        self.assertTrue(np.allclose(fit2.F_p_value, F_p_value_ref, rtol=1e-4))

    def test_contrasts_fit(self):
        fit = MArrayLM(
            np.arange(20, dtype=float).reshape((10, 2)),
            np.arange(10, 30, dtype=float).reshape((10, 2)) / 30.0,
            None,
            None,
            None,
        )
        fit.coefficients = pd.DataFrame(
            fit.coefficients,
            columns=[f"x{i}" for i in range(fit.coefficients.shape[1])],
        )
        fit.stdev_unscaled = pd.DataFrame(
            fit.stdev_unscaled,
            columns=[f"x{i}" for i in range(fit.stdev_unscaled.shape[1])],
        )
        fit2 = contrasts_fit(fit, self.contrast_matrix)

        self.assertTrue(isinstance(fit2.coefficients, pd.DataFrame))
        self.assertTrue(isinstance(fit2.stdev_unscaled, pd.DataFrame))
        self.assertTrue(isinstance(fit2.cov_coefficients, pd.DataFrame))

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
                "First3": fit.stdev_unscaled.iloc[:, 0],
                "Last3": fit.stdev_unscaled.iloc[:, 1],
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

        fit.coefficients.iloc[0, 0] = np.nan
        fit3 = contrasts_fit(fit, self.contrast_matrix)
        coef_ref = pd.DataFrame(
            {
                "First3": [np.nan, 2, 4, 6, 8, 10, 12, 14, 16, 18],
                "Last3": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
                "Last3-First3": [np.nan, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            }
        )
        self.assertTrue(np.allclose(coef_ref, fit3.coefficients, equal_nan=True))

        fit.coefficients.iloc[0, 0] = 10
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
