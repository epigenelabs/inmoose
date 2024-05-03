import unittest
import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy.stats import norm

from inmoose.utils import Factor
from inmoose.limma import lmFit, nonEstimable, contrasts_fit, MArrayLM, eBayes, topTable


class Test(unittest.TestCase):
    def setUp(self):
        y = norm.rvs(size=(10, 6), scale=0.3, random_state=42)
        y[0, :2] += 2
        self.y = pd.DataFrame(
            y,
            index=[f"gene{i}" for i in range(10)],
            columns=[f"sample{i}" for i in range(6)],
        )
        group = Factor([1, 1, 1, 2, 2, 2])
        self.design = dmatrix("0+group")
        self.contrast_matrix = pd.DataFrame(
            {"First3": [1, 0], "Last3": [0, 1], "Last3-First3": [-1, 1]},
            index=["group1", "group2"],
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
        coef_ref = pd.DataFrame(
            {
                "group[1]": [
                    1.4339472,
                    0.1877173,
                    -0.3396236,
                    -0.0854679,
                    -0.1584454,
                    0.1237074,
                    -0.3078993,
                    -0.1895274,
                    -0.1095338,
                    0.1123062,
                ],
                "group[2]": [
                    0.10547395,
                    -0.03865874,
                    -0.12608713,
                    -0.15829963,
                    -0.05166344,
                    -0.14560097,
                    0.11066961,
                    -0.01233607,
                    -0.04503280,
                    0.09975962,
                ],
            },
            index=self.y.index,
        )

        pd.testing.assert_frame_equal(fit.coefficients, coef_ref, rtol=1e-6)

        self.assertTrue(np.array_equal(fit.df_residual, [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]))

        sigma_ref = pd.Series(
            [
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
            ],
            index=self.y.index,
        )
        pd.testing.assert_series_equal(fit.sigma, sigma_ref)

        cov_ref = pd.DataFrame(
            {"group[1]": [0.3333333, 0.0], "group[2]": [0.0, 0.3333333]},
            index=["group[1]", "group[2]"],
        )
        pd.testing.assert_frame_equal(fit.cov_coefficients, cov_ref)

        self.assertTrue(np.allclose(fit.stdev_unscaled, 0.5773503))
        self.assertTrue(np.array_equal(fit.stdev_unscaled.index, self.y.index))
        self.assertTrue(
            np.array_equal(fit.stdev_unscaled.columns, fit.coefficients.columns)
        )

        Amean_ref = pd.Series(
            [
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
            ],
            index=self.y.index,
        )
        pd.testing.assert_series_equal(fit.Amean, Amean_ref)

    def test_ebayes(self):
        fit = lmFit(self.y, self.design)
        fit2 = eBayes(fit)

        pd.testing.assert_frame_equal(fit2.coefficients, fit.coefficients)
        pd.testing.assert_series_equal(fit2.sigma, fit.sigma)
        pd.testing.assert_frame_equal(fit2.stdev_unscaled, fit.stdev_unscaled)
        pd.testing.assert_series_equal(fit2.Amean, fit.Amean)

        self.assertTrue(np.abs(fit2.df_prior - 71047.76) / 71047.76 < 1e-3)
        self.assertAlmostEqual(fit2.s2_prior, 0.1198481)
        self.assertTrue(np.allclose(fit2.var_prior, [36.68822975, 0.08343894]))
        pd.testing.assert_series_equal(
            fit2.s2_post,
            pd.Series(
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
                index=self.y.index,
            ),
        )
        pd.testing.assert_frame_equal(
            fit2.t,
            pd.DataFrame(
                {
                    "group[1]": [
                        7.1734231,
                        0.9391936,
                        -1.6992077,
                        -0.4276087,
                        -0.7927456,
                        0.6189273,
                        -1.5404929,
                        -0.9482493,
                        -0.5480202,
                        0.5618936,
                    ],
                    "group[2]": [
                        0.52764098,
                        -0.19341874,
                        -0.63084025,
                        -0.79199683,
                        -0.25848634,
                        -0.72846394,
                        0.55370627,
                        -0.06172022,
                        -0.22530838,
                        0.49912011,
                    ],
                },
                index=self.y.index,
            ),
        )
        pd.testing.assert_frame_equal(
            fit2.p_value,
            pd.DataFrame(
                {
                    "group[1]": [
                        1.076797e-08,
                        3.532684e-01,
                        9.704589e-02,
                        6.712294e-01,
                        4.326031e-01,
                        5.394734e-01,
                        1.313148e-01,
                        3.486966e-01,
                        5.867236e-01,
                        5.773237e-01,
                    ],
                    "group[2]": [
                        0.6006634,
                        0.8476099,
                        0.5317331,
                        0.4330345,
                        0.7973571,
                        0.4705731,
                        0.5828621,
                        0.9510930,
                        0.8228865,
                        0.6204289,
                    ],
                },
                index=self.y.index,
            ),
        )
        pd.testing.assert_frame_equal(
            fit2.lods,
            pd.DataFrame(
                {
                    "group[1]": [
                        9.767247,
                        -6.507090,
                        -5.534718,
                        -6.857523,
                        -6.633502,
                        -6.756554,
                        -5.779612,
                        -6.498600,
                        -6.798220,
                        -6.790460,
                    ],
                    "group[2]": [
                        -4.678431,
                        -4.702983,
                        -4.666348,
                        -4.643355,
                        -4.699973,
                        -4.653013,
                        -4.675576,
                        -4.706428,
                        -4.701616,
                        -4.681400,
                    ],
                },
                index=self.y.index,
            ),
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
            index=self.y.index,
            columns=[f"x{i}" for i in range(fit.coefficients.shape[1])],
        )
        fit.stdev_unscaled = pd.DataFrame(
            fit.stdev_unscaled,
            index=self.y.index,
            columns=[f"x{i}" for i in range(fit.stdev_unscaled.shape[1])],
        )
        fit2 = contrasts_fit(fit, self.contrast_matrix)

        coef_ref = pd.DataFrame(
            {
                "First3": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
                "Last3": [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0],
                "Last3-First3": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            },
            index=self.y.index,
        )
        pd.testing.assert_frame_equal(coef_ref, fit2.coefficients)
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
            },
            index=self.y.index,
        )
        pd.testing.assert_frame_equal(std_ref, fit2.stdev_unscaled)
        cov_ref = pd.DataFrame(
            {
                "First3": [0.4377778, 0.0, -0.4377778],
                "Last3": [0.0, 0.4811111, 0.4811111],
                "Last3-First3": [-0.4377778, 0.4811111, 0.9188889],
            },
            index=["First3", "Last3", "Last3-First3"],
        )
        pd.testing.assert_frame_equal(cov_ref, fit2.cov_coefficients)

        fit.coefficients.iloc[0, 0] = np.nan
        fit3 = contrasts_fit(fit, self.contrast_matrix)
        coef_ref = pd.DataFrame(
            {
                "First3": [np.nan, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
                "Last3": [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0],
                "Last3-First3": [np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            },
            index=self.y.index,
        )
        # NB: np.allclose instead of pd.testing.assert_frame_equal here to also
        # check columns with nan values
        self.assertTrue(np.allclose(coef_ref, fit3.coefficients, equal_nan=True))
        self.assertTrue(np.array_equal(coef_ref.index, fit3.coefficients.index))
        self.assertTrue(np.array_equal(coef_ref.columns, fit3.coefficients.columns))

        fit.coefficients.iloc[0, 0] = 10
        fit.cov_coefficients = np.cov(fit.coefficients.T)
        fit4 = contrasts_fit(fit, self.contrast_matrix)
        cov_ref = pd.DataFrame(
            {
                "First3": [26.66667, 26.66667, 4.586534e-15],
                "Last3": [26.66667, 36.66667, 10.0],
                "Last3-First3": [4.586534e-15, 10.0, 10.0],
            },
            index=["First3", "Last3", "Last3-First3"],
        )
        pd.testing.assert_frame_equal(cov_ref, fit4.cov_coefficients)

    def test_topTable(self):
        fit = lmFit(self.y, self.design)
        fit2 = eBayes(fit)
        tt = topTable(fit2)
        tt_ref = pd.DataFrame(
            {
                "group[1]": [
                    1.4339472,
                    -0.3396236,
                    -0.3078993,
                    0.1877173,
                    0.1237074,
                    -0.1895274,
                    -0.0854679,
                    -0.1584454,
                    0.1123062,
                    -0.1095338,
                ],
                "group[2]": [
                    0.10547395,
                    -0.12608713,
                    0.11066961,
                    -0.03865874,
                    -0.14560097,
                    -0.01233607,
                    -0.15829963,
                    -0.05166344,
                    0.09975962,
                    -0.04503280,
                ],
                "AveExpr": [
                    0.76971056,
                    -0.23285536,
                    -0.09861482,
                    0.07452928,
                    -0.01094676,
                    -0.10093174,
                    -0.12188377,
                    -0.10505440,
                    0.10603292,
                    -0.07728329,
                ],
                "F": [
                    25.8682018,
                    1.6426331,
                    1.3398546,
                    0.4597478,
                    0.4568653,
                    0.4514930,
                    0.4050541,
                    0.3476304,
                    0.2824226,
                    0.1755450,
                ],
                "P_Value": [
                    5.883976e-12,
                    1.934773e-01,
                    2.618904e-01,
                    6.314448e-01,
                    6.332675e-01,
                    6.366787e-01,
                    6.669423e-01,
                    7.063611e-01,
                    7.539558e-01,
                    8.390000e-01,
                ],
                "adj_P_Val": [
                    5.883976e-11,
                    8.377287e-01,
                    8.377287e-01,
                    8.377287e-01,
                    8.377287e-01,
                    8.377287e-01,
                    8.377287e-01,
                    8.377287e-01,
                    8.377287e-01,
                    8.390000e-01,
                ],
            },
            index=[
                "gene0",
                "gene2",
                "gene6",
                "gene1",
                "gene5",
                "gene7",
                "gene3",
                "gene4",
                "gene9",
                "gene8",
            ],
        )
        pd.testing.assert_frame_equal(tt, tt_ref, rtol=1e-4)

        fit_con = eBayes(contrasts_fit(fit, self.contrast_matrix))
        tt = topTable(
            fit_con, sort_by="logFC", number=np.inf, coef="Last3-First3", confint=0.9
        )
        tt_ref = pd.DataFrame(
            {
                "logFC": [
                    -1.32847322,
                    0.41856886,
                    -0.26930840,
                    -0.22637606,
                    0.21353645,
                    0.17719132,
                    0.10678193,
                    -0.07283173,
                    0.06450099,
                    -0.01254659,
                ],
                "CI_L": [
                    -1.80449340,
                    -0.05738799,
                    -0.74527323,
                    -0.70233319,
                    -0.26242309,
                    -0.29876614,
                    -0.36917141,
                    -0.54879684,
                    -0.41145876,
                    -0.48850432,
                ],
                "CI_R": [
                    -0.8524530,
                    0.8945257,
                    0.2066564,
                    0.2495811,
                    0.6894960,
                    0.6531488,
                    0.5827353,
                    0.4031334,
                    0.5404607,
                    0.4634111,
                ],
                "AveExpr": [
                    0.76971056,
                    -0.09861482,
                    -0.01094676,
                    0.07452928,
                    -0.23285536,
                    -0.10093174,
                    -0.10505440,
                    -0.12188377,
                    -0.07728329,
                    0.10603292,
                ],
                "t": [
                    -4.69927758,
                    1.48082246,
                    -0.95274945,
                    -0.80087790,
                    0.75544985,
                    0.62687070,
                    0.37777834,
                    -0.25766129,
                    0.22819170,
                    -0.04438754,
                ],
                "P_Value": [
                    3.070827e-05,
                    1.464885e-01,
                    3.464392e-01,
                    4.279339e-01,
                    4.544048e-01,
                    5.343058e-01,
                    7.075926e-01,
                    7.979893e-01,
                    8.206597e-01,
                    9.648163e-01,
                ],
                "adj_P_Val": [
                    0.0003070827,
                    0.7324425460,
                    0.8905096522,
                    0.8905096522,
                    0.8905096522,
                    0.8905096522,
                    0.9118441666,
                    0.9118441666,
                    0.9118441666,
                    0.9648163452,
                ],
                "B": [
                    2.248778,
                    -5.456045,
                    -6.076312,
                    -6.207346,
                    -6.242278,
                    -6.330372,
                    -6.455078,
                    -6.493272,
                    -6.500445,
                    -6.525565,
                ],
            },
            index=[
                "gene0",
                "gene6",
                "gene5",
                "gene1",
                "gene2",
                "gene7",
                "gene4",
                "gene3",
                "gene8",
                "gene9",
            ],
        )
        pd.testing.assert_frame_equal(tt, tt_ref)
