import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import t

from inmoose.limma import eBayes, lmFit
from inmoose.limma.ebayes import _ebayes
from inmoose.limma.squeezeVar import _squeezeVar


class Test(unittest.TestCase):
    def setUp(self):
        self.design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])
        self.coefficients = np.column_stack(
            [np.arange(1, 7), [0, 1, -1, 2, -2, 0.5]]
        )
        self.residuals = np.array([-1, 0, 1, -1, 0, 1])
        self.y = pd.DataFrame(
            self.coefficients @ self.design.T + self.residuals,
            index=["gene5", "gene2", "gene0", "gene4", "gene1", "gene3"],
        )

    def test_infinite_prior_df(self):
        # Issue #181: identical residual variances give scalar df_prior = Inf.
        # References from R 4.3.3 / limma 3.65.1:
        # design <- cbind(1, rep(0:1, each=3))
        # coef <- cbind(1:6, c(0, 1, -1, 2, -2, 0.5))
        # y <- coef %*% t(design) + rep(c(-1, 0, 1, -1, 0, 1), each=6)
        # fit <- eBayes(lmFit(y, design), trend=trend)
        references = [
            (
                False,
                1.0,
                [16.0, 4.554202998685711],
                [
                    [-5.071642244087860, -5.624184399476577],
                    [-0.663478978781731, -4.969953884648403],
                    [6.683459796728486, -4.969953884648401],
                    [16.969174082442787, -3.007262340163881],
                    [30.193663878361185, -3.007262340163875],
                    [46.356929184483640, -5.460626770769533],
                ],
            ),
            (
                True,
                1.310439851632204,
                [12.209640892766947, 1.899564113381714],
                [
                    [-5.294772025669603, -5.269071504095818],
                    [-1.952069089976809, -4.845426179279310],
                    [3.619102469511181, -4.845426179279309],
                    [11.418742652794371, -3.574490204829788],
                    [21.446851459872754, -3.574490204829784],
                    [33.703428890746324, -5.163160172891691],
                ],
            ),
        ]
        for trend, s2_prior, var_prior, lods in references:
            with self.subTest(trend=trend):
                fit = eBayes(lmFit(self.y, self.design), trend=trend)
                self.assertEqual(np.ndim(fit.df_prior), 0)
                self.assertTrue(np.isposinf(fit.df_prior))
                np.testing.assert_allclose(
                    fit.coefficients, self.coefficients, atol=1e-14
                )
                np.testing.assert_allclose(fit.sigma**2, 1.0, rtol=1e-12)
                np.testing.assert_allclose(fit.s2_prior, s2_prior, rtol=1e-12)
                np.testing.assert_allclose(fit.s2_post, s2_prior, rtol=1e-12)
                np.testing.assert_array_equal(fit.df_total, np.full(6, 24.0))
                expected_t = self.coefficients / np.sqrt(
                    np.array([1 / 3, 2 / 3]) * s2_prior
                )
                np.testing.assert_allclose(fit.t, expected_t, atol=1e-12)
                np.testing.assert_allclose(
                    fit.p_value, 2 * t.sf(np.abs(expected_t), df=24), rtol=1e-10
                )
                np.testing.assert_allclose(fit.var_prior, var_prior, rtol=1e-10)
                np.testing.assert_allclose(fit.lods, lods, rtol=1e-10)
                # The B kernel uses the normal limit even though df_total is
                # capped at pooled residual df for the moderated t-test.
                r = 1 + np.array(var_prior) / np.array([1 / 3, 2 / 3])
                expected_lods = (
                    np.log(0.01 / 0.99)
                    - np.log(r) / 2
                    + expected_t**2 * (1 - 1 / r) / 2
                )
                np.testing.assert_allclose(fit.lods, expected_lods, rtol=1e-10)
                for result in (fit.t, fit.p_value, fit.lods):
                    self.assertTrue(np.isfinite(result).all().all())
                    pd.testing.assert_index_equal(result.index, self.y.index)
                    pd.testing.assert_index_equal(
                        result.columns, fit.coefficients.columns
                    )
                pd.testing.assert_index_equal(fit.s2_post.index, self.y.index)

    def test_finite_prior_df(self):
        # Same R reference as above, with residual standard deviations
        # c(0.25, 0.5, 1, 2, 4, 8) and trend=FALSE.
        scales = np.array([0.25, 0.5, 1, 2, 4, 8])
        y = self.y + (scales[:, None] - 1) * self.residuals
        fit = eBayes(lmFit(y, self.design))
        self.assertEqual(np.ndim(fit.df_prior), 0)
        self.assertAlmostEqual(fit.df_prior, 0.885883321947893, places=12)
        self.assertAlmostEqual(fit.s2_prior, 0.607715118902142, places=12)
        np.testing.assert_allclose(fit.coefficients, self.coefficients, atol=1e-14)
        np.testing.assert_allclose(fit.sigma**2, scales**2, rtol=1e-12)
        s2_post = np.array(
            [
                0.161355610927009,
                0.314859071934955,
                0.928872915966736,
                3.384928292093864,
                13.209149796602381,
                52.506035814636441,
            ]
        )
        np.testing.assert_allclose(fit.s2_post, s2_post, rtol=1e-12)
        np.testing.assert_allclose(fit.df_total, 4.885883321947893, rtol=1e-12)
        expected_t = self.coefficients / np.sqrt(
            s2_post[:, None] * np.array([1 / 3, 2 / 3])
        )
        np.testing.assert_allclose(fit.t, expected_t, atol=1e-12)
        np.testing.assert_allclose(
            fit.p_value,
            2 * t.sf(np.abs(expected_t), df=4.885883321947893),
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            fit.var_prior, [22.628407175229739, 0.043845016608588], rtol=1e-10
        )
        np.testing.assert_allclose(
            fit.lods,
            [
                [-2.249961847084895, -4.626967460267164],
                [-0.626709410956682, -4.535916831804467],
                [-1.249827704814880, -4.581505019119686],
                [-2.825752917320506, -4.578220064823750],
                [-4.491331562310333, -4.611479335669955],
                [-5.695258557124906, -4.626702369379813],
            ],
            rtol=1e-10,
        )

    def test_prior_df_shapes(self):
        # _ebayes supports feature-specific priors, although robust squeezeVar
        # is not implemented yet. Supply its hyperparameters directly to test
        # the kernels independently of prior estimation.
        mixed = np.array([3.0, np.inf, 10.0, 1e6, 1e6 + 1, np.inf])
        priors = [
            8.0,
            np.float64(8),
            np.array(8.0),
            np.array([8.0]),
            float("inf"),
            np.float64("inf"),
            np.array(np.inf),
            np.array([np.inf]),
            np.full(6, 8.0),
            np.full(6, np.inf),
            pd.Series([8.0]),
            pd.Series([np.inf]),
            mixed,
            pd.Series(mixed, index=self.y.index),
        ]
        for df_prior in priors:
            with self.subTest(df_prior=df_prior):
                fit = lmFit(self.y, self.design)
                # Different standard errors in each row/column expose accidental
                # column masking or pandas label alignment.
                fit.stdev_unscaled.iloc[:, :] = (
                    np.arange(12).reshape(6, 2) + 1
                ) / 10
                prior = np.broadcast_to(np.asarray(df_prior), (6,))
                s2_post = _squeezeVar(fit.sigma**2, fit.df_residual, 1.5, prior)
                with (
                    patch(
                        "inmoose.limma.ebayes.squeezeVar",
                        return_value={
                            "df_prior": df_prior,
                            "var_prior": 1.5,
                            "var_post": s2_post,
                        },
                    ),
                    # Hold coefficient priors fixed to isolate the B kernel.
                    patch(
                        "inmoose.limma.ebayes.tmixture_matrix",
                        return_value=np.array([0.2, 0.7]),
                    ),
                ):
                    out = _ebayes(fit)
                np.testing.assert_array_equal(out["df_prior"], df_prior)
                expected_df = np.minimum(4 + prior, 24)
                np.testing.assert_array_equal(out["df_total"], expected_df)
                expected_t = self.coefficients / fit.stdev_unscaled.to_numpy()
                expected_t /= np.sqrt(s2_post[:, None])
                np.testing.assert_allclose(out["t"], expected_t, atol=1e-12)
                np.testing.assert_allclose(
                    out["p_value"],
                    2 * t.sf(np.abs(expected_t), df=expected_df[:, None]),
                    rtol=1e-10,
                )
                expected_lods = np.empty((6, 2))
                # Evaluate one feature/contrast at a time, avoiding the masking
                # and broadcasting operations under test.
                for i in range(6):
                    for j, var_prior in enumerate([0.2, 0.7]):
                        r = 1 + var_prior / fit.stdev_unscaled.iloc[i, j] ** 2
                        t2 = expected_t[i, j] ** 2
                        if prior[i] > 1e6:
                            kernel = t2 * (1 - 1 / r) / 2
                        else:
                            df = expected_df[i]
                            kernel = (1 + df) / 2 * np.log(
                                (t2 + df) / (t2 / r + df)
                            )
                        expected_lods[i, j] = (
                            np.log(0.01 / 0.99) - np.log(r) / 2 + kernel
                        )
                pd.testing.assert_frame_equal(
                    out["lods"],
                    pd.DataFrame(
                        expected_lods,
                        index=self.y.index,
                        columns=fit.coefficients.columns,
                    ),
                    rtol=1e-12,
                    atol=1e-12,
                )
