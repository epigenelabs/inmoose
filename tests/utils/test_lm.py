import unittest

import numpy as np

from inmoose.utils import lm_fit, lm_wfit


class Test(unittest.TestCase):
    def assertArrayEqual(self, x, y, tol=1e-6):
        self.assertTrue(np.allclose(x, y, atol=tol))

    def test_lm_fit(self):
        res = lm_fit(
            np.arange(5, 11).reshape(3, 2), np.array([1.0 / 30, 1.0 / 31, 1.0 / 32])
        )
        self.assertArrayEqual(res.coefficients, np.array([-0.03644713, 0.03592630]))
        self.assertArrayEqual(
            res.residuals, [1.120072e-05, -2.240143e-05, 1.120072e-05]
        )
        self.assertArrayEqual(res.fitted_values, [0.03332213, 0.03228047, 0.03123880])
        self.assertArrayEqual(res.effects, [-5.411474e-02, 1.413684e-02, 2.743604e-05])
        self.assertEqual(res.rank, 2)
        self.assertArrayEqual(res.df_residuals, [1])

    def test_lm_wfit(self):
        res = lm_wfit(
            np.arange(5, 11).reshape(3, 2),
            np.array([1.0 / 30, 1.0 / 31, 1.0 / 32]),
            np.array([100, 50, 1]),
        )
        self.assertArrayEqual(res.coefficients, np.array([-0.03654927, 0.03601318]))
        self.assertArrayEqual(
            res.residuals, [6.165532e-07, -2.466213e-06, 6.165532e-05]
        )
        self.assertArrayEqual(res.fitted_values, [0.03333272, 0.03226053, 0.03118834])
        self.assertArrayEqual(res.effects, [-3.981168e-01, 7.496570e-02, 6.437005e-05])
        self.assertEqual(res.rank, 2)
        self.assertArrayEqual(res.df_residuals, [1])
