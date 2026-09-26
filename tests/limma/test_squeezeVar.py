import unittest

import numpy as np

from inmoose.limma import squeezeVar


class Test(unittest.TestCase):
    def test_squeezeVar(self):
        with self.assertRaisesRegex(ValueError, expected_regex="var is empty"):
            squeezeVar(np.array([]), df=3)

        res = squeezeVar(np.array([1]), df=3)
        self.assertEqual(res["df_prior"], 0)
        self.assertTrue(np.array_equal(res["var_prior"], [1]))
        self.assertTrue(np.array_equal(res["var_post"], [1]))

        res = squeezeVar(np.array([0.5, 1.3, 7.8]), df=3)
        self.assertAlmostEqual(res["df_prior"], 2.830777, places=6)
        self.assertAlmostEqual(res["var_prior"], 1.676927, places=6)
        self.assertTrue(
            np.allclose(res["var_post"], [1.071385, 1.482994, 4.827317], atol=1e-6)
        )

        res = squeezeVar(
            np.array([0.5, 0.5, 1.3, 1.3, 7.8, 7.8]),
            df=20,
            covariate=[1, 2, 3, 4, 5, 6],
        )
        self.assertAlmostEqual(res["df_prior"], 15.44626, places=5)
        self.assertTrue(
            np.allclose(
                res["var_prior"],
                [0.4322342, 0.6426360, 1.0331707, 1.9359150, 4.2277336, 9.9836662],
                atol=1e-6,
            )
        )
        self.assertTrue(
            np.allclose(
                res["var_post"],
                [0.4704700, 0.5621558, 1.1837250, 1.5771098, 6.2433295, 8.7515664],
                atol=1e-6,
            )
        )

    def test_squeezeVar_zero_residual_df(self):
        # When some genes have df=0 and df_prior evaluates to 0, total df is 0.
        # Ensure no zero-division / NaNs occur and fallback to var_prior is respected.
        res = squeezeVar(np.array([1.5, 2.5]), df=np.array([0.0, 3.0]))
        self.assertFalse(np.isnan(res["var_post"]).any())
        self.assertTrue(np.allclose(res["var_post"], [2.5, 2.5]))

