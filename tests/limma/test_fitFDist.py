import unittest

import numpy as np

from inmoose.limma.fitFDist import fitFDist, trigammaInverse


class Test(unittest.TestCase):
    def test_trigammainverse(self):
        res = trigammaInverse(np.array([np.nan, 0, 1e8, -1, 5]))
        self.assertTrue(
            np.allclose(
                res,
                [np.nan, np.inf, 0.0001000, np.nan, 0.4961687],
                equal_nan=True,
                atol=1e-5,
            )
        )

    def test_fitfdist_nok_zero(self):
        # When all observations are invalid/negative/NaN, nok=0.
        # Ensure scale and df2 return NaN without RuntimeWarnings or spurious inf.
        res = fitFDist(np.array([np.nan, -1.0]), df1=np.array([2.0, 2.0]))
        self.assertTrue(np.isnan(res["scale"]))
        self.assertTrue(np.isnan(res["df2"]))

    def test_fitfdist_nok_one_covariate(self):
        # When only 1 observation is valid and a covariate is provided,
        # ensure scale returns an array of shape (n,) consistent with docstring.
        res = fitFDist(
            np.array([np.nan, 3.5]),
            df1=np.array([2.0, 2.0]),
            covariate=np.array([10.0, 20.0]),
        )
        self.assertEqual(res["df2"], 0)
        self.assertEqual(len(res["scale"]), 2)
        self.assertTrue(np.allclose(res["scale"], [3.5, 3.5]))

