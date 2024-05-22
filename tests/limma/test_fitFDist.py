import unittest

import numpy as np

from inmoose.limma.fitFDist import trigammaInverse


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
