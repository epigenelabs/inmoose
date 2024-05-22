import unittest

import numpy as np

from inmoose.utils import cov2cor


class Test(unittest.TestCase):
    def test_cov2cor(self):
        M = np.arange(25).reshape((5, 5))
        cov = np.cov(M)
        cor = np.corrcoef(M)
        self.assertTrue(np.allclose(cov2cor(cov), cor))
