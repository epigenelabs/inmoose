import unittest

import numpy as np

from inmoose.limma import classifyTestsF


class Test(unittest.TestCase):
    def test_classifyTestsF(self):
        M = np.arange(1, 13).reshape((3, 4))
        self.assertTrue(
            np.array_equal(
                classifyTestsF(M), [[0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
            )
        )
        self.assertTrue(
            np.array_equal(
                classifyTestsF(M / 2), [[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]]
            )
        )
        self.assertTrue(
            np.array_equal(
                classifyTestsF(M / 3), [[0, 0, 0, 0], [0, 1, 1, 1], [1, 1, 1, 1]]
            )
        )
        self.assertTrue(
            np.array_equal(
                classifyTestsF(M, df=3), [[0, 0, 0, 0], [0, 1, 1, 1], [1, 1, 1, 1]]
            )
        )
        self.assertTrue(
            np.array_equal(
                classifyTestsF(M, p_value=0.00001),
                [[0, 0, 0, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            )
        )

        fstat = classifyTestsF(M, fstat_only=True)
        self.assertTrue(np.array_equal(fstat, [7.5, 43.5, 111.5]))
        self.assertEqual(fstat.df1, 4)
        self.assertTrue(np.isposinf(fstat.df2))
