import unittest

import numpy as np

from inmoose.limma import tmixture_matrix, tmixture_vector


class Test(unittest.TestCase):
    def test_tmixture_matrix(self):
        self.assertTrue(
            np.allclose(
                tmixture_matrix(
                    np.arange(1, 7).reshape(2, 3),
                    np.arange(2, 8).reshape(2, 3),
                    np.arange(3, 5),
                    0.5,
                ),
                np.array([648.387, 1544.773, 3105.003]),
            )
        )

    def test_tmixture_vector(self):
        self.assertAlmostEqual(
            tmixture_vector(
                np.arange(1, 7),
                np.arange(2, 8),
                np.arange(3, 9),
                0.5,
            ),
            1124.83,
            places=2,
        )
