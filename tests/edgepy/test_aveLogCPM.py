import unittest

import numpy as np

from inmoose.edgepy import DGEList, aveLogCPM
from inmoose.utils import rnbinom


class test_aveLogCPM(unittest.TestCase):
    def setUp(self):
        y = np.array(rnbinom(80, size=5, mu=20, seed=42)).reshape((20, 4))
        y = np.vstack(([0, 0, 0, 0], [0, 0, 2, 2], y))
        group = np.array([1, 1, 2, 2])
        self.d = DGEList(y, group=group)

    def test_aveLogCPM(self):
        ref = np.array(
            [
                12.12019,
                12.70436,
                15.55656,
                15.07438,
                15.53215,
                15.56454,
                15.77740,
                15.83572,
                15.29030,
                15.58778,
                15.77746,
                15.38536,
                15.78073,
                15.76788,
                15.74191,
                16.12822,
                15.31202,
                15.85351,
                16.13554,
                16.30177,
                15.82458,
                15.52800,
            ]
        )
        self.assertTrue(np.allclose(self.d.aveLogCPM(), ref, atol=1e-5, rtol=0))

        self.assertTrue(
            np.allclose(self.d.aveLogCPM(dispersion=np.nan), ref, atol=1e-5, rtol=0)
        )
        dispersion = [0.05 for i in range(22)]
        dispersion[0] = np.nan
        self.assertTrue(
            np.allclose(self.d.aveLogCPM(dispersion=dispersion), ref, atol=1e-5, rtol=0)
        )

        self.d.samples.lib_size = None
        self.assertTrue(np.allclose(self.d.aveLogCPM(), ref, atol=1e-5, rtol=0))

        # Check special cases
        self.assertEqual(aveLogCPM(np.full((0, 0), 42)), 0)
        self.assertTrue(
            np.allclose(
                aveLogCPM(np.zeros(shape=(3, 2))),
                [18.34661, 18.34661, 18.34661],
                atol=1e-5,
                rtol=0,
            )
        )
        with self.assertRaisesRegex(ValueError, expected_regex="y should be a matrix"):
            aveLogCPM(42)
