import unittest

import numpy as np

from inmoose.edgepy import nbinomDeviance
from inmoose.utils import rnbinom


class test_nbinomDeviance(unittest.TestCase):
    def setUp(self):
        y = np.array(rnbinom(80, size=5, mu=20, seed=42)).reshape((20, 4))
        self.y = np.vstack(([0, 0, 0, 0], [0, 0, 2, 2], y))

    def test_nbinomDeviance(self):
        dev_ref = np.array(
            [
                7.999999,
                5.545177,
                321.818901,
                184.761910,
                355.515748,
                340.390858,
                414.784086,
                461.324010,
                235.746631,
                354.718673,
                410.643778,
                265.824737,
                422.344282,
                412.896876,
                392.316278,
                599.686176,
                242.088733,
                501.330767,
                608.676351,
                706.581453,
                430.474599,
                310.504051,
            ]
        )
        dev = nbinomDeviance(self.y, np.ones(self.y.shape))
        self.assertTrue(np.allclose(dev, dev_ref, atol=0, rtol=10 - 9))

        dev_ref = np.array([8.454617])
        dev = nbinomDeviance(np.array([1, 2, 3, 4]), np.ones(4), dispersion=np.zeros(4))
        self.assertTrue(np.allclose(dev, dev_ref, atol=0, rtol=10 - 9))

        with self.assertRaisesRegex(
            ValueError, expected_regex="y is a matrix but mean is not"
        ):
            nbinomDeviance(self.y, np.ones((22,)))

        with self.assertRaisesRegex(
            ValueError, expected_regex="mean is a matrix but y is not"
        ):
            nbinomDeviance(np.array([1, 2, 3, 4]), np.ones((4, 3)))

        with self.assertRaisesRegex(
            ValueError, expected_regex="length of mean differs from that of y"
        ):
            nbinomDeviance(np.array([1, 2, 3, 4]), np.ones(3))
