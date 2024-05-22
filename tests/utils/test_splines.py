import unittest

import numpy as np

from inmoose.utils import ns, spline_design


class Test(unittest.TestCase):
    def test_splineDesign(self):
        spline = spline_design(np.arange(10), [4.5, 5, 5.5], 4)
        self.assertTrue(
            np.allclose(
                spline,
                np.array(
                    [
                        [0, 0.02083333, 0.47916667, 0.4791667, 0.02083333, 0.0],
                        [0, 0.00000000, 0.16666667, 0.6666667, 0.16666667, 0.00000000],
                        [0, 0.00000000, 0.02083333, 0.4791667, 0.47916667, 0.02083333],
                    ]
                ),
            )
        )

    def test_splines(self):
        res = ns([1, 2, 3, 4, 5, 6], df=3, include_intercept=True)
        self.assertTrue(
            np.allclose(
                res.basis,
                [
                    [-0.45528153, 0.5915984, -0.39439896],
                    [0.04538466, 0.4796729, -0.30911524],
                    [0.39866939, 0.3936580, -0.17710530],
                    [0.46333206, 0.3583848, 0.04641015],
                    [0.23937270, 0.3738533, 0.36143111],
                    [-0.13196814, 0.4152326, 0.72317830],
                ],
            )
        )

        res2 = res.predict(newx=[70, 81, 92])
        self.assertTrue(
            np.allclose(
                res2.basis,
                [
                    [-25.46985, 3.339883, 24.37341],
                    [-29.82480, 3.842558, 28.43829],
                    [-34.17975, 4.345232, 32.50318],
                ],
            )
        )
