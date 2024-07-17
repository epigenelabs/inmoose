import unittest

import numpy as np

from inmoose.edgepy import (
    DGEList,
    dispCoxReid,
    maximizeInterpolant,
    movingAverageByCol,
    systematicSubset,
)
from inmoose.utils import rnbinom


class test_dispersion(unittest.TestCase):
    def setUp(self):
        y = np.array(rnbinom(80, size=5, mu=20, seed=42)).reshape((20, 4))
        y = np.vstack(([0, 0, 0, 0], [0, 0, 2, 2], y))
        self.group = np.array([1, 1, 2, 2])
        self.d = DGEList(counts=y, group=self.group, lib_size=np.arange(1001, 1005))

    def test_maximizeInterpolant(self):
        spline_pts = np.array([0, 1, 3, 7])
        with self.assertRaisesRegex(
            ValueError, expected_regex="y is not a matrix: cannot perform interpolation"
        ):
            maximizeInterpolant(spline_pts, np.array([1, 2, 3]))
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="number of columns must equal number of spline points",
        ):
            maximizeInterpolant(spline_pts, np.ones((2, 3)))
        with self.assertRaisesRegex(
            ValueError, expected_regex="spline points must be unique and sorted"
        ):
            maximizeInterpolant([3, 2, 1], np.ones((2, 3)))
        with self.assertRaisesRegex(
            ValueError, expected_regex="spline points must be unique and sorted"
        ):
            maximizeInterpolant([1, 2, 2], np.ones((2, 3)))

        ref = [
            0.0000000,
            5.2951564,
            0.0000000,
            1.2921216,
            1.6173398,
            5.3493726,
            5.1489371,
            1.6854523,
            5.0732169,
            7.0000000,
            0.0000000,
            1.3839820,
            0.0000000,
            0.7054761,
            0.0000000,
            5.2193349,
            1.0046347,
            1.6395426,
            5.0557785,
            4.8235934,
            7.0000000,
            4.3981600,
        ]
        interpolation = maximizeInterpolant(spline_pts, self.d.counts)
        self.assertTrue(np.allclose(interpolation, ref, atol=1e-6, rtol=0))

    def test_systematicSubset(self):
        res = systematicSubset(3, np.arange(1, 10))
        self.assertTrue(np.array_equal(res + 1, [2, 5, 8]))
        res = systematicSubset(1000, np.arange(1, 10))
        self.assertTrue(np.array_equal(res + 1, [1, 2, 3, 4, 5, 6, 7, 8, 9]))
        res = systematicSubset(3, np.array([1, 1, 3, 2, 6, 5, 2, 5, 7]))
        self.assertTrue(np.array_equal(res + 1, [2, 3, 5]))

    def test_dispCoxReid(self):
        # TODO also test with varying tolerance
        self.assertAlmostEqual(
            dispCoxReid(self.d.counts, tol=1e-15), 0.158033594104552, 7
        )
        self.assertAlmostEqual(
            dispCoxReid(self.d.counts, subset=5, tol=1e-15), 0.091028284006027, 7
        )

        with self.assertRaisesRegex(
            ValueError, expected_regex="no data rows with required number of counts"
        ):
            dispCoxReid(self.d.counts, min_row_sum=1000)
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="please give a non-negative interval for the dispersion",
        ):
            dispCoxReid(self.d.counts, interval=(-1, 4))

    @unittest.skip("TODO")
    def test_dispCoxReidInterpolateTagwise(self):
        # TODO
        self.assertTrue(False)

    def test_movingAverageByCol(self):
        y = self.d.counts
        self.assertTrue(np.array_equal(movingAverageByCol(y, width=1), y))
        with self.assertLogs("inmoose", level="WARNING") as logChecker:
            res = movingAverageByCol(y, width=23)
        self.assertRegex(
            logChecker.output[0], "reducing moving average width to x.shape\[0\]"
        )
        ref = np.array(
            [
                [13.25000, 17.41667, 17.41667, 17.75000],
                [14.84615, 17.76923, 16.61538, 18.61538],
                [15.92857, 18.78571, 16.64286, 18.21429],
                [16.60000, 18.53333, 17.06667, 18.73333],
                [16.81250, 18.93750, 18.37500, 19.93750],
                [16.82353, 19.00000, 18.05882, 19.64706],
                [16.38889, 20.61111, 19.00000, 19.00000],
                [17.68421, 20.63158, 20.21053, 18.89474],
                [18.75000, 21.25000, 21.20000, 19.20000],
                [18.85714, 21.19048, 21.23810, 19.85714],
                [18.77273, 21.09091, 21.31818, 19.77273],
                [19.66667, 22.09524, 22.33333, 20.71429],
                [20.65000, 23.20000, 23.35000, 21.65000],
                [20.42105, 23.42105, 24.00000, 21.57895],
                [21.00000, 23.77778, 24.77778, 21.83333],
                [21.41176, 23.11765, 24.64706, 23.00000],
                [21.81250, 24.12500, 24.43750, 22.62500],
                [21.93333, 24.73333, 23.86667, 22.46667],
                [23.00000, 24.00000, 23.57143, 22.07143],
                [23.61538, 24.76923, 23.92308, 22.53846],
                [25.16667, 24.83333, 24.08333, 21.91667],
                [24.27273, 25.18182, 25.00000, 21.90909],
            ]
        )
        self.assertTrue(np.allclose(res, ref, atol=0, rtol=1e-6))

        res = movingAverageByCol(y)
        ref = np.array(
            [
                [8.333333, 6.333333, 4.333333, 8.333333],
                [8.750000, 9.000000, 5.750000, 10.500000],
                [9.800000, 14.200000, 10.000000, 8.800000],
                [12.800000, 15.600000, 15.600000, 14.600000],
                [16.800000, 18.600000, 21.800000, 19.200000],
                [13.200000, 21.800000, 25.200000, 20.200000],
                [14.200000, 21.200000, 27.000000, 20.000000],
                [12.400000, 19.000000, 26.000000, 25.600000],
                [16.400000, 21.800000, 23.200000, 24.200000],
                [15.000000, 23.200000, 19.600000, 23.000000],
                [20.400000, 20.600000, 15.400000, 23.200000],
                [23.400000, 24.200000, 15.000000, 22.600000],
                [27.600000, 22.400000, 15.200000, 21.800000],
                [24.600000, 23.200000, 20.000000, 25.000000],
                [25.400000, 22.800000, 19.600000, 24.200000],
                [20.400000, 28.000000, 25.200000, 20.000000],
                [22.600000, 25.800000, 30.200000, 20.800000],
                [25.200000, 29.400000, 33.600000, 20.600000],
                [25.400000, 28.400000, 30.400000, 19.600000],
                [25.400000, 28.200000, 32.400000, 20.200000],
                [29.500000, 23.250000, 31.750000, 23.250000],
                [25.666667, 24.000000, 28.333333, 25.333333],
            ]
        )
        self.assertTrue(np.allclose(res, ref, atol=0, rtol=1e-7))

        res = movingAverageByCol(y, full_length=False)
        ref = np.array(
            [
                [9.8, 14.2, 10.0, 8.8],
                [12.8, 15.6, 15.6, 14.6],
                [16.8, 18.6, 21.8, 19.2],
                [13.2, 21.8, 25.2, 20.2],
                [14.2, 21.2, 27.0, 20.0],
                [12.4, 19.0, 26.0, 25.6],
                [16.4, 21.8, 23.2, 24.2],
                [15.0, 23.2, 19.6, 23.0],
                [20.4, 20.6, 15.4, 23.2],
                [23.4, 24.2, 15.0, 22.6],
                [27.6, 22.4, 15.2, 21.8],
                [24.6, 23.2, 20.0, 25.0],
                [25.4, 22.8, 19.6, 24.2],
                [20.4, 28.0, 25.2, 20.0],
                [22.6, 25.8, 30.2, 20.8],
                [25.2, 29.4, 33.6, 20.6],
                [25.4, 28.4, 30.4, 19.6],
                [25.4, 28.2, 32.4, 20.2],
            ]
        )
        self.assertTrue(np.array_equal(res, ref))

        res = movingAverageByCol(y, width=22, full_length=False)
        ref = np.array([18.77273, 21.09091, 21.31818, 19.77273])
        self.assertTrue(np.allclose(res, ref, atol=0, rtol=1e-6))

    def test_estimateGLMCommonDisp(self):
        e = self.d.estimateGLMCommonDisp()
        self.assertAlmostEqual(e.common_dispersion, 0.16157151, 5)

    def test_estimateGLMTagwiseDisp(self):
        # first initialize d.common_dispersion
        self.d.estimateGLMCommonDisp()
        e = self.d.estimateGLMTagwiseDisp()
        ref = [
            0.1615715,
            0.1615715,
            0.1391734,
            0.1337411,
            0.3107981,
            0.1923997,
            0.1378647,
            0.1995537,
            0.1175669,
            0.2083242,
            0.1458728,
            0.1251945,
            0.1974399,
            0.1553991,
            0.1252869,
            0.1348553,
            0.1209722,
            0.2864446,
            0.1629382,
            0.1191966,
            0.1243685,
            0.1157305,
        ]
        self.assertTrue(np.allclose(e.tagwise_dispersion, ref, atol=1e-6, rtol=0))


if __name__ == "__main__":
    unittest.main()
