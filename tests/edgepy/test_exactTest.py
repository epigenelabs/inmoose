import unittest

import numpy as np
import pandas as pd

from inmoose.edgepy import DGEList, binomTest, exactTest, exactTestBetaApprox, topTags
from inmoose.utils import rnbinom


class Test(unittest.TestCase):
    def setUp(self):
        y = np.array(rnbinom(80, size=5, mu=20, seed=42)).reshape((20, 4))
        y = np.vstack(([0, 0, 0, 0], [0, 0, 2, 2], y))
        self.group = np.array([1, 2, 2, 2])
        self.d = DGEList(counts=y, group=self.group)
        self.d = self.d.estimateGLMCommonDisp()

    def test_exactTest_doubletail(self):
        e = exactTest(self.d, rejection_region="doubletail")
        table_ref = pd.DataFrame(
            {
                "log2FoldChange": [
                    0.00000000,
                    3.51533881,
                    -0.63058208,
                    0.41122894,
                    0.43984489,
                    0.37139369,
                    0.13977308,
                    1.95460401,
                    -0.01979911,
                    2.17873153,
                    -1.01346677,
                    0.37844670,
                    -0.93741820,
                    -0.68416066,
                    -0.41992156,
                    0.60973257,
                    -0.22824454,
                    1.57918440,
                    -0.77088316,
                    -0.40296414,
                    0.11776420,
                    0.08922781,
                ],
                "lfcSE": [
                    4.879406,
                    4.343170,
                    0.748468,
                    0.831304,
                    0.786678,
                    0.780016,
                    0.755259,
                    0.868254,
                    0.786410,
                    0.934109,
                    0.730749,
                    0.796331,
                    0.731355,
                    0.735686,
                    0.741946,
                    0.750018,
                    0.776667,
                    0.830651,
                    0.717838,
                    0.716268,
                    0.751390,
                    0.771176,
                ],
                "logCPM": [
                    12.12019,
                    12.70317,
                    15.56153,
                    15.07428,
                    15.52507,
                    15.56517,
                    15.77696,
                    15.83033,
                    15.29041,
                    15.58364,
                    15.78337,
                    15.38442,
                    15.78799,
                    15.76965,
                    15.74505,
                    16.12701,
                    15.31314,
                    15.84442,
                    16.13715,
                    16.30238,
                    15.82620,
                    15.52722,
                ],
                "pvalue": [
                    1.00000000,
                    0.67966025,
                    0.37574431,
                    0.77735744,
                    0.69008766,
                    0.72780471,
                    0.99603624,
                    0.04941292,
                    1.00000000,
                    0.02552397,
                    0.14116664,
                    0.74391410,
                    0.16836537,
                    0.33562846,
                    0.55601398,
                    0.52617216,
                    0.79259676,
                    0.09778055,
                    0.25246831,
                    0.53871925,
                    0.99845052,
                    1.00000000,
                ],
            },
            index=[f"gene{i}" for i in range(22)],
        )
        pd.testing.assert_frame_equal(table_ref, e, check_frame_type=False)

    @unittest.skip("R returns NAs, need to find meaningful test data")
    def test_exactTest_deviance(self):
        e = exactTest(self.d, rejection_region="deviance")
        table_ref = pd.DataFrame(
            {
                "logFC": [
                    0.00000000,
                    3.51533881,
                    -0.63058208,
                    0.41122894,
                    0.43984489,
                    0.37139369,
                    0.13977308,
                    1.95460401,
                    -0.01979911,
                    2.17873153,
                    -1.01346677,
                    0.37844670,
                    -0.93741820,
                    -0.68416066,
                    -0.41992156,
                    0.60973257,
                    -0.22824454,
                    1.57918440,
                    -0.77088316,
                    -0.40296414,
                    0.11776420,
                    0.08922781,
                ],
                "logCPM": [
                    12.12019,
                    12.70317,
                    15.56153,
                    15.07428,
                    15.52507,
                    15.56517,
                    15.77696,
                    15.83033,
                    15.29041,
                    15.58364,
                    15.78337,
                    15.38442,
                    15.78799,
                    15.76965,
                    15.74505,
                    16.12701,
                    15.31314,
                    15.84442,
                    16.13715,
                    16.30238,
                    15.82620,
                    15.52722,
                ],
                "PValue": [
                    1.00000,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
            }
        )
        self.assertTrue(np.allclose(table_ref["logFC"], e["logFC"], atol=1e-6, rtol=0))
        self.assertTrue(
            np.allclose(table_ref["logCPM"], e["logCPM"], atol=1e-5, rtol=0)
        )
        self.assertTrue(
            np.allclose(table_ref["PValue"], e["PValue"], atol=1e-6, rtol=0)
        )

    def test_exactTest_smallp(self):
        e = exactTest(self.d, rejection_region="smallp")
        table_ref = pd.DataFrame(
            {
                "log2FoldChange": [
                    0.00000000,
                    3.51533881,
                    -0.63058208,
                    0.41122894,
                    0.43984489,
                    0.37139369,
                    0.13977308,
                    1.95460401,
                    -0.01979911,
                    2.17873153,
                    -1.01346677,
                    0.37844670,
                    -0.93741820,
                    -0.68416066,
                    -0.41992156,
                    0.60973257,
                    -0.22824454,
                    1.57918440,
                    -0.77088316,
                    -0.40296414,
                    0.11776420,
                    0.08922781,
                ],
                "lfcSE": [
                    4.879406,
                    4.343170,
                    0.748468,
                    0.831304,
                    0.786678,
                    0.780016,
                    0.755259,
                    0.868254,
                    0.786410,
                    0.934109,
                    0.730749,
                    0.796331,
                    0.731355,
                    0.735686,
                    0.741946,
                    0.750018,
                    0.776667,
                    0.830651,
                    0.717838,
                    0.716268,
                    0.751390,
                    0.771176,
                ],
                "logCPM": [
                    12.12019,
                    12.70317,
                    15.56153,
                    15.07428,
                    15.52507,
                    15.56517,
                    15.77696,
                    15.83033,
                    15.29041,
                    15.58364,
                    15.78337,
                    15.38442,
                    15.78799,
                    15.76965,
                    15.74505,
                    16.12701,
                    15.31314,
                    15.84442,
                    16.13715,
                    16.30238,
                    15.82620,
                    15.52722,
                ],
                "pvalue": [
                    1.00000000,
                    0.03903554,
                    0.03903554,
                    0.03903554,
                    0.03903554,
                    0.03903554,
                    0.03903554,
                    0.03903554,
                    0.03903554,
                    0.03903554,
                    0.03903554,
                    0.03903554,
                    0.03903554,
                    0.03903554,
                    0.03903554,
                    0.03903554,
                    0.03903554,
                    0.03903554,
                    0.03903554,
                    0.03903554,
                    0.03903554,
                    0.03903554,
                ],
            },
            index=[f"gene{i}" for i in range(22)],
        )
        pd.testing.assert_frame_equal(table_ref, e, check_frame_type=False)

    def test_exactTestBetaApprox(self):
        pref = [6.988452e-145, 5.981632e-124]
        pval = exactTestBetaApprox(
            np.array([[1000, 1200], [1300, 1400]]),
            np.array([[2000, 2200], [2300, 2400]]),
        )
        self.assertTrue(np.allclose(pref, pval, atol=1e-6, rtol=0))

    def test_binomTest(self):
        pval = binomTest(self.d.counts.iloc[:, 0], self.d.counts.iloc[:, 1])
        pref = [
            1.000000e00,
            1.000000e00,
            2.275075e-01,
            3.385877e-01,
            9.734914e-03,
            5.510056e-02,
            2.413184e-01,
            6.708009e-05,
            7.109008e-01,
            1.204722e-03,
            2.277012e-02,
            3.097011e-01,
            4.480458e-02,
            8.989952e-01,
            4.168068e-02,
            7.667345e-01,
            1.000000e00,
            8.761834e-07,
            3.168496e-03,
            2.395773e-01,
            6.406607e-01,
            1.000000e00,
        ]
        self.assertTrue(np.allclose(pval, pref, atol=1e-6, rtol=0))

    def test_topTags(self):
        t = topTags(exactTest(self.d))
        self.assertListEqual(
            list(t.index),
            [
                "gene9",
                "gene7",
                "gene17",
                "gene10",
                "gene12",
                "gene18",
                "gene13",
                "gene2",
                "gene15",
                "gene19",
            ],
        )

        self.assertListEqual(
            list(t.columns),
            [
                "log2FoldChange",
                "lfcSE",
                "logCPM",
                "pvalue",
                "FDR",
                "adj_pvalue",
            ],
        )
