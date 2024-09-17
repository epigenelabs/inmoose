import unittest

import numpy as np
import pandas as pd

from inmoose.edgepy import DGEList, validDGEList


class test_utils(unittest.TestCase):
    def test_isAllZero(self):
        from inmoose.edgepy.utils import _isAllZero

        self.assertEqual(_isAllZero(np.array([])), False)
        self.assertEqual(_isAllZero(pd.DataFrame(np.array([]))), False)
        self.assertEqual(_isAllZero(np.array([0, 0, 0])), True)
        self.assertEqual(_isAllZero(pd.DataFrame(np.array([0, 0, 0]))), True)
        self.assertEqual(_isAllZero(np.array([2, 1, 0])), False)
        self.assertEqual(_isAllZero(pd.DataFrame(np.array([2, 1, 0]))), False)
        with self.assertRaisesRegex(
            ValueError, expected_regex="NaN counts are not allowed"
        ):
            _isAllZero(np.array([1, 2, 3, np.nan, 4, 5]))
        with self.assertRaisesRegex(
            ValueError, expected_regex="NaN counts are not allowed"
        ):
            _isAllZero(pd.DataFrame(np.array([1, 2, 3, np.nan, 4, 5])))
        with self.assertRaisesRegex(
            ValueError, expected_regex="infinite counts are not allowed"
        ):
            _isAllZero(np.array([1, 2, 3, np.inf, 4, 5]))
        with self.assertRaisesRegex(
            ValueError, expected_regex="infinite counts are not allowed"
        ):
            _isAllZero(pd.DataFrame(np.array([1, 2, 3, np.inf, 4, 5])))
        with self.assertRaisesRegex(
            ValueError, expected_regex="negative counts are not allowed"
        ):
            _isAllZero(np.array([1, 2, 3, -np.inf, 4, 5]))
        with self.assertRaisesRegex(
            ValueError, expected_regex="negative counts are not allowed"
        ):
            _isAllZero(pd.DataFrame(np.array([1, 2, 3, -np.inf, 4, 5])))
        with self.assertRaisesRegex(
            ValueError, expected_regex="negative counts are not allowed"
        ):
            _isAllZero(np.array([1, 2, 3, -42, 4, 5]))
        with self.assertRaisesRegex(
            ValueError, expected_regex="negative counts are not allowed"
        ):
            _isAllZero(pd.DataFrame(np.array([1, 2, 3, -42, 4, 5])))


class test_DGEList(unittest.TestCase):
    def test_constructor(self):
        with self.assertRaisesRegex(
            ValueError, expected_regex="'counts' is not a matrix"
        ):
            d = DGEList([1, 2])
        with self.assertRaisesRegex(
            ValueError, expected_regex="non-numeric values found in 'counts'"
        ):
            d = DGEList(None)
        with self.assertRaisesRegex(
            ValueError, expected_regex="non-numeric values found in 'counts'"
        ):
            d = DGEList([["foobar"]])
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="length of 'lib_size' must be equal to the number of columns in 'counts'",
        ):
            d = DGEList([[42]], lib_size=[1, 2])
        with self.assertRaisesRegex(
            ValueError, expected_regex="negative library size not permitted"
        ):
            d = DGEList([[42]], lib_size=[-1])
        with self.assertRaisesRegex(
            ValueError, expected_regex="library size set to zero but counts are nonzero"
        ):
            d = DGEList([[42]], lib_size=[0])
        with self.assertLogs("inmoose", level="WARNING") as logChecker:
            d = DGEList([[0]], lib_size=[0])
        self.assertRegex(logChecker.output[0], "library size of zero detected")

        d = DGEList([[42]])
        self.assertTrue(isinstance(d.counts, pd.DataFrame))
        self.assertIsNotNone(d.samples)
        self.assertIsNone(d.common_dispersion)
        self.assertIsNone(d.trended_dispersion)
        self.assertIsNone(d.tagwise_dispersion)
        self.assertIsNone(d.design)
        self.assertIsNone(d.weights)
        self.assertIsNone(d.offset)
        self.assertIsNone(d.genes)
        self.assertIsNone(d.prior_df)
        self.assertIsNone(d.AveLogCPM)

    def test_getOffset(self):
        d = DGEList([[42]])
        self.assertTrue(np.allclose(d.getOffset(), 3.73767, atol=1e-6, rtol=0))
        d.offset = 0
        self.assertEqual(d.getOffset(), 0)

    def test_getDispersion(self):
        d = DGEList([[42]])
        self.assertIsNone(d.getDispersion())
        d.common_dispersion = "common"
        self.assertEqual(d.getDispersion(), "common")
        d.trended_dispersion = "trended"
        self.assertEqual(d.getDispersion(), "trended")
        d.tagwise_dispersion = "tagwise"
        self.assertEqual(d.getDispersion(), "tagwise")

    def test_validDGEList(self):
        d = DGEList([[42]])
        d.counts = None
        with self.assertRaisesRegex(RuntimeError, expected_regex="No count matrix"):
            validDGEList(d)
        d = DGEList([[42]])
        d = validDGEList(d)
        self.assertTrue(isinstance(d.counts, pd.DataFrame))
        self.assertFalse((d.samples.group.values == None).any())  # noqa: E711
        self.assertFalse((d.samples.lib_size.values == None).any())  # noqa: E711
        self.assertFalse((d.samples.norm_factors.values == None).any())  # noqa: E711
