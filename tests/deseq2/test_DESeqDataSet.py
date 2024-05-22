import unittest

import numpy as np
from pandas.api.types import CategoricalDtype

from inmoose.deseq2 import DESeqDataSet


class Test(unittest.TestCase):
    def test_counts(self):
        """test that normalized counts are properly computed"""
        dds = DESeqDataSet(
            np.arange(24).reshape(4, 6), clinicalData=["A", "A", "B", "B"], design="~1"
        )
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="first calculate size factors, add normalizationFactors, or set normalized=False",
        ):
            dds.counts(normalized=True)
        dds = dds.estimateSizeFactors()
        ref = np.array(
            [
                [0.000000, 3.201086, 6.402172, 9.603258, 12.80434, 16.00543],
                [6.402172, 7.469200, 8.536229, 9.603258, 10.67029, 11.73731],
                [7.682606, 8.322823, 8.963040, 9.603258, 10.24347, 10.88369],
                [8.231364, 8.688662, 9.145960, 9.603258, 10.06056, 10.51785],
            ]
        )
        res = dds.counts(normalized=True)
        self.assertTrue(np.allclose(res, ref))

    def test_design(self):
        """test that categorical variable in the design are properly accounted for"""
        dds = DESeqDataSet(np.arange(24).reshape(4, 6))
        dds.obs["x"] = ["A", "A", "B", "B"]
        dds.obs["y"] = [1, 2, 1, 2]
        dds.design = "x + y"
        self.assertTrue("C(x)" in dds.obs)
        self.assertTrue("C(y)" not in dds.obs)
        self.assertFalse(isinstance(dds.obs["x"].dtype, CategoricalDtype))
        self.assertTrue(isinstance(dds.obs["C(x)"].dtype, CategoricalDtype))
        self.assertFalse(isinstance(dds.obs["y"].dtype, CategoricalDtype))

        dds = DESeqDataSet(np.arange(24).reshape(4, 6))
        dds.obs["x"] = ["A", "A", "B", "B"]
        dds.obs["y"] = [1, 2, 1, 2]
        dds.design = "C(x) + C(y)"
        self.assertTrue("C(x)" in dds.obs)
        self.assertTrue("C(y)" in dds.obs)
        self.assertFalse(isinstance(dds.obs["x"].dtype, CategoricalDtype))
        self.assertTrue(isinstance(dds.obs["C(x)"].dtype, CategoricalDtype))
        self.assertFalse(isinstance(dds.obs["y"].dtype, CategoricalDtype))
        self.assertTrue(isinstance(dds.obs["C(y)"].dtype, CategoricalDtype))
