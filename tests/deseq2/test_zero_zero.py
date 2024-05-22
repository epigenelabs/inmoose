import unittest

import numpy as np

from inmoose.deseq2 import DESeq, makeExampleDESeqDataSet
from inmoose.utils import Factor


class zero_zero(unittest.TestCase):
    def test_zero_zero(self):
        """test that contrast of two groups with all zeros - LFC zeroed out"""

        # test comparison of two groups with all zeros
        dds = makeExampleDESeqDataSet(
            m=8, n=100, sizeFactors=[1, 1, 0.5, 0.5, 1, 1, 2, 2]
        )
        dds.obs["condition"] = Factor(np.repeat(["A", "B", "C", "D"], 2))
        dds.design = "~condition"
        dds.counts()[:, 0] = [100, 110, 0, 0, 100, 110, 0, 0]
        dds.counts()[:, 1] = np.repeat(0, 8)
        dds = DESeq(dds)

        res = dds.results(contrast=["condition", "D", "B"])
        self.assertEqual(res.log2FoldChange.iloc[0], 0)
        res = dds.results(contrast=[0, -1, 0, 1])
        self.assertEqual(res.log2FoldChange.iloc[0], 0)
        res = dds.results(contrast=["condition_D_vs_A", "condition_B_vs_A"])
        self.assertEqual(res.log2FoldChange.iloc[0], 0)

        res = dds.results(name="condition_D_vs_A")
        self.assertNotEqual(res.log2FoldChange.iloc[0], 0)
        res = dds.results([0, 0, 0, 1])
        self.assertNotEqual(res.log2FoldChange.iloc[0], 0)

        # if all samples have 0, should be NA
        res = dds.results(contrast=["condition", "D", "B"])
        self.assertTrue(np.isnan(res.log2FoldChange.iloc[1]))
        res = dds.results(contrast=[0, -1, 0, 1])
        self.assertTrue(np.isnan(res.log2FoldChange.iloc[1]))
