import unittest

import numpy as np

from inmoose.deseq2 import DESeq, makeExampleDESeqDataSet, nbinomWaldTest
from inmoose.utils import Factor


class Test(unittest.TestCase):
    def test_addMLE(self):
        """test that adding MLE works as expected"""
        dds = makeExampleDESeqDataSet(n=200, m=12, betaSD=1, seed=42)
        dds.obs["condition"] = Factor(np.repeat(["A", "B", "C"], 4))
        dds.design = "~condition"
        dds = DESeq(dds, betaPrior=True)
        ddsNP = nbinomWaldTest(dds.copy(), betaPrior=False)

        res1 = dds.results(contrast=["condition", "C", "A"], addMLE=True)
        res2 = ddsNP.results(contrast=["condition", "C", "A"])
        self.assertTrue(np.allclose(res1.lfcMLE, res2.log2FoldChange, equal_nan=True))

        res1 = dds.results(contrast=["condition", "A", "B"], addMLE=True)
        res2 = ddsNP.results(contrast=["condition", "A", "B"])
        self.assertTrue(np.allclose(res1.lfcMLE, res2.log2FoldChange, equal_nan=True))

        res1 = dds.results(contrast=["condition", "C", "B"], addMLE=True)
        res2 = ddsNP.results(contrast=["condition", "C", "B"])
        self.assertTrue(np.allclose(res1.lfcMLE, res2.log2FoldChange, equal_nan=True))
