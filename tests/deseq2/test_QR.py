import unittest

import numpy as np

from inmoose.deseq2 import DESeq, makeExampleDESeqDataSet, nbinomWaldTest


class Test(unittest.TestCase):
    def test_QR(self):
        """test that not using QR works as expected"""

        dds = makeExampleDESeqDataSet(n=100, betaSD=1, seed=42)
        dds = DESeq(dds, quiet=True)
        ddsNoQR = nbinomWaldTest(dds.copy(), useQR=False)
        res = dds.results()
        resNoQR = ddsNoQR.results()
        self.assertTrue(np.allclose(res.log2FoldChange, resNoQR.log2FoldChange))
