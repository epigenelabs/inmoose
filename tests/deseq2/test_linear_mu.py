import unittest

import numpy as np

from inmoose.deseq2 import (
    DESeqDataSet,
    estimateDispersionsFit,
    estimateDispersionsGeneEst,
    estimateDispersionsMAP,
    makeExampleDESeqDataSet,
    nbinomWaldTest,
)


class Test(unittest.TestCase):
    def test_linear_mu(self):
        """test that the use of linear model for fitting mu works as expected"""

        dds = makeExampleDESeqDataSet(
            n=100,
            m=4,
            interceptMean=10,
            interceptSD=3,
            dispMeanRel=lambda x: 0.5,
            sizeFactors=[0.5, 1, 1, 2],
        )
        dds = dds.estimateSizeFactors()
        dds1 = DESeqDataSet(dds.copy())
        dds1 = estimateDispersionsGeneEst(dds1, linearMu=False)
        dds2 = estimateDispersionsGeneEst(dds.copy(), linearMu=True)
        mu1 = dds1.layers["mu"]
        mu2 = dds2.layers["mu"]
        cors = np.diag(np.corrcoef(mu1, mu2)[: mu1.shape[0], mu1.shape[1] :])
        self.assertTrue(np.all(cors > 1 - 1e-6))

        dds2 = estimateDispersionsFit(dds2, fitType="mean")
        dds2 = estimateDispersionsMAP(dds2)
        dds2 = nbinomWaldTest(dds2)
        dds2.results()
