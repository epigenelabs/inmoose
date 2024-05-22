import unittest

import numpy as np
import patsy

from inmoose.deseq2 import (
    DESeq,
    estimateBetaPriorVar,
    estimateDispersionsGeneEst,
    estimateMLEForBetaPriorVar,
    makeExampleDESeqDataSet,
    nbinomLRT,
    nbinomWaldTest,
)
from inmoose.utils import Factor, pt


class Test(unittest.TestCase):
    def test_nbinomWald_errors(self):
        """test that nbinomWald throws various errors and works with edge cases"""
        dds = makeExampleDESeqDataSet(n=100, m=4)
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="testing requires dispersion estimates, first call estimateDispersions()",
        ):
            nbinomWaldTest(dds)
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="testing requires dispersion estimates, first call estimateDispersions()",
        ):
            nbinomLRT(dds)

        dds = dds.estimateSizeFactors()
        dds = dds.estimateDispersions()
        mm = patsy.dmatrix("~condition", dds.obs)
        # mm0 = patsy.dmatrix("~1", dds.obs)
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="user-supplied model matrix with betaPrior=True requires supplying betaPriorVar",
        ):
            nbinomWaldTest(dds, betaPrior=True, modelMatrix=mm)
        # TODO
        # with self.assertRaisesRegex(ValueError, expected_regex="unused argument (betaPrior = TRUE)"):
        #    nbinomLRT(dds, betaPrior=True, full=mm, reduced=mm0)
        with self.assertRaisesRegex(
            ValueError, expected_regex="expanded model matrices require a beta prior"
        ):
            nbinomWaldTest(dds, betaPrior=False, modelMatrixType="expanded")
        # with self.assertRaisesRegex(ValueError, expected_regex="unused arguments (betaPrior = FALSE, modelMatrixType = 'expanded')"):
        #    nbinomLRT(dds, betaPrior=False, modelMatrixType="expanded")

        dds2 = estimateMLEForBetaPriorVar(dds.copy())
        estimateBetaPriorVar(dds2, betaPriorMethod="quantile")
        dds = nbinomWaldTest(dds, modelMatrixType="standard")
        # TODO
        # covarianceMatrix(dds, 1)

        # changing 'df'
        dds = makeExampleDESeqDataSet(n=100, m=4)
        dds.X[:, :4] = 0
        dds = dds.estimateSizeFactors()
        dds = dds.estimateDispersions()
        dds = nbinomWaldTest(dds)
        dds.results().pvalue[:8]
        dds = nbinomWaldTest(dds, useT=True, df=np.ones(100))
        dds.results().pvalue[:8]

        # try nbinom after no fitted dispersions
        dds = makeExampleDESeqDataSet(n=100, m=4)
        dds = dds.estimateSizeFactors()
        dds = estimateDispersionsGeneEst(dds)
        dds.var["dispersion"] = dds.var["dispGeneEst"]
        dds = nbinomWaldTest(dds)

    def test_nbinomWald_useT(self):
        """test that useT uses proper degrees of freedom"""
        dds = makeExampleDESeqDataSet(n=200, m=15, seed=42)
        dds.X[:, 100:105] = 0
        dds.obs["condition"] = Factor(np.repeat(["A", "B", "C"], 5))
        dds.design = "~condition"
        dds = DESeq(dds, useT=True)
        dds = dds.removeResults()
        w = np.ones(dds.shape)
        w[0, :100] = 0
        w[:4, 0] = 0
        w[5:9, 0] = 0
        w[10:14, 0] = 0
        dds.layers["weights"] = w
        dds = DESeq(dds, useT=True)
        res = dds.results()
        self.assertTrue(np.isnan(res.pvalue.iloc[0]))
        self.assertEqual(dds.var["tDegreesFreedom"].iloc[1], 15 - 1 - 3)
        self.assertEqual(
            res.pvalue.iloc[1],
            2 * pt(np.abs(res.stat.iloc[1]), df=15 - 1 - 3, lower_tail=False),
        )

        # also lfcThreshold
        res = dds.results(lfcThreshold=1, altHypothesis="greaterAbs")
        idx = np.nonzero(((res.log2FoldChange > 1) & ~np.isnan(res.pvalue)).values)[0][
            0
        ]
        self.assertEqual(
            res.pvalue.iloc[idx],
            2 * pt(res.stat.iloc[idx], df=15 - 1 - 3, lower_tail=False),
        )
        res = dds.results(lfcThreshold=1, altHypothesis="greater")
        idx = np.nonzero(((res.log2FoldChange > 1) & ~np.isnan(res.pvalue)).values)[0][
            0
        ]
        self.assertEqual(
            res.pvalue.iloc[idx],
            pt(res.stat.iloc[idx], df=15 - 1 - 3, lower_tail=False),
        )

        res = dds.results(lfcThreshold=1, altHypothesis="less")
        idx = np.nonzero(((res.log2FoldChange < -1) & ~np.isnan(res.pvalue)).values)[0][
            0
        ]
        self.assertEqual(
            res.pvalue.iloc[idx],
            pt(-res.stat.iloc[idx], df=15 - 1 - 3, lower_tail=False),
        )

        res = dds.results(lfcThreshold=1, altHypothesis="lessAbs")
        idx = np.nonzero(
            ((np.abs(res.log2FoldChange) < 1) & ~np.isnan(res.pvalue)).values
        )[0][0]
        self.assertEqual(
            res.pvalue.iloc[idx],
            pt(res.stat.iloc[idx], df=15 - 1 - 3, lower_tail=False),
        )

        # also novel contrasts
        res = dds.results(contrast=["condition", "C", "B"])
        self.assertTrue(np.isnan(res.pvalue.iloc[0]))
        self.assertTrue(dds.var["tDegreesFreedom"].iloc[1] == 15 - 1 - 3)
        self.assertTrue(
            res.pvalue.iloc[1]
            == 2 * pt(abs(res.stat.iloc[1]), df=15 - 1 - 3, lower_tail=False)
        )
