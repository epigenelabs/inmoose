import unittest

import numpy as np
import pandas as pd
from scipy.stats import f

from inmoose.deseq2 import DESeq, DESeqDataSet, makeExampleDESeqDataSet, replaceOutliers


class Test(unittest.TestCase):
    def test_outlier_filtering_replacement(self):
        """test that outlier filtering and replacement works as expected"""
        dds = makeExampleDESeqDataSet(
            n=100, m=12, dispMeanRel=lambda x: 4 / x + 0.5, seed=42
        )
        dds.X[:, 0] = 0
        dds.X[:, 1] = 10
        dds.X[0, 1] = 100000
        dds.X[:, 2] = 0
        dds.X[0, 2] = 100000

        dds0 = DESeq(dds.copy(), minReplicatesForReplace=np.inf)
        dds1 = DESeq(dds.copy(), minReplicatesForReplace=6)
        pval0 = dds0.results().pvalue[0:3]
        pval1 = dds1.results().pvalue[0:3]
        LFC0 = dds0.results().log2FoldChange[0:3]
        LFC1 = dds1.results().log2FoldChange[0:3]

        padj_ref = [
            np.nan,
            np.nan,
            np.nan,
            0.8580403,
            0.8685670,
            0.8685670,
            0.8580403,
            0.8548205,
            0.9564876,
            0.8548205,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8548205,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.9607312,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8937823,
            0.9837392,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8685670,
            0.8580403,
            0.8580403,
            0.9377901,
            0.8580403,
            0.8580403,
            0.8580403,
            0.9118744,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.9068915,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.9068915,
            0.8580403,
            0.8922563,
            0.8580403,
            0.8580403,
            np.nan,
            0.9068915,
            0.8580403,
            0.8580403,
            0.8937823,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8548205,
            0.9288343,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8548205,
            0.8937823,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8580403,
            0.8937823,
            0.8580403,
        ]
        self.assertTrue(
            np.allclose(dds0.results().adj_pvalue, padj_ref, equal_nan=True)
        )

        # filtered
        self.assertTrue(np.all(np.isnan(pval0)))
        # not filtered
        self.assertTrue(np.all(~np.isnan(pval1[1:3])))
        # counts still the same
        self.assertTrue(np.all(dds1.counts() == dds.counts()))
        # first is NA
        self.assertTrue(np.isnan(LFC1.iloc[0]))
        # replaced, reduced LFC
        self.assertTrue(np.abs(LFC1.iloc[1]) < np.abs(LFC0.iloc[1]))
        # replaced, LFC now zero
        self.assertTrue(LFC1.iloc[2] == 0)
        idx = ~dds1.var["replace"]
        # the pvalue for those not replaced is equal
        self.assertTrue(
            np.array_equal(
                dds1.results().pvalue[idx], dds0.results().pvalue[idx], equal_nan=True
            )
        )

        # check that outlier filtering catched throughout range of mu
        beta0 = np.linspace(1, 16, 100)
        idx = np.repeat(np.repeat([True, False], [1, 9]), 10)
        for disp0 in [0.01, 0.1]:
            for m in [10, 20, 80]:
                dds = makeExampleDESeqDataSet(
                    n=100,
                    m=m,
                    interceptMean=beta0,
                    interceptSD=0,
                    dispMeanRel=lambda x: disp0,
                    seed=42,
                )
                dds.counts()[0, idx] = 1000 * 2 ** beta0[idx]
                dds = DESeq(
                    dds, minReplicatesForReplace=np.inf, quiet=True, fitType="mean"
                )
                res = dds.results()
                cutoff = f.ppf(0.99, 2, m - 2)
                outlierCooks = dds.layers["cooks"][0, idx] > cutoff
                nonoutlierCooks = dds.var["maxCooks"][~idx] < cutoff
                self.assertTrue(np.all(np.isnan(res.pvalue[idx])))
                self.assertTrue(np.all(outlierCooks))
                self.assertTrue(np.all(nonoutlierCooks))

        # TODO LRT not implemented so far
        # dds = makeExampleDESeqDataSet(n=100)
        # dds.counts()[0,0] = 1000000
        # dds = DESeq(dds, test="LRT", reduced="~1", minReplicatesForReplace=6)

        # test replace function
        dds = makeExampleDESeqDataSet(n=100, m=4)
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="first run DESeq, nbinomWaldTest or nbinomLRT to identify outliers",
        ):
            replaceOutliers(dds)
        dds = DESeq(dds)
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="at least 3 replicates are necessary in order to identify a sample as a count outlier",
        ):
            replaceOutliers(dds, minReplicates=2)

        # check model matrix standard bug
        dds = makeExampleDESeqDataSet(n=100, m=20)
        dds.counts()[:, 0] = 0
        dds.counts()[0, 0] = 100000
        dds = DESeq(dds, modelMatrixType="standard")

    def test_outlier_filtering_small_counts(self):
        """test that outlier filtering does not flag small counts"""
        dds = makeExampleDESeqDataSet(n=100, m=8, dispMeanRel=lambda x: 0.01, seed=42)
        dds.counts()[:, 0] = [0, 0, 0, 100, 2100, 2200, 2300, 2400]
        dds.counts()[0, 1:2] = 100000
        dds.counts()[0, 3] = 0
        dds = DESeq(dds, fitType="mean")
        res = dds.results()
        self.assertFalse(np.isnan(res["pvalue"].iloc[0]))
        self.assertTrue(np.all(np.isnan(res.pvalue[1:2])))

    def test_CR391(self):
        """
        test that the array indexing error reported in CR-391 is fixed
        NB: this test only checks that no exception is raised during computation
        """
        dds = DESeqDataSet(
            np.array(
                [
                    [527, 524, 10, 1541, 341, 396, 73, 31, 382, 122],
                    [585, 498, 7, 1329, 285, 352, 58, 24, 332, 154],
                    [33, 439, 9, 2114, 199, 115, 47, 0, 425, 19],
                    [34, 652, 20, 3040, 308, 153, 62, 2, 543, 34],
                    [42, 752, 14, 3796, 349, 215, 67, 1, 658, 30],
                    [38, 565, 15, 2743, 266, 118, 58, 5, 512, 28],
                ]
            ),
            clinicalData=pd.DataFrame({"group": [0, 0, 1, 1, 1, 1]}),
            design="~ group",
        )
        DESeq(dds)
