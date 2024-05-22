import unittest

from inmoose.deseq2 import DESeq, makeExampleDESeqDataSet, nbinomLRT
from inmoose.utils import Factor


class Test(unittest.TestCase):
    def test_LRT(self):
        """test that test='LRT' gives correct errors"""
        dds = makeExampleDESeqDataSet(n=100, m=4)
        dds.obs["group"] = Factor([1, 2, 1, 2])
        dds.design = "~condition"
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="the following variables in the reduced formula not in the full formula: group",
        ):
            DESeq(dds, test="LRT", reduced="~group")
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="test='LRT' does not support use of expanded model matrix",
        ):
            DESeq(dds, test="LRT", reduced="~1", modelMatrixType="expanded")
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="test='LRT' does not support use of LFC shrinkage, use betaPrior=False",
        ):
            DESeq(dds, test="LRT", reduced="~group", betaPrior=True)
        dds = dds.estimateSizeFactors()
        dds = dds.estimateDispersions()
        with self.assertRaisesRegex(
            ValueError, expected_regex="provide a reduced formula for the LRT"
        ):
            nbinomLRT(dds)

    @unittest.skip("unimplemented")
    def test_glmGamPoi(self):
        raise NotImplementedError()

    def test_LRT2(self):
        """test that test='LRT' with quasi-likelihood estimates gives correct errors"""
        dds = makeExampleDESeqDataSet(n=100, m=4)
        dds.obs["group"] = Factor([1, 2, 1, 2])
        dds.design = "~condition + group"
        with self.assertRaises(NotImplementedError):
            DESeq(dds, test="Wald", fitType="glmGamPoi")
        dds = dds.estimateSizeFactors()
        dds_gp = dds.estimateDispersions()
        with self.assertRaises(NotImplementedError):
            nbinomLRT(dds_gp, reduced="~condition", type_="glmGamPoi")
