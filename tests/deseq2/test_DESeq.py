import unittest

import patsy

from inmoose.deseq2 import DESeq, makeExampleDESeqDataSet
from inmoose.utils import Factor


class Test(unittest.TestCase):
    def test_DESeq(self):
        """test that DESeq() gives correct errors"""
        dds = makeExampleDESeqDataSet(n=100, m=8)
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="likelihood ratio test requires a 'reduced' design",
        ):
            DESeq(dds, test="LRT")
        with self.assertRaisesRegex(
            ValueError, expected_regex="'reduced' ignored when test='Wald'"
        ):
            DESeq(dds, test="Wald", full="~condition", reduced="~1")
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="'full' specified as formula should match obj.design",
        ):
            DESeq(dds, full="~1")

        m = patsy.dmatrix("~condition", dds.obs)
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="if one of 'full' or 'reduced' is a matrix, the other must also be a matrix",
        ):
            DESeq(dds, test="LRT", full=m, reduced="~1")
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="the number of columns of 'full' should be larger than the number of columns of 'reduced'",
        ):
            DESeq(dds, test="LRT", full=m, reduced=m)
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="'betaPrior'=True is not supported for user-provided model matrices",
        ):
            DESeq(dds, full=m, betaPrior=True)

        dds.design = "~0 + condition"
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="betaPrior=True can only be used if the design has an intercept. If not, use betaPrior=False",
        ):
            DESeq(dds, betaPrior=True)

        dds = makeExampleDESeqDataSet(n=100)
        dds.obs["condition"] = Factor(dds.obs["condition"]).add_categories("C")
        dds.design = "~condition"
        with self.assertRaisesRegex(
            ValueError, expected_regex="full model matrix is not full rank"
        ):
            DESeq(dds)
