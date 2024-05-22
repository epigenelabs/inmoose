import unittest

import numpy as np
import patsy

from inmoose.deseq2 import DESeq, makeExampleDESeqDataSet
from inmoose.utils import Factor


@unittest.skip("lfcShrink is not implemented")
class Test(unittest.TestCase):
    def test_interation_errors(self):
        """test that interactions throw errors"""
        dds = makeExampleDESeqDataSet(n=100, m=8)
        dds.obs["group"] = Factor(np.repeat(["X", "Y"], dds.n_obs / 2))
        dds.design = "~ condition + group + condition:group"
        dds = DESeq(dds)
        self.assertEqual(dds.resultsNames()[3], "conditionB.groupY")

        # interactions error
        with self.assertRaisesRegex(
            ValueError, expected_regex="designs with interations"
        ):
            DESeq(dds, betaPrior=True)

        # also lfcShrink
        res = dds.results(name="conditionB.groupY")
        with self.assertRaises(NotImplementedError):
            res = lfcShrink(dds, coef=4, res=res, type="normal")  # noqa: F821

        res = dds.results(contrast=["condition", "B", "A"])
        with self.assertRaises(NotImplementedError):
            res = lfcShrink(  # noqa: F821
                dds, contrast=["condition", "B", "A"], res=res, type="normal"
            )

        # however, this is allowed
        dds2 = dds.copy()
        dds2.design = patsy.dmatrix("~ condition + group + condition:group", dds2.obs)
        dds2 = DESeq(dds2)
        dds2.results(name="conditionB.groupY")
        lfcShrink(dds2, coef=4, res=res, type="normal")  # noqa: F821
