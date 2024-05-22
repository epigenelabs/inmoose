import unittest

import numpy as np
import pandas as pd
import patsy
import scipy.stats

from inmoose.deseq2 import DESeq, DESeqDataSet, makeExampleDESeqDataSet
from inmoose.utils import Factor


class Test(unittest.TestCase):
    def test_model_matrix(self):
        """test that supplying custom model matrix works"""
        dds = makeExampleDESeqDataSet(n=100, m=18)
        dds.obs["group"] = Factor(np.repeat([1, 2, 3], [6, 6, 6]))
        dds.obs["condition"] = Factor(
            np.repeat([np.repeat(["A", "B", "C"], [2, 2, 2])], 3, axis=0).flatten()
        )
        # note: design is not used
        dds.design = "~1"
        dds = dds[:16]

        m1 = patsy.dmatrix("~ group*condition", dds.obs)
        m2 = m1[:, :8]
        m2.design_info = m1.design_info
        m1 = m2
        m1.design_info.column_name_indexes.popitem()
        # m0 = patsy.dmatrix("~ group + condition", dds.obs)

        # TODO
        # dds = DESeq(dds, full=m1, reduced=m0, test="LRT")
        # dds.result()
        # dds.results(name="group2.conditionC", test="Wald")
        dds = dds.removeResults()
        dds = DESeq(dds, full=m1, test="Wald", betaPrior=False)
        dds.results()

        # test better error than error during matrix inversion
        clindata = pd.DataFrame(
            {
                "group": Factor(np.repeat([1, 2, 3], 6)),
                "group2": Factor(np.repeat([1, 2, 3], 6)),
                "condition": Factor(np.repeat([1, 2, 3, 4, 5, 6], 3)),
            }
        )
        counts = scipy.stats.poisson.rvs(100, size=(18, 10))
        m1 = patsy.dmatrix("~ group + group2", clindata)
        m2 = patsy.dmatrix("~ condition + group", clindata)
        dds = DESeqDataSet(counts, clindata, "~ group")
        with self.assertRaisesRegex(
            ValueError, expected_regex="the model matrix is not full rank"
        ):
            dds = DESeq(dds, full=m1, fitType="mean")
        with self.assertRaisesRegex(
            ValueError, expected_regex="the model matrix is not full rank"
        ):
            dds = DESeq(dds, full=m2, reduced=m1, test="LRT", fitType="mean")
