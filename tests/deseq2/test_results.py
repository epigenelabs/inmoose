import unittest

import numpy as np
import pandas as pd
from scipy.stats import norm

from inmoose.deseq2 import DESeq, makeExampleDESeqDataSet
from inmoose.utils import Factor


class Test(unittest.TestCase):
    def test_results(self):
        """test that results work as expected and throw errors"""
        # test contrasts
        dds = makeExampleDESeqDataSet(n=200, m=12, seed=42)
        dds.obs["condition"] = Factor(np.repeat([1, 2, 3], 4))
        dds.obs["group"] = Factor(np.repeat([[1, 2]], 6, axis=0).flatten())
        dds.obs["foo"] = np.repeat(["lo", "hi"], 6)
        dds.counts()[:, 0] = np.repeat([100, 200, 800], 4)

        dds.design = "~ group + condition"

        # calling results too early
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="could not find results in obj. first run DESeq()",
        ):
            dds.results()

        dds.sizeFactors = np.ones(dds.n_obs)
        dds = DESeq(dds)
        res = dds.results()
        # TODO
        # show_res = res.show()
        summary = res.summary()
        print(summary)
        summary_ref = """
out of 200 with nonzero total read count
adjusted p-value < 0.1
LFC > 0 (up)       : 1, 0.50%
LFC < 0 (down)     : 0, 0.00%
outliers [1]       : 0, 0.00%
low counts [2]     : 0, 0.00%
(mean count < 0)
[1] see 'cooksCutoff' argument of results()
[2] see 'independentFiltering' argument of results()
"""
        self.assertEqual(summary, summary_ref)

        # various results error checking
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="the LRT requires the user to run nbinomLRT or DESeq",
        ):
            dds.results(test="LRT")
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="when testing altHypothesis='lessAbs', set the argument lfcThreshold to a positive value",
        ):
            dds.results(altHypothesis="lessAbs")
        with self.assertRaisesRegex(
            ValueError, expected_regex="'name' should be a string"
        ):
            dds.results(name=["Intercept", "group1"])
        with self.assertRaisesRegex(ValueError, expected_regex="foo is not a factor"):
            dds.results(contrast=["foo", "B", "A"])
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="as 1 is the reference level, was expecting condition_4_vs_1 to be present in",
        ):
            dds.results(contrast=["condition", "4", "1"])
        with self.assertRaisesRegex(
            ValueError, expected_regex="invalid value for test: foo"
        ):
            dds.results(test="foo")
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="numeric contrast vector should have one element for every element of",
        ):
            dds.results(contrast=False)
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="'contrast', as a pair of lists, should have length 2",
        ):
            dds.results(contrast=["a", "b", "c", "d"])
        with self.assertRaisesRegex(
            ValueError, expected_regex="1 and 1 should be different level names"
        ):
            dds.results(contrast=["condition", "1", "1"])

        dds.results(independentFiltering=False)
        dds.results(contrast=["condition_2_vs_1"])

        with self.assertRaisesRegex(
            ValueError,
            expected_regex="condition_3_vs_1 and condition_3_vs_1 should be different level names",
        ):
            dds.results(
                contrast=["condition_2_vs_1", "condition_3_vs_1", "condition_3_vs_1"]
            )
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="'contrast', as a pair of lists, should have lists of strings as elements",
        ):
            dds.results(contrast=["condition_2_vs_1", 1])
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="all elements of the 2-element contrast should be elements of",
        ):
            dds.results(contrast=["condition_2_vs_1", "foo"])
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="elements in the 2-element contrast should only appear in the numerator",
        ):
            dds.results(contrast=["condition_2_vs_1", "condition_2_vs_1"])
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="all elements of the 2-element contrast should be elements of",
        ):
            dds.results(contrast=["", ""])
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="numeric contrast vector should have one element for every element of",
        ):
            dds.results(contrast=np.repeat(0, 6))
        with self.assertRaisesRegex(ValueError, expected_regex="foo is not a factor"):
            dds.results(contrast=["foo", "lo", "hi"])

        self.assertAlmostEqual(
            dds.results(contrast=["condition", "1", "3"]).log2FoldChange.iloc[0],
            -3,
            delta=1e-6,
        )
        self.assertAlmostEqual(
            dds.results(contrast=["condition", "1", "2"]).log2FoldChange.iloc[0],
            -1,
            delta=1e-6,
        )
        self.assertAlmostEqual(
            dds.results(contrast=["condition", "2", "3"]).log2FoldChange.iloc[0],
            -2,
            delta=1e-6,
        )

        # test a number of contrast as list options
        self.assertAlmostEqual(
            dds.results(
                contrast=["condition_3_vs_1", "condition_2_vs_1"]
            ).log2FoldChange.iloc[0],
            2,
            delta=1e-6,
        )
        dds.results(
            contrast=["condition_3_vs_1", "condition_2_vs_1"], listValues=[0.5, -0.5]
        )
        dds.results(contrast=["condition_3_vs_1", []])
        dds.results(contrast=["condition_3_vs_1", []], listValues=[0.5, -0.5])
        dds.results(contrast=[[], "condition_2_vs_1"])
        dds.results(contrast=[[], "condition_2_vs_1"], listValues=[0.5, -0.5])

        # test no prior on intercept
        self.assertTrue(np.array_equal(dds.betaPriorVar, np.repeat(1e6, 4)))

        # test thresholding
        dds.results(lfcThreshold=np.log2(1.5))
        dds.results(lfcThreshold=1, altHypothesis="lessAbs")
        dds.results(lfcThreshold=1, altHypothesis="greater")
        dds.results(lfcThreshold=1, altHypothesis="less")

        dds3 = DESeq(dds, betaPrior=True)
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="testing altHypothesis='lessAbs' requires setting the DESeq\(\) argument betaPrior=False",
        ):
            dds3.results(lfcThreshold=1, altHypothesis="lessAbs")

    def test_results_zero_intercept(self):
        """test results on designs with zero intercept"""
        dds = makeExampleDESeqDataSet(n=100, m=12, seed=42)
        dds.obs["condition"] = Factor(np.repeat([1, 2, 3], 4))
        dds.obs["group"] = Factor(np.repeat([[1, 2]], 6, axis=0).flatten())

        dds.X[:, 0] = np.repeat([100, 200, 400], 4)

        dds.design = "~ 0 + condition"
        dds = DESeq(dds, betaPrior=False)

        self.assertAlmostEqual(dds.results().log2FoldChange.iloc[0], 2, delta=0.1)
        self.assertAlmostEqual(
            dds.results(contrast=["condition", "2", "1"]).log2FoldChange.iloc[0],
            1.25,
            delta=0.1,
        )
        self.assertAlmostEqual(
            dds.results(contrast=["condition", "3", "2"]).log2FoldChange.iloc[0],
            0.68,
            delta=0.1,
        )
        self.assertAlmostEqual(
            dds.results(contrast=["condition", "1", "3"]).log2FoldChange.iloc[0],
            -2,
            delta=0.1,
        )
        self.assertAlmostEqual(
            dds.results(contrast=["condition", "1", "2"]).log2FoldChange.iloc[0],
            -1.25,
            delta=0.1,
        )
        self.assertAlmostEqual(
            dds.results(contrast=["condition", "2", "3"]).log2FoldChange.iloc[0],
            -0.68,
            delta=0.1,
        )
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="condition\[4\] and condition\[1\] are expected to be in",
        ):
            dds.results(contrast=["condition", "4", "1"])

        dds.design = "~ 0 + group + condition"
        dds = DESeq(dds, betaPrior=False)

        self.assertAlmostEqual(dds.results().log2FoldChange.iloc[0], 2, delta=0.1)
        self.assertAlmostEqual(
            dds.results(contrast=["condition", "3", "1"]).log2FoldChange.iloc[0],
            2,
            delta=0.1,
        )
        self.assertAlmostEqual(
            dds.results(contrast=["condition", "2", "1"]).log2FoldChange.iloc[0],
            1.25,
            delta=0.1,
        )
        self.assertAlmostEqual(
            dds.results(contrast=["condition", "3", "2"]).log2FoldChange.iloc[0],
            0.68,
            delta=0.1,
        )
        self.assertAlmostEqual(
            dds.results(contrast=["condition", "1", "3"]).log2FoldChange.iloc[0],
            -2,
            delta=0.1,
        )
        self.assertAlmostEqual(
            dds.results(contrast=["condition", "1", "2"]).log2FoldChange.iloc[0],
            -1.25,
            delta=0.1,
        )
        self.assertAlmostEqual(
            dds.results(contrast=["condition", "2", "3"]).log2FoldChange.iloc[0],
            -0.68,
            delta=0.1,
        )

    @unittest.skip("LRT is not implemented yet")
    def test_results_likelihood_ratio_test(self):
        """test results with likelihood ratio test"""
        dds = makeExampleDESeqDataSet(n=100)
        dds.obs["group"] = Factor([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
        dds.design = "~ group + condition"
        dds = DESeq(dds, test="LRT", reduced="~group")

        self.assertFalse(
            np.all(
                dds.results(name="condition_B_vs_A").stat
                == dds.results(name="condition_B_vs_A", test="Wald").stat
            )
        )

        # LFC are already MLE
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="addMLE=TRUE is only for when a beta prior was used",
        ):
            dds.results(addMLE=True)
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="tests of log fold change above or below a theshold must be Wald tests",
        ):
            dds.results(lfcThreshold=1, test="LRT")

        self.assertTrue(
            np.all(
                dds.results(test="LRT", contrast=["group", "1", "2"]).log2FoldChange
                == -dds.results(test="LRT", contrast=["group", "2", "1"]).log2FoldChange
            )
        )

    def test_results_basics(self):
        """test that results basics regarding format, saveCols, tidy, MLE, remove are working"""
        dds = makeExampleDESeqDataSet(n=100)
        dds.var["score"] = np.arange(1, 101)
        dds = DESeq(dds)

        # try saving metadata columns
        res = dds.results(saveCols="score")  # string

        # check tidy-ness (unimplemented)
        with self.assertRaises(NotImplementedError):
            res = dds.results(tidy=True)
            self.assertTrue(res.columns[0] == "rows")

        # test MLE and 'name'
        dds2 = DESeq(dds, betaPrior=True)
        dds2.results(addMLE=True)
        with self.assertRaises(ValueError):
            dds2.results(name="condition_B_vs_A", addMLE=True)

        # test remove results
        dds = dds.removeResults()
        self.assertTrue(dds.var.description.filter("results").empty)

    def test_confidence_intervals(self):
        """test that confidence intervals are properly computed when required"""
        dds = makeExampleDESeqDataSet(n=100)
        dds = DESeq(dds)

        res = dds.results()
        self.assertFalse("CI_L" in res.columns)
        self.assertFalse("CI_R" in res.columns)

        res = dds.results(confint=True)
        pd.testing.assert_series_equal(
            res["CI_L"],
            res["log2FoldChange"] + norm.isf(0.975) * res["lfcSE"],
            rtol=1e-6,
            check_names=False,
        )
        pd.testing.assert_series_equal(
            res["CI_R"],
            res["log2FoldChange"] + norm.ppf(0.975) * res["lfcSE"],
            rtol=1e-6,
            check_names=False,
        )

        res = dds.results(confint=0.75)
        pd.testing.assert_series_equal(
            res["CI_L"],
            res["log2FoldChange"] + norm.isf(0.875) * res["lfcSE"],
            rtol=1e-6,
            check_names=False,
        )
        pd.testing.assert_series_equal(
            res["CI_R"],
            res["log2FoldChange"] + norm.ppf(0.875) * res["lfcSE"],
            rtol=1e-6,
            check_names=False,
        )

    @unittest.skip("not sure what to test")
    def test_results_custom_filters(self):
        """test that custom filters can be provided to results()"""
        dds = makeExampleDESeqDataSet(n=200, m=4, betaSD=np.repeat([0, 2], [150, 50]))
        dds = DESeq(dds)
        _res = dds.results()
        _method = "BH"
        _alpha = 0.1

        raise NotImplementedError()
