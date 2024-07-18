import unittest

import numpy as np
import pandas as pd

from inmoose.deseq2 import DESeqDataSet
from inmoose.utils import Factor


class Test(unittest.TestCase):
    def test_method_errors(self):
        """test that methods throw errors"""
        coldata = pd.DataFrame({"x": Factor(["A", "A", "B", "B"])})
        counts = np.arange(1, 17).reshape((4, 4))
        dds = DESeqDataSet(counts, coldata, "~x")
        with self.assertLogs("inmoose", level="WARNING") as logChecker:
            dds.counts(replaced=True)
        self.assertRegex(
            logChecker.output[0],
            "There is no layer named 'replacedCounts', using original.",
        )
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="first calculate size factors, add normalizationFactors, or set normalized=False",
        ):
            dds.counts(normalized=True)
        with self.assertRaisesRegex(
            ValueError, expected_regex="size factors should be positive"
        ):
            dds.sizeFactors = [-1, -1, -1, -1]
        with self.assertRaisesRegex(
            ValueError, expected_regex="normalization factors should be positive"
        ):
            dds.normalizationFactors = np.full((4, 4), -1)
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="first call estimateSizeFactors or provide a normalizationFactor matrix before calling estimateDispersions",
        ):
            dds.estimateDispersions()
