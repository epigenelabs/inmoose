import unittest

import pandas as pd

from inmoose.diffexp import DEResults


class Test(unittest.TestCase):
    def test_DEResults(self):
        df = pd.DataFrame()
        with self.assertRaisesRegex(
            ValueError, "log2FoldChange missing from results table"
        ):
            res = DEResults(df)

        df["log2FoldChange"] = [1, 2, 3]
        with self.assertRaisesRegex(ValueError, "lfcSE missing from results table"):
            res = DEResults(df)

        df["lfcSE"] = [0.1, 0.2, 0.3]
        with self.assertRaisesRegex(ValueError, "pvalue missing from results table"):
            res = DEResults(df)

        df["pvalue"] = [0.01, 0.02, 0.03]
        res = DEResults(df)

        _subres = res.loc[:1, :]
