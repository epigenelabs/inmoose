import unittest

import numpy as np

from inmoose.deseq2 import DESeqDataSet


class Test(unittest.TestCase):
    @unittest.skip("fpkm not implemented")
    def test_fpkm(self):
        """test that fpkm works as expected"""
        df = {"x": [1, 2]}
        dds = DESeqDataSet(np.array([[1, 2, 3, 4], [2, 4, 6, 8]]), df, "~1")
        self.assertEqual(dds.fpkm()[0, 0], 1e7)
        self.assertEqual(dds.fpm()[0, 0], 1e5)
        self.assertEqual(dds.fpm(robuts=False)[0, 0], 1e5)
