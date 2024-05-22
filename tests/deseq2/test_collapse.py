import unittest

import numpy as np

from inmoose.deseq2 import collapseReplicates, makeExampleDESeqDataSet


class Test(unittest.TestCase):
    def test_collapse(self):
        """test that collapse replicates works"""
        dds = makeExampleDESeqDataSet(n=10, m=8)
        dds.obs["sample"] = np.repeat([1, 2, 3, 4], 2)
        dds.obs["run"] = np.arange(8)
        dds2 = collapseReplicates(dds, groupby=dds.obs["sample"], run=dds.obs["run"])
        self.assertTrue(np.all(dds2.counts()[0, :] == np.sum(dds.counts()[0:2, :], 0)))
        self.assertEqual(dds2.obs["runsCollapsed"].iloc[0], "0,1")
