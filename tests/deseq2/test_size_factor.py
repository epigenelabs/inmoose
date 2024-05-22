import unittest

import numpy as np
import pandas as pd
import patsy

from inmoose.deseq2 import (
    DESeqDataSet,
    estimateSizeFactorsForMatrix,
    makeExampleDESeqDataSet,
)


class Test(unittest.TestCase):
    def test_size_factor(self):
        """test that size factor works"""

        m = np.arange(1, 17).T.reshape(4, 4)
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="geoMeans should be as long as the number of columns of counts",
        ):
            estimateSizeFactorsForMatrix(m, geoMeans=np.arange(1, 6))

        with self.assertRaisesRegex(
            ValueError,
            expected_regex="every gene contains at least one zero, cannot compute log geometric means",
        ):
            estimateSizeFactorsForMatrix(m, geoMeans=np.zeros(4))

        with self.assertRaisesRegex(IndexError, expected_regex=""):
            estimateSizeFactorsForMatrix(m, controlGenes="foo")

        sf = estimateSizeFactorsForMatrix(m, geoMeans=np.arange(1, 5))
        ref = [0.3495015, 0.9246941, 1.4964761, 2.0676788]
        self.assertTrue(np.allclose(sf, ref))
        sf = estimateSizeFactorsForMatrix(m, controlGenes=np.array([0, 1]))
        ref = [0.2520327, 0.9761184, 1.6906866, 2.4042385]
        self.assertTrue(np.allclose(sf, ref))

        # norm matrix works
        nm = m / np.exp(np.mean(np.log(m), 0))
        true_sf = np.array([2, 1, 1, 0.5])
        counts = 2 * m
        counts[0, :] *= 2
        counts[3, :] = m[3, :]
        dds = DESeqDataSet(counts, pd.DataFrame({"x": np.arange(1, 5)}), "~1")
        dds = dds.estimateSizeFactors(normMatrix=nm)
        self.assertTrue(np.allclose((dds.normalizationFactors / nm)[:, 0], true_sf))

        # make some counts with zeros
        true_sf = 2.0 ** np.array([-2, -2, -1, -1, 0, 0, 0, 0, 1, 1, 2, 2])

        def dmr(x):
            return 0.01

        dds = makeExampleDESeqDataSet(sizeFactors=true_sf, n=100, dispMeanRel=dmr)
        cts = dds.X
        # set one random zero per row
        rng = np.random.default_rng(1)
        idx = rng.choice(np.arange(cts.shape[1]), size=cts.shape[0], replace=True)
        for i in range(cts.shape[0]):
            cts[i, idx[i]] = 0
        cts[0, 0] = 1000000  # an outlier
        dds.X = cts

        # positive counts method
        dds = dds.estimateSizeFactors(type_="poscounts")
        sf = dds.sizeFactors
        coefs = np.linalg.lstsq(patsy.dmatrix("~true_sf"), sf, rcond=None)[0]
        self.assertLess(np.abs(coefs[0]), 0.1)
        self.assertLess(np.abs(coefs[1] - 1), 0.1)

        # iterate method
        # TODO not implemented yet
        # dds = dds.estimateSizeFactors(type_="iterate")
        # sf = dds.sizeFactors
        # coefs = np.linalg.lstsq(patsy.dmatrix("~true_sf"), sf, rcond=None)[0]
        # self.assertLess(np.abs(coefs[0]), .1)
        # self.assertLess(np.abs(coefs[1]-1), .1)
