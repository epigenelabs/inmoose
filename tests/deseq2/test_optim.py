import unittest

import numpy as np
import patsy
import scipy.stats

from inmoose.deseq2 import DESeq, makeExampleDESeqDataSet
from inmoose.deseq2.fitNbinomGLMs import fitNbinomGLMs


class Test(unittest.TestCase):
    def test_optim(self):
        """test that optim gives same results"""
        dds = makeExampleDESeqDataSet(n=100, interceptMean=10, interceptSD=3, seed=51)
        dds = dds.estimateSizeFactors()
        dds = dds.estimateDispersions()
        # make a larger predictor to test scaling
        dds.obs["condition"] = scipy.stats.norm.rvs(
            loc=0, scale=1000, size=dds.n_obs, random_state=51
        )
        modelMatrix = patsy.dmatrix("~condition", dds.obs)
        fit = fitNbinomGLMs(
            dds,
            modelMatrix=modelMatrix,
            modelFormula="~condition",
            alpha_hat=dds.var["dispersion"],
            lambda_=(2, 2),
            renameCols=True,
            betaTol=1e-8,
            maxit=100,
            useOptim=True,
            useQR=True,
            forceOptim=False,
        )
        fitOptim = fitNbinomGLMs(
            dds,
            modelMatrix=modelMatrix,
            modelFormula="~condition",
            alpha_hat=dds.var["dispersion"],
            lambda_=(2, 2),
            renameCols=True,
            betaTol=1e-8,
            maxit=100,
            useOptim=True,
            useQR=True,
            forceOptim=True,
        )

        self.assertTrue(np.allclose(fit["betaMatrix"], fitOptim["betaMatrix"]))
        self.assertTrue(np.allclose(fit["betaSE"], fitOptim["betaSE"]))

        # test optim gives same lfcSE
        dds = makeExampleDESeqDataSet(n=100, m=10, seed=42)
        dds.X[:, 0] = [0, 0, 0, 0, 0, 1000, 1000, 0, 0, 0]
        dds = DESeq(dds, betaPrior=False)
        # beta iter = 100 implies optim used for fitting
        self.assertEqual(dds.var["betaIter"].iloc[0], 100)
        res1 = dds.results(contrast=["condition", "B", "A"])
        res2 = dds.results(contrast=[0, 1])
        self.assertTrue(np.allclose(res1.lfcSE, res2.lfcSE))
        self.assertTrue(np.allclose(res1.pvalue, res2.pvalue, equal_nan=True))
