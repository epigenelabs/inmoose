import sys
import unittest

import numpy as np
import pandas as pd

from inmoose.deseq2 import DESeq, makeExampleDESeqDataSet, nbinomWaldTest
from inmoose.deseq2.fitNbinomGLMs import fitNbinomGLMsOptim
from inmoose.utils import Factor


class Test(unittest.TestCase):
    def test_weights(self):
        """test that weights work"""
        dds = makeExampleDESeqDataSet(n=10, seed=42)
        w = np.ones(dds.shape)
        w[0, 0] = 0
        # check that weight to 0 is like a remove sample
        dds = DESeq(dds, quiet=True)
        dds2 = dds.copy()
        dds2.layers["weights"] = w
        dds2 = nbinomWaldTest(dds2)
        dds3 = dds.copy()[1:, :]
        dds3 = nbinomWaldTest(dds3)

        # in terms of LFC, SE and deviance
        self.assertAlmostEqual(
            dds2.results().log2FoldChange.iloc[0],
            dds3.results().log2FoldChange.iloc[0],
            delta=1e-5,
        )
        self.assertAlmostEqual(
            dds2.results().lfcSE.iloc[0], dds3.results().lfcSE.iloc[0], delta=1e-6
        )
        self.assertAlmostEqual(
            dds2.var["deviance"].iloc[0], dds3.var["deviance"].iloc[0], delta=1e-7
        )

        # check weights working in the optim code
        nf = np.repeat(dds.sizeFactors.values[:, None], dds.n_vars, axis=1)
        o = fitNbinomGLMsOptim(
            obj=dds,
            modelMatrix=dds.design,
            lambda_=np.repeat(1e-6, 2),
            colsForOptim=[0],
            colStable=np.repeat(True, dds.n_vars),
            normalizationFactors=nf,
            alpha_hat=dds.var["dispersion"],
            weights=w,
            useWeights=True,
            betaMatrix=pd.DataFrame(np.zeros((dds.n_vars, 2))),
            betaSE=pd.DataFrame(np.zeros((dds.n_vars, 2))),
            betaConv=np.repeat(False, dds.n_vars),
            beta_mat=np.zeros((dds.n_vars, 2)),
            mu=np.zeros(dds.shape),
            logLike=np.zeros(dds.n_vars),
        )
        self.assertAlmostEqual(
            dds3.results().log2FoldChange.iloc[0],
            o["betaMatrix"].iloc[0, 1],
            delta=1e-4,
        )

    def test_weights_with_beta_prior(self):
        """test that weights can be used with betaPrior=True"""
        dds = makeExampleDESeqDataSet(n=10, seed=42)
        w = np.ones(dds.shape)
        w[0, 0] = 0
        dds.layers["weights"] = w
        dds = DESeq(dds, betaPrior=True, quiet=True)

        # check weights working for intercept only
        dds.design = "~1"
        dds = DESeq(dds, quiet=True)
        dds2 = dds
        dds2.layers["weights"] = w
        dds2 = nbinomWaldTest(dds2)
        dds3 = dds[1:, :]
        dds3 = nbinomWaldTest(dds3)

        self.assertEqual(
            dds2.results().log2FoldChange.iloc[0], dds3.results().log2FoldChange.iloc[0]
        )
        self.assertEqual(dds2.results().lfcSE.iloc[0], dds3.results().lfcSE.iloc[0])
        self.assertEqual(dds2.var["deviance"].iloc[0], dds3.var["deviance"].iloc[0])

    def test_weights_downweight_outliers(self):
        """test that weights downweight outlier in dispersion estimation"""
        dds = makeExampleDESeqDataSet(n=10, seed=42)
        dds.counts()[0, 0] = 100
        dds.sizeFactors = np.ones(dds.n_obs)
        dds = dds.estimateDispersions()
        dds2 = dds.copy()
        w = np.ones(dds.shape)
        w[0, 0] = 0
        dds2.layers["weights"] = w
        dds2 = dds2.estimateDispersions()
        dds3 = dds.copy()[1:, :]
        dds3 = dds3.estimateDispersions()

        self.assertAlmostEqual(
            dds2.var["dispGeneEst"].iloc[0], dds3.var["dispGeneEst"].iloc[0], delta=1e-1
        )
        # MAP estimates won't be equal because of different dispersion prior widths
        self.assertGreater(dds.var["dispMAP"].iloc[0], dds2.var["dispMAP"].iloc[0])

    def test_weights_warning(self):
        """test that weights failing check gives warning, passes them through"""
        dds = makeExampleDESeqDataSet(n=10, seed=42)
        w = np.ones(dds.shape)
        w[0:6, 0] = 0
        dds.layers["weights"] = w
        with self.assertLogs("inmoose", level="WARNING") as logChecker:
            dds = DESeq(dds)
        self.assertRegex(
            # account for https://github.com/python/cpython/issues/86109
            logChecker.output[0]
            if sys.version_info >= (3, 10)
            else logChecker.output[3],
            "for 1 genes, the weights as supplied won't allow parameter estimation",
        )
        self.assertTrue(dds.var["allZero"].iloc[0])
        self.assertTrue(dds.var["weightsFail"].iloc[0])
        dds.results()

    @unittest.skip("not sure what is tested here")
    def test_weights_CR(self):
        """test that weights with and without CR term included"""
        alpha = 0.25

        def dmr(x):
            return alpha

        dds = makeExampleDESeqDataSet(
            n=50, m=100, betaSD=1, interceptMean=10, interceptSD=0.5, dispMeanRel=dmr
        )
        dds.obs["group"] = Factor(np.repeat(np.arange(50), 2))
        dds.design = "~0 + group + condition"
        w = np.ones(dds.shape)
        o = 35
        w[0:o, :] = 1e-6
        w[50 : (50 + o), :] = 1e-6
        dds.layers["weights"] = w
        dds.counts()[1:o, :] = 1
        dds.counts()[50 : (50 + o), :] = 1
        dds.sizeFactors = 1
        dds = dds.estimateDispersions(fitType="mean")
        dds.estimateDispersions(fitType="mean", useCR=False)
