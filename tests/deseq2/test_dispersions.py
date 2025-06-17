import unittest

import numpy as np
import pandas as pd
import patsy
from scipy.optimize import minimize

from inmoose.deseq2 import (
    DESeq,
    DESeqDataSet,
    estimateDispersionsFit,
    estimateDispersionsGeneEst,
    estimateDispersionsMAP,
    makeExampleDESeqDataSet,
)
from inmoose.deseq2.deseq2_cpp import fitDisp, fitDispGridWrapper
from inmoose.deseq2.fitNbinomGLMs import fitNbinomGLMs
from inmoose.utils import Factor, dnbinom_mu, dnorm


class Test(unittest.TestCase):
    def test_dispersion_errors(self):
        """test that expected errors are thrown during dispersion estimation"""
        dds = makeExampleDESeqDataSet(n=100, m=2)
        dds = dds.estimateSizeFactors()
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="the number of samples and the number of model coefficients are equal",
        ):
            estimateDispersionsGeneEst(dds)

        dds = makeExampleDESeqDataSet(
            n=100,
            m=4,
            dispMeanRel=lambda x: 0.001 + x / 1e3,
            interceptMean=8,
            interceptSD=2,
        )
        dds = dds.estimateSizeFactors()
        dds.var["dispGeneEst"] = np.repeat(1e-7, 100)
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="all gene-wise dispersion estimates are within 2 orders of magnitude",
        ):
            estimateDispersionsFit(dds)
        dds = estimateDispersionsGeneEst(dds)
        # TODO expect message "note: fitType='parametric', but the dispersion trend was not well captured"
        # estimateDispersionsFit(dds)

        dds = makeExampleDESeqDataSet(n=100, m=4)
        dds = dds.estimateSizeFactors()
        dds.var["dispGeneEst"] = np.repeat(1e-7, 100)
        dds.setDispFunction(lambda x: 1e-6)
        with self.assertLogs("inmoose", level="WARNING") as logChecker:
            estimateDispersionsMAP(dds)
        self.assertRegex(
            logChecker.output[0],
            "all genes have dispersion estimates < 1e-06, returning disp = 1e-07",
        )

        dds = makeExampleDESeqDataSet(n=100, m=4)
        dds = dds.estimateSizeFactors()
        dds.obs["condition"] = Factor(dds.obs["condition"]).add_categories("C")
        dds.design = "~condition"
        with self.assertRaisesRegex(
            ValueError, expected_regex="the model matrix is not full rank"
        ):
            dds.estimateDispersions()
        dds.obs["condition"] = Factor(dds.obs["condition"]).droplevels()
        dds.obs["group"] = dds.obs["condition"]
        dds.design = "~ group + condition"
        with self.assertRaisesRegex(
            ValueError, expected_regex="the model matrix is not full rank"
        ):
            dds.estimateDispersions()

        dds = makeExampleDESeqDataSet(n=100, m=2)
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="The design matrix has the same number of samples and coefficients to fit",
        ):
            DESeq(dds)

    def test_dispersion_fitting(self):
        """test that the fitting of dispersion gives the expected values using various methods"""
        # test the optimization of the logarithm of dispersion (alpha)
        # parameter with Cox-Reid adjustment and prior distribution.
        # also test the derivatives of the log posterior w.r.t. log alpha
        m = 10
        # y = scipy.stats.poisson.rvs(20, size=m, random_state=42)
        y = np.array([17, 25, 25, 21, 13, 22, 23, 22, 18, 26])
        sf = np.ones(m)
        condition = Factor(np.repeat([0, 1], [m / 2, m / 2]))
        x = patsy.dmatrix("1+condition", data={"condition": condition})

        lambda_ = 2
        alpha = 0.5

        # make a DESeqDataSet but don't use the design formula
        # instead we supply directly a model matrix
        dds = DESeqDataSet(
            y[:, None],
            clinicalData=pd.DataFrame({"condition": condition}),
            design="~condition",
        )
        dds.sizeFactors = sf
        dds.var["dispersion"] = alpha
        dds.var["baseMean"] = np.mean(y)

        # for testing we convert beta to the natural log scale:
        # convert lambda from log to log2 scale by multiplying by log(2)**2
        # then convert beta back from log2 to log scale by multiplying by log(2)
        betaDESeq = (
            np.log(2)
            * fitNbinomGLMs(dds, lambda_=[0, lambda_ * np.log(2) ** 2], modelMatrix=x)[
                "betaMatrix"
            ]
        )
        log_alpha_prior_mean = 0.5
        log_alpha_prior_sigmasq = 1
        mu_hat = np.exp(x @ betaDESeq.T).values.squeeze()

        dispRes = fitDisp(
            y[:, None],
            x=x,
            mu_hat=mu_hat[:, None],
            log_alpha=0,
            log_alpha_prior_mean=log_alpha_prior_mean,
            log_alpha_prior_sigmasq=log_alpha_prior_sigmasq,
            min_log_alpha=np.log(1e-8),
            kappa_0=1,
            tol=1e-16,
            maxit=100,
            usePrior=True,
            weights=np.ones((len(y), 1)),
            useWeights=False,
            weightThreshold=1e-2,
            useCR=True,
        )

        # maximum a posteriori (MAP) estimate from DESeq
        dispDESeq = dispRes["log_alpha"]

        # MAP estimate using optim
        def logPost(log_alpha):
            alpha = np.exp(log_alpha)
            w = np.diag(1 / (1 / mu_hat**2 * (mu_hat + alpha * mu_hat**2)))
            logLike = np.sum(dnbinom_mu(y, mu=mu_hat, size=1 / alpha, log=True))
            coxReid = -0.5 * np.linalg.slogdet(x.T @ w @ x)[1]
            logPrior = dnorm(
                log_alpha,
                log_alpha_prior_mean,
                np.sqrt(log_alpha_prior_sigmasq),
                log=True,
            )
            res = logLike + coxReid + logPrior
            return res

        dispOptim = minimize(lambda p: -logPost(p), 0).x

        self.assertTrue(np.allclose(dispDESeq, dispOptim))

        # check derivatives:

        # from Ted Harding https://stat.ethz.ch/pipermail/r-help/2007-September/140013.html
        def num_deriv(f, x, h=0.001):
            return (f(x + h / 2) - f(x - h / 2)) / h

        def num_2nd_deriv(f, x, h=0.001):
            return (f(x + h) - 2 * f(x) + f(x - h)) / h**2

        # first derivative of log posterior wrt log alpha at start
        dispDerivDESeq = dispRes["initial_dlp"]
        dispDerivNum = num_deriv(logPost, 0)
        self.assertTrue(np.allclose(dispDerivDESeq, dispDerivNum))

        # second derivative at finish
        dispD2DESeq = dispRes["last_d2lp"]
        dispD2Num = num_2nd_deriv(logPost, dispRes["log_alpha"])
        self.assertTrue(np.allclose(dispD2DESeq, dispD2Num))

        # test fit alternative
        dds = makeExampleDESeqDataSet()
        dds = dds.estimateSizeFactors()
        # ddsLocal = dds.copy().estimateDispersions(fitType="local")
        dds.copy().estimateDispersions(fitType="mean")
        ddsMed = estimateDispersionsGeneEst(dds.copy())
        useForMedian = ddsMed.var["dispGeneEst"] > 1e-7
        medianDisp = np.nanmedian(ddsMed.var["dispGeneEst"][useForMedian])
        ddsMed.setDispFunction(lambda x: medianDisp)
        ddsMed = estimateDispersionsMAP(ddsMed)

        # test iterative
        dds = makeExampleDESeqDataSet(m=50, n=100, betaSD=1, interceptMean=8, seed=42)
        dds = dds.estimateSizeFactors()
        dds = estimateDispersionsGeneEst(dds, niter=5)
        dds = dds[:, ~dds.var["allZero"]]
        self.assertTrue(
            np.allclose(dds.var["trueDisp"], dds.var["dispGeneEst"], rtol=0.7)
        )

    def test_fitDispGridWrapper_pandas_series(self):
        """test that fitDispGridWrapper properly handles pandas Series input for log_alpha_prior_mean"""
        # Create test data
        n_samples = 4
        n_genes = 8
        y = np.random.poisson(20, size=(n_samples, n_genes))
        x = np.column_stack([np.ones(n_samples), np.repeat([0, 1], n_samples // 2)])
        mu = np.random.gamma(2, 10, size=(n_samples, n_genes))
        weights = np.ones((n_samples, n_genes))

        # Create a pandas Series with gene names as index
        gene_names = [f"gene_{i}" for i in range(n_genes)]
        log_alpha_prior_mean = pd.Series(
            np.random.normal(-1, 0.5, n_genes), index=gene_names, name="dispFit"
        )

        # Test that fitDispGridWrapper doesn't fail with pandas Series input
        try:
            result = fitDispGridWrapper(
                y=y,
                x=x,
                mu=mu,
                log_alpha_prior_mean=log_alpha_prior_mean,
                log_alpha_prior_sigmasq=1.0,
                usePrior=True,
                weights=weights,
                useWeights=False,
                weightThreshold=1e-2,
                useCR=True,
            )

            # Verify the result is a numpy array with correct shape
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, (n_genes,))
            self.assertTrue(np.all(np.isfinite(result)))
            self.assertTrue(np.all(result > 0))  # dispersions should be positive

        except Exception as e:
            self.fail(f"fitDispGridWrapper failed with pandas Series input: {e}")

        # Test that it also works with numpy array input (for comparison)
        try:
            result_numpy = fitDispGridWrapper(
                y=y,
                x=x,
                mu=mu,
                log_alpha_prior_mean=log_alpha_prior_mean.values,
                log_alpha_prior_sigmasq=1.0,
                usePrior=True,
                weights=weights,
                useWeights=False,
                weightThreshold=1e-2,
                useCR=True,
            )

            # Results should be similar (allowing for small numerical differences)
            self.assertTrue(np.allclose(result, result_numpy, rtol=1e-10))

        except Exception as e:
            self.fail(f"fitDispGridWrapper failed with numpy array input: {e}")
