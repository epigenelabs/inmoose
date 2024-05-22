import unittest

import numpy as np
import pandas as pd
import scipy.stats
from scipy.optimize import minimize

from inmoose.deseq2 import DESeqDataSet
from inmoose.deseq2.fitNbinomGLMs import fitNbinomGLMs
from inmoose.utils import Factor, dnbinom_mu, dnorm


class Test(unittest.TestCase):
    def test_betaFitting(self):
        """test that estimates of beta fit from various methods are equal"""
        m = 10
        y = scipy.stats.poisson.rvs(20, size=m, random_state=42)
        sf = np.ones(m)
        condition = Factor(
            [0 for i in range(int(m / 2))] + [1 for i in range(int(m / 2))]
        )
        x = np.hstack([np.repeat(1, m), np.repeat(0, m / 2), np.repeat(1, m / 2)])
        x = np.ones((m, 2))
        x[: int(m / 2), 1] = 0
        lambda_ = 2
        alpha = 0.5

        dds = DESeqDataSet(
            y[:, None],
            clinicalData=pd.DataFrame({"condition": condition}),
            design="~condition",
        )
        dds.sizeFactors = sf
        dds.var["dispersion"] = alpha
        dds.var["baseMean"] = np.mean(y)

        # for testing we convert beta to the natural log scale:
        # convert lambda_ from log to log2 scale by multiplying by log(2)**2
        # then convert beta back from log2 to log scale by multiplying by log(2)
        betaDESeq = (
            np.log(2)
            * fitNbinomGLMs(dds, lambda_=[0, lambda_ * np.log(2) ** 2])[
                "betaMatrix"
            ].iloc[0, :]
        )

        # the IRLS algorithm
        betaIRLS = np.array([1, 1])
        for t in range(100):
            mu_hat = sf * np.exp(x @ betaIRLS)
            w = np.diag(1 / (1 / mu_hat**2 * (mu_hat + alpha * mu_hat**2)))
            z = np.log(mu_hat / sf) + (y - mu_hat) / mu_hat
            ridge = np.diag([0, lambda_])
            betaIRLS = np.linalg.inv(x.T @ w @ x + ridge) @ x.T @ w @ z

        # using minimize
        def objectiveFn(p):
            mu = np.exp(x @ p)
            logLike = np.sum(dnbinom_mu(y, mu=mu, size=1 / alpha, log=True))
            prior = dnorm(p[1], 0, np.sqrt(1 / lambda_), log=True)
            return -(logLike + prior)

        betaOptim = minimize(objectiveFn, np.array([0.1, 0.1])).x

        self.assertTrue(np.allclose(betaDESeq, betaIRLS))
        self.assertTrue(np.allclose(betaDESeq, betaOptim))
