import unittest

import numpy as np
import pandas as pd
from anndata import AnnData

from inmoose.pycombat import pycombat_norm
from inmoose.pycombat.covariates import make_design_matrix
from inmoose.pycombat.pycombat_norm import (
    adjust_data,
    calculate_mean_var,
    calculate_stand_mean,
    check_mean_only,
    compute_prior,
    fit_model,
    postmean,
    postvar,
    standardise_data,
)


class test_pycombat(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        matrix = np.transpose(
            [
                np.random.normal(size=1000, loc=3, scale=1),
                np.random.normal(size=1000, loc=3, scale=1),
                np.random.normal(size=1000, loc=3, scale=1),
                np.random.normal(size=1000, loc=2, scale=0.6),
                np.random.normal(size=1000, loc=2, scale=0.6),
                np.random.normal(size=1000, loc=4, scale=1),
                np.random.normal(size=1000, loc=4, scale=1),
                np.random.normal(size=1000, loc=4, scale=1),
                np.random.normal(size=1000, loc=4, scale=1),
            ]
        )

        self.matrix = pd.DataFrame(
            data=matrix,
            columns=["sample_" + str(i + 1) for i in range(9)],
            index=["gene_" + str(i + 1) for i in range(1000)],
        )

        # test normal execution
        self.batch = np.asarray([1, 1, 1, 2, 2, 3, 3, 3, 3])
        self.matrix_adjusted = pycombat_norm(self.matrix, self.batch)

        # useful constants for unit testing
        ref_batch = None
        mean_only = False
        par_prior = False
        precision = None
        mod = None
        self.dat = self.matrix.values
        vci = make_design_matrix(
            self.dat, self.batch, covar_mod=mod, ref_batch=ref_batch
        )
        self.dat = vci.counts
        self.design = np.transpose(vci.design)
        self.batchmod = vci.batch_mod
        self.batches = [vci.batch_composition[b] for b in vci.batch.categories]
        self.n_batches = [len(v) for v in self.batches]
        self.batch = vci.batch
        ref = vci.ref_batch_idx
        self.n_batch = vci.n_batch
        self.n_array = self.dat.shape[1]

        self.B_hat, self.grand_mean, self.var_pooled = calculate_mean_var(
            self.design,
            self.batches,
            ref,
            self.dat,
            self.n_batches,
            self.n_batch,
            self.n_array,
        )
        self.stand_mean = calculate_stand_mean(
            self.grand_mean, self.n_array, self.design, self.n_batch, self.B_hat
        )
        self.s_data = standardise_data(
            self.dat, self.stand_mean, self.var_pooled, self.n_array
        )
        self.gamma_star, self.delta_star, self.batch_design = fit_model(
            self.design,
            self.n_batch,
            self.s_data,
            self.batches,
            mean_only,
            par_prior,
            precision,
            ref,
        )
        self.bayes_data = adjust_data(
            self.s_data,
            self.gamma_star,
            self.delta_star,
            self.batch_design,
            self.n_batches,
            self.var_pooled,
            self.stand_mean,
            self.n_array,
            ref,
            self.batches,
            self.dat,
        )

    def test_compute_prior(self):
        print("aprior", compute_prior("a", self.gamma_star, False))
        self.assertEqual(compute_prior("a", self.gamma_star, True), 1)
        print("bprior", compute_prior("b", self.gamma_star, False))
        self.assertEqual(compute_prior("b", self.gamma_star, True), 1)

    def test_postmean(self):
        self.assertEqual(
            np.shape(
                postmean(
                    self.gamma_star, self.delta_star, self.gamma_star, self.delta_star
                )
            ),
            np.shape(self.gamma_star),
        )

    def test_postvar(self):
        self.assertEqual(np.sum(postvar([2, 4, 6], 2, 1, 1) - [2, 3, 4]), 0)

    # def test_it_sol(self):
    #    ()

    # def test_int_eprior(self):
    #    ()

    @unittest.skip("automate the verification")
    def test_check_mean_only(self):
        check_mean_only(True)
        check_mean_only(False)
        print("Only one line of text should have been printed above.")

    def test_calculate_mean_var(self):
        self.assertEqual(np.shape(self.B_hat)[0], np.shape(self.design)[0])
        self.assertEqual(np.shape(self.grand_mean)[0], np.shape(self.dat)[0])
        self.assertEqual(np.shape(self.var_pooled)[0], np.shape(self.dat)[0])

    def test_calculate_stand_mean(self):
        self.assertEqual(np.shape(self.stand_mean), np.shape(self.dat))

    def test_standardise_data(self):
        self.assertEqual(np.shape(self.s_data), np.shape(self.dat))

    def test_fit_model(self):
        self.assertEqual(np.shape(self.gamma_star)[1], np.shape(self.dat)[0])
        self.assertEqual(np.shape(self.delta_star)[1], np.shape(self.dat)[0])
        self.assertEqual(np.shape(self.batch_design), np.shape(self.design))

    def test_adjust_data(self):
        self.assertEqual(np.shape(self.bayes_data), np.shape(self.dat))

    def test_pycombat(self):
        self.assertEqual(np.shape(self.matrix), np.shape(self.matrix_adjusted))
        self.assertTrue(
            np.mean(self.matrix.values) - np.mean(self.matrix_adjusted.values)
            <= np.mean(self.matrix.values) * 0.05
        )
        self.assertTrue(
            np.var(self.matrix_adjusted.values) <= np.var(self.matrix.values)
        )
        # test raise error for single sample batch
        with self.assertRaisesRegex(
            ValueError,
            r"Batches 4 contain a single sample, which is not supported for batch effect correction. Please review your inputs.",
        ):
            pycombat_norm(self.matrix, np.asarray([1, 1, 1, 2, 2, 3, 3, 3, 4]))

    def test_pycombat_anndata(self):
        ad = AnnData(
            self.matrix.T,
            obs=pd.DataFrame({"batch": self.batch}, index=self.matrix.columns),
        )
        res = pycombat_norm(ad, batch="batch")
        self.assertTrue(np.allclose(res.X.T, self.matrix_adjusted))
        with self.assertRaisesRegex(
            ValueError, 'the batch column "foo" must appear in'
        ):
            pycombat_norm(ad, batch="foo")
