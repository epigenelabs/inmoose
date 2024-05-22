import unittest

import numpy as np
import pandas as pd
import patsy

from inmoose.pycombat.covariates import VirtualCohortInput, make_design_matrix


class Test(unittest.TestCase):
    def test_VirtualCohortInput(self):
        counts = np.ones((100, 5))
        batch = np.array([1, 1, 2, 2, 2])

        vci = VirtualCohortInput(counts, batch)
        self.assertEqual(vci.nan_batch, False)
        self.assertEqual(vci.nan_genes, 0)
        self.assertEqual(vci.nan_samples, 0)
        self.assertEqual(len(vci.batch_composition), 2)
        self.assertTrue(np.array_equal(vci.batch_composition[1], [0, 1]))
        self.assertTrue(np.array_equal(vci.batch_composition[2], [2, 3, 4]))
        self.assertEqual(vci.n_batch, 2)
        self.assertEqual(vci.nan_cov, 0)
        self.assertEqual(vci.nan_cov_samples, {1: 0, 2: 0})
        self.assertEqual(vci.confounded_cov, [])
        self.assertTrue(vci.batch_mod is not None)
        self.assertTrue(isinstance(vci.covar_mod, patsy.DesignMatrix))
        self.assertTrue(vci.design is not None)

        counts[0, 0] = np.nan
        counts[0, 1] = np.nan
        vci = VirtualCohortInput(counts, batch)
        self.assertEqual(vci.nan_batch, False)
        self.assertEqual(vci.nan_genes, 1)
        self.assertEqual(vci.nan_samples, 2)
        self.assertEqual(len(vci.batch_composition), 2)
        self.assertTrue(np.array_equal(vci.batch_composition[1], [0, 1]))
        self.assertTrue(np.array_equal(vci.batch_composition[2], [2, 3, 4]))
        self.assertEqual(vci.n_batch, 2)
        self.assertEqual(vci.nan_cov, 0)
        self.assertEqual(vci.nan_cov_samples, {1: 0, 2: 0})
        self.assertEqual(vci.confounded_cov, [])
        self.assertTrue(vci.batch_mod is not None)
        self.assertTrue(isinstance(vci.covar_mod, patsy.DesignMatrix))
        self.assertTrue(vci.design is not None)

        vci = VirtualCohortInput(counts, [1, 1, np.nan, 2, 2])
        self.assertEqual(vci.nan_batch, True)
        self.assertEqual(len(vci.batch_composition), 2)
        self.assertTrue(np.array_equal(vci.batch_composition[1], [0, 1]))
        self.assertTrue(np.array_equal(vci.batch_composition[2], [3, 4]))
        self.assertEqual(vci.n_batch, 2)
        self.assertEqual(vci.nan_cov, 0)
        self.assertEqual(vci.nan_cov_samples, {1: 0, 2: 0})
        self.assertEqual(vci.confounded_cov, [])
        self.assertTrue(vci.batch_mod is None)
        self.assertTrue(isinstance(vci.covar_mod, patsy.DesignMatrix))
        self.assertTrue(vci.design is None)

        vci = VirtualCohortInput(counts, batch, ["a", "b", "c", "d", "e"])
        self.assertEqual(vci.nan_cov, 0)
        self.assertEqual(vci.nan_cov_samples, {1: 0, 2: 0})
        self.assertEqual(vci.confounded_cov, [])
        self.assertTrue(vci.batch_mod is not None)
        self.assertTrue(isinstance(vci.covar_mod, patsy.DesignMatrix))
        self.assertTrue(vci.design is not None)

        vci = VirtualCohortInput(counts, batch, ["a", "b", np.nan, "d", "e"])
        self.assertEqual(vci.nan_cov, 1)
        self.assertEqual(vci.nan_cov_samples, {1: 0, 2: 1})
        self.assertEqual(vci.confounded_cov, None)
        self.assertTrue(vci.batch_mod is not None)
        self.assertTrue(isinstance(vci.covar_mod, pd.DataFrame))
        self.assertTrue(vci.design is None)

        vci = VirtualCohortInput(counts, batch, ["a", "a", "b", "b", "b"])
        self.assertEqual(vci.nan_cov, 0)
        self.assertEqual(vci.nan_cov_samples, {1: 0, 2: 0})
        self.assertEqual(vci.confounded_cov, [1])

    def test_make_design_matrix(self):
        batch = np.asarray([1, 1, 1, 2, 2, 3, 3, 3, 3])

        counts = np.ones((100, 5))
        vci = make_design_matrix(counts, [0, 0, 0, 0, 0], covar_mod=[1, 1, 0, 1, 0])
        self.assertEqual(vci.covar_mod.shape, (5, 2))
        self.assertTrue(
            np.array_equal(
                vci.covar_mod, np.array([[1, 1], [1, 1], [1, 0], [1, 1], [1, 0]])
            )
        )

        counts = np.ones((100, 9))
        vci = make_design_matrix(counts, batch, None, None)
        self.assertEqual(vci.batch_mod.shape, (9, 3))

        # test batches
        self.assertEqual(vci.n_batch, 3)
        self.assertEqual(vci.batch_composition[1].tolist(), [0, 1, 2])
        self.assertEqual(vci.batch_composition[2].tolist(), [3, 4])
        self.assertEqual(vci.batch_composition[3].tolist(), [5, 6, 7, 8])

        # test covariates
        self.assertEqual(np.sum(vci.design - vci.batch_mod), 0)

        # test reference batch
        self.assertEqual(
            0, make_design_matrix(counts, batch, ref_batch=1).ref_batch_idx
        )
        self.assertEqual(
            1, make_design_matrix(counts, batch, ref_batch=2).ref_batch_idx
        )
        self.assertEqual(None, make_design_matrix(counts, batch).ref_batch_idx)
