import unittest
import numpy as np

from inmoose.pycombat.covariates import make_design_matrix


class Test(unittest.TestCase):
    def test_make_design_matrix(self):
        batch = np.asarray([1, 1, 1, 2, 2, 3, 3, 3, 3])

        counts = np.ones((100, 5))
        _, _, mod, _, _, _, _, _, _, _ = make_design_matrix(
            counts, [0, 0, 0, 0, 0], [1, 1, 0, 1, 0], None
        )
        self.assertEqual(mod.shape, (5, 2))
        self.assertTrue(
            np.array_equal(mod, np.array([[1, 1], [1, 1], [1, 0], [1, 1], [1, 0]]))
        )

        counts = np.ones((100, 9))
        (
            design,
            batchmod,
            _,
            batches,
            n_batches,
            n_batch,
            n_sample,
            _,
            _,
            _,
        ) = make_design_matrix(counts, batch, None, None)
        self.assertEqual(batchmod.shape, (9, 3))

        # test batches
        self.assertEqual(n_batch, 3)
        self.assertEqual(batches[0].tolist(), [0, 1, 2])
        self.assertEqual(batches[1].tolist(), [3, 4])
        self.assertEqual(batches[2].tolist(), [5, 6, 7, 8])
        self.assertEqual(n_batches, [3, 2, 4])
        self.assertEqual(n_sample, 9)

        # test covariates
        self.assertEqual(np.sum(design - batchmod), 0)

        # test reference batch
        self.assertEqual(0, make_design_matrix(counts, batch, None, 1)[7])
        self.assertEqual(1, make_design_matrix(counts, batch, None, 2)[7])
        self.assertEqual(None, make_design_matrix(counts, batch, None, None)[7])
