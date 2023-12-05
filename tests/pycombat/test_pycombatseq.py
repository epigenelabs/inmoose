import unittest
import numpy as np

from inmoose.utils import rnbinom
from inmoose.pycombat import pycombat_seq


class test_pycombatseq(unittest.TestCase):
    def setUp(self):
        y = np.array(rnbinom(100, size=5, mu=20, seed=42)).reshape((20, 5))
        self.y = np.vstack(([0, 0, 0, 0, 0], [0, 0, 2, 2, 2], y))
        self.batch = np.array([1, 1, 2, 2, 2])

    def test_pycombat_seq(self):
        ref = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 2, 2, 2],
                [21, 14, 14, 26, 14],
                [24, 13, 15, 13, 28],
                [29, 5, 13, 3, 30],
                [30, 20, 16, 33, 26],
                [12, 36, 31, 26, 8],
                [10, 20, 17, 8, 24],
                [18, 31, 34, 24, 18],
                [23, 14, 22, 15, 19],
                [32, 15, 11, 32, 33],
                [27, 15, 15, 30, 17],
                [25, 28, 20, 25, 36],
                [28, 14, 26, 15, 18],
                [12, 47, 36, 6, 42],
                [23, 39, 15, 44, 34],
                [31, 20, 26, 24, 26],
                [29, 17, 21, 26, 18],
                [37, 20, 14, 37, 33],
                [21, 19, 13, 28, 21],
                [23, 14, 16, 14, 25],
                [13, 33, 16, 45, 11],
            ]
        )
        res = pycombat_seq(self.y, self.batch)
        self.assertTrue(np.array_equal(res, ref))

        ref = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 2, 2, 2],
                [19, 14, 14, 29, 13],
                [22, 13, 15, 13, 30],
                [34, 2, 15, 8, 25],
                [24, 18, 17, 41, 24],
                [11, 43, 24, 25, 12],
                [14, 21, 16, 7, 23],
                [31, 39, 29, 19, 10],
                [28, 18, 18, 12, 17],
                [17, 12, 11, 45, 47],
                [24, 14, 16, 33, 17],
                [24, 26, 20, 22, 42],
                [34, 18, 21, 13, 16],
                [6, 67, 33, 10, 35],
                [19, 31, 20, 51, 33],
                [34, 23, 23, 21, 24],
                [30, 18, 19, 26, 16],
                [32, 17, 19, 38, 32],
                [17, 16, 14, 36, 20],
                [23, 14, 15, 13, 25],
                [8, 30, 18, 49, 15],
            ]
        )
        res = pycombat_seq(self.y, self.batch, covar_mod=[1, 1, 1, 2, 2])
        self.assertTrue(np.array_equal(res, ref))

        # test with reference batch
        res = pycombat_seq(self.y, self.batch, ref_batch=1)
        ref_col = self.batch == 1
        non_ref_col = self.batch != 1
        # make sure that reference batch counts have not been adjusted
        self.assertTrue(np.array_equal(res[:, ref_col], self.y[:, ref_col]))
        # make sure that batch effects have still been corrected
        self.assertFalse(np.array_equal(res[:, non_ref_col], self.y[:, non_ref_col]))

        # also test with non-integer batch ids
        res2 = pycombat_seq(self.y, ["a", "a", "b", "b", "b"], ref_batch="a")
        self.assertTrue(np.array_equal(res, res2))

        # test raise error for single sample batch
        with self.assertRaisesRegex(
            ValueError, expected_regex="Batches a contain a single sample"
        ):
            pycombat_seq(self.y, ["a", "b", "b", "b", "b"])

        # test with incomplete group
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="2 values are missing in covariates cov_0. Correct your covariates or use the cov_missing_value parameters",
        ):
            pycombat_seq(self.y, self.batch, covar_mod=["a", np.nan, "a", np.nan, "a"])

        ref = pycombat_seq(self.y, self.batch, covar_mod=["a", "b", "a", "c", "a"])
        with self.assertWarnsRegex(
            UserWarning,
            r"1 missing covariates in covar_mod. Creating a distinct covariate per batch for the missing values. You may want to double check your covariates.",
        ):
            res = pycombat_seq(
                self.y,
                self.batch,
                covar_mod=[1, 2, 1, np.nan, 1],
                na_cov_action="fill",
            )
        self.assertTrue(np.array_equal(res, ref))

        with self.assertWarnsRegex(
            UserWarning,
            r"1 samples with missing covariates in covar_mod. They are removed from the data. You may want to double check your covariates.",
        ):
            res = pycombat_seq(
                self.y,
                self.batch,
                covar_mod=[1, 2, 1, np.nan, 1],
                na_cov_action="remove",
            )
        ref_y = np.delete(self.y, (3), axis=1)
        ref_batch = np.array([1, 1, 2, 2])
        ref = pycombat_seq(ref_y, ref_batch, covar_mod=[1, 2, 1, 1])
        self.assertTrue(np.array_equal(res, ref))

        # test error/warning message for data type of covariates
        with self.assertRaisesRegex(
            ValueError,
            expected_regex=r"Cannot create new categories for numerical covariates cov_0. Please fix the NA in those covariates manually.",
        ):
            pycombat_seq(
                self.y,
                self.batch,
                covar_mod=[1, 2.9, 1, 1, np.nan],
                na_cov_action="fill",
            )
