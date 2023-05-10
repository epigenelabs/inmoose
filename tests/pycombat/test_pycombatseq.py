import unittest
import numpy as np

from inmoose.utils import rnbinom
from inmoose.batch import pycombat_seq

class test_pycombatseq(unittest.TestCase):

    def setUp(self):
        y = np.array(rnbinom(80, size=5, mu=20, seed=42)).reshape((20,4))
        self.y = np.vstack(([0,0,0,0], [0,0,2,2], y))
        self.batch = np.array([1,1,2,2])

    def test_pycombat_seq(self):
        ref = np.array([
            [ 0, 0, 0, 0],
            [ 0, 0, 2, 2],
            [24,14,15,26],
            [10,18,11,17],
            [ 5,30,35, 6],
            [22,14,16,19],
            [25,20,27,20],
            [12,38,19,23],
            [17,16,18,15],
            [10,30,11,26],
            [28,18,17,29],
            [13,22,16,20],
            [31,12,13,35],
            [21,22,25,20],
            [27,18,20,26],
            [26,32,30,30],
            [15,18,16,18],
            [ 9,42,41, 9],
            [40,20,45,19],
            [38,30,42,28],
            [25,23,21,29],
            [18,21,22,18]])
        res = pycombat_seq(self.y, self.batch)
        self.assertTrue(np.array_equal(res, ref))

        ref = np.array([
            [ 0, 0, 0, 0],
            [ 0, 0, 2, 2],
            [18,13,16,34],
            [ 9,15,12,21],
            [15,36,27, 2],
            [23,11,19,20],
            [27,20,26,19],
            [ 9,40,25,25],
            [17,16,18,15],
            [ 7,29,19,25],
            [25,15,21,33],
            [12,20,17,22],
            [17,11,15,60],
            [22,24,24,18],
            [27,16,23,26],
            [26,32,31,31],
            [14,17,16,19],
            [10,53,33, 8],
            [46,24,38,16],
            [40,34,40,25],
            [22,21,22,33],
            [19,21,22,17]])
        res = pycombat_seq(self.y, self.batch, group=[1,1,1,2])
        self.assertTrue(np.array_equal(res, ref))

        # test with reference batch
        res = pycombat_seq(self.y, self.batch, ref_batch=1)
        ref_col = self.batch == 1
        non_ref_col = self.batch != 1
        # make sure that reference batch counts have not been adjusted
        self.assertTrue(np.array_equal(res[:,ref_col], self.y[:,ref_col]))
        # make sure that batch effects have still been corrected
        self.assertFalse(np.array_equal(res[:,non_ref_col], self.y[:,non_ref_col]))

        # also test with non-integer batch ids
        res2 = pycombat_seq(self.y, ['a', 'a', 'b', 'b'], ref_batch='a')
        self.assertTrue(np.array_equal(res, res2))

        # test with incomplete group
        ref = pycombat_seq(self.y, self.batch, group=[1,2,1,3])
        with self.assertWarnsRegex(UserWarning, r"\d+ missing covariates in group. Creating a distinct covariate per batch for the missing values. You may want to double check your covariates."):
            res = pycombat_seq(self.y, self.batch, group=[1,np.nan,1,np.nan])
        self.assertTrue(np.array_equal(res, ref))
