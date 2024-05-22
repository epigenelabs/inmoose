import unittest

import numpy as np

from inmoose.edgepy import DGEList, adjustedProfileLik
from inmoose.utils import rnbinom


class test_APL(unittest.TestCase):
    def setUp(self):
        y = np.array(rnbinom(80, size=5, mu=20, seed=42)).reshape((20, 4))
        y = np.vstack(([0, 0, 0, 0], [0, 0, 2, 2], y))
        self.group = np.array([1, 1, 2, 2])
        self.d = DGEList(counts=y, group=self.group, lib_size=np.arange(1001, 1005))

    def test_adjustedProfileLik(self):
        apl = adjustedProfileLik(
            0.05,
            self.d.counts,
            np.ones((self.d.counts.shape[1], 1)),
            self.d.getOffset(),
        )
        ref = [
            np.inf,
            -6.052019642,
            -14.4426662,
            -12.70072688,
            -23.58414577,
            -17.64428625,
            -15.06042351,
            -18.9616632,
            -12.42791699,
            -18.5664074,
            -15.50008961,
            -13.15651905,
            -18.58032921,
            -16.07588861,
            -14.2002113,
            -15.92987782,
            -12.68396098,
            -24.62550341,
            -17.70222029,
            -15.50752335,
            -14.39845558,
            -13.04470154,
        ]
        self.assertTrue(np.allclose(apl, ref, atol=0, rtol=1e-9))
