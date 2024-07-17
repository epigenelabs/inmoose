import unittest

import numpy as np

from inmoose.edgepy import DGEList
from inmoose.utils import rnbinom


class test_predFC(unittest.TestCase):
    def setUp(self):
        y = np.array(rnbinom(80, size=5, mu=20, seed=42)).reshape((20, 4))
        y = np.vstack(([0, 0, 0, 0], [0, 0, 2, 2], y))
        self.group = np.array([1, 1, 2, 2])
        self.d = DGEList(counts=y, group=self.group, lib_size=np.arange(1001, 1005))

    def test_predFC(self):
        design = np.array([[1, 0], [1, 0], [0, 1], [0, 2]])
        with self.assertLogs("inmoose", level="WARNING") as logChecker:
            res = self.d.predFC(design=design)
        self.assertRegex(logChecker.output[0], "dispersion set to zero")
        ref = np.array(
            [
                [-12.969746, -11.384905],
                [-12.969746, -7.315955],
                [-5.500709, -4.270495],
                [-6.200135, -4.612279],
                [-5.346262, -5.081762],
                [-6.492589, -3.739418],
                [-5.828765, -3.784606],
                [-5.567436, -3.769349],
                [-6.097954, -4.416721],
                [-6.097954, -3.800063],
                [-5.154532, -4.247723],
                [-5.828765, -4.366038],
                [-5.154532, -4.099175],
                [-5.008310, -4.642870],
                [-5.601993, -3.914313],
                [-5.468468, -3.385668],
                [-5.749148, -4.642870],
                [-5.129109, -4.416721],
                [-5.008310, -3.897303],
                [-4.793386, -3.681757],
                [-5.601993, -3.710232],
                [-5.788407, -4.225364],
            ]
        )
        self.assertTrue(np.allclose(res, ref, atol=1e-6, rtol=0))
