import unittest

import numpy as np

from inmoose.limma import makeContrasts


class Test(unittest.TestCase):

    def test_makeContrasts(self):
        self.assertTrue(
            np.array_equal(
                makeContrasts(contrasts=["B-A", "C-B", "C-A"], levels=["A", "B", "C"]),
                [[-1, 0, -1], [1, -1, 0], [0, 1, 1]],
            )
        )
        self.assertTrue(
            np.array_equal(
                makeContrasts(contrasts="A-(B+C)/2", levels=["A", "B", "C"]),
                [[1.0], [-0.5], [-0.5]],
            )
        )
