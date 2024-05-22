import unittest

import numpy as np
from patsy import dmatrix

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

        design = dmatrix("0+A+B", {"A": ["a1", "a2", "a1"], "B": ["b1", "b2", "b3"]})
        self.assertTrue(
            np.array_equal(
                makeContrasts(contrasts="A[a1]-A[a2]", levels=design),
                [[1.0], [-1.0], [0.0], [0.0]],
            )
        )
