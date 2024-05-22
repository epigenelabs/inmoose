import unittest

import numpy as np

from inmoose.edgepy.makeCompressedMatrix import makeCompressedMatrix


class Test(unittest.TestCase):
    def test_makeCompressedMatrix(self):
        self.assertTrue(
            np.array_equal(
                makeCompressedMatrix(42, dims=(2, 3), byrow=True), np.full((2, 3), 42)
            )
        )
        self.assertTrue(
            np.array_equal(
                makeCompressedMatrix(42, dims=(2, 3), byrow=False), np.full((2, 3), 42)
            )
        )

        ref = np.array([[1, 2, 3] for i in range(2)])
        self.assertTrue(
            np.array_equal(
                makeCompressedMatrix([1, 2, 3], dims=(2, 3), byrow=True), ref
            )
        )
        self.assertTrue(
            np.array_equal(
                makeCompressedMatrix([1, 2, 3], dims=(3, 2), byrow=False), ref.T
            )
        )

        with self.assertRaisesRegex(
            ValueError, expected_regex="dims\[.\] should be equal to length of x"
        ):
            makeCompressedMatrix([1, 2, 3], dims=(3, 2), byrow=True)
        with self.assertRaisesRegex(
            ValueError, expected_regex="dims\[.\] should be equal to length of x"
        ):
            makeCompressedMatrix([1, 2, 3], dims=(2, 3), byrow=False)
        with self.assertRaisesRegex(
            ValueError, expected_regex="dims does not represent the shape of a matrix"
        ):
            makeCompressedMatrix(42, dims=(5,))
        with self.assertRaisesRegex(
            ValueError, expected_regex="dims does not represent the shape of a matrix"
        ):
            makeCompressedMatrix(42, dims=(1, 2, 3))
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="input has too many dimensions to be interpreted as a matrix",
        ):
            makeCompressedMatrix(np.ones(shape=(1, 2, 3)), dims=(2, 3))


if __name__ == "__main__":
    unittest.main()
