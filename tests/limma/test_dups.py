import unittest

import numpy as np

from inmoose.limma import uniquegenelist, unwrapdups


class Test(unittest.TestCase):
    def test_unwrapdups(self):
        M = np.arange(12).reshape((4, 3))
        self.assertTrue(np.array_equal(unwrapdups(M, ndups=1), M))
        self.assertTrue(np.array_equal(unwrapdups(M, ndups=1, spacing=2), M))
        self.assertTrue(
            np.array_equal(unwrapdups(M), [[0, 3, 1, 4, 2, 5], [6, 9, 7, 10, 8, 11]])
        )
        self.assertTrue(
            np.array_equal(
                unwrapdups(M, ndups=4), [[0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]]
            )
        )
        self.assertTrue(
            np.array_equal(
                unwrapdups(M, ndups=2, spacing=2),
                [[0, 6, 1, 7, 2, 8], [3, 9, 4, 10, 5, 11]],
            )
        )

    def test_uniquegenelist(self):
        M = np.arange(12).reshape((4, 3))
        self.assertTrue(np.array_equal(uniquegenelist(M, ndups=1), M))
        self.assertTrue(np.array_equal(uniquegenelist(M, ndups=1, spacing=2), M))
        self.assertTrue(np.array_equal(uniquegenelist(M), [[0, 1, 2], [6, 7, 8]]))
        self.assertTrue(np.array_equal(uniquegenelist(M, ndups=4), [[0, 1, 2]]))
        self.assertTrue(
            np.array_equal(
                uniquegenelist(M, ndups=2, spacing=2), [[0, 1, 2], [3, 4, 5]]
            )
        )

        M = np.arange(6)
        self.assertTrue(np.array_equal(uniquegenelist(M, ndups=1), M))
        self.assertTrue(np.array_equal(uniquegenelist(M, ndups=1, spacing=2), M))
        self.assertTrue(np.array_equal(uniquegenelist(M), [0, 2, 4]))
        self.assertTrue(np.array_equal(uniquegenelist(M, ndups=3), [0, 3]))
        self.assertTrue(np.array_equal(uniquegenelist(M, ndups=3, spacing=2), [0, 1]))
        self.assertTrue(
            np.array_equal(uniquegenelist(M, ndups=2, spacing=3), [0, 1, 2])
        )
