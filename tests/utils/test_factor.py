import unittest

import numpy as np

from inmoose.utils import Factor, asfactor, gl


class test_factor(unittest.TestCase):
    def test_class(self):
        f = Factor([1, 1, 2, 3, 3])
        self.assertTrue(np.array_equal(f.__array__(), [1, 1, 2, 3, 3]))
        self.assertTrue(np.array_equal(f.categories, [1, 2, 3]))

        self.assertEqual(len(f), len(f.__array__()))
        self.assertEqual(f.nlevels(), len(f.categories))

        f = Factor([1, 3, 3])
        g = f.droplevels()
        self.assertTrue(np.array_equal(g.__array__(), [1, 3, 3]))
        self.assertTrue(np.array_equal(g.categories, [1, 3]))

    # def test_equality(self):
    #    f1 = Factor([1,2,3,3])
    #    f2 = Factor([1,2,2,3])
    #    f3 = Factor([1,2,3,3])

    #    self.assertTrue(f1 == f3)
    #    self.assertTrue(f3 == f1)
    #    self.assertFalse(f1 != f3)
    #    self.assertFalse(f3 != f1)
    #    self.assertFalse(f1 == f2)
    #    self.assertFalse(f2 == f1)
    #    self.assertTrue(f1 != f2)
    #    self.assertTrue(f2 != f1)

    #    self.assertFalse(f1 == 0)
    #    self.assertFalse(0 == f1)
    #    self.assertTrue(f1 != 0)
    #    self.assertTrue(0 != f1)

    def test_asfactor(self):
        f = Factor([1, 2, 3, 3])
        self.assertTrue(f is asfactor(f))
        self.assertTrue(
            np.array_equal(f.__array__(), asfactor([1, 2, 3, 3]).__array__())
        )

    def test_gl(self):
        for n in range(1, 6):
            for k in range(1, 6):
                with self.subTest("test_gl_sub", n=n, k=k):
                    f = gl(n, k)
                    self.assertEqual(f.nlevels(), n)
                    self.assertEqual(len(f), n * k)
                    self.assertTrue((f.__array__() == np.sort(f.__array__())).all())
                    for j in range(1, n + 1):
                        self.assertEqual((f.__array__() == j).sum(), k)
