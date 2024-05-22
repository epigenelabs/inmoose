import unittest

from inmoose.edgepy import pnbinom, qnbinom


class test_nbinom(unittest.TestCase):
    def test_nbinom(self):
        for i in range(10):
            hi = pnbinom(i, size=5, prob=0.2, lower_tail=False)
            lo = pnbinom(i, size=5, prob=0.2, lower_tail=True)
            self.assertEqual(hi, 1.0 - lo)
            self.assertEqual(i, qnbinom(hi, size=5, prob=0.2, lower_tail=False))
            self.assertEqual(i, qnbinom(lo, size=5, prob=0.2, lower_tail=True))

            hi = pnbinom(i, size=5, mu=4, lower_tail=False)
            lo = pnbinom(i, size=5, mu=4, lower_tail=True)
            self.assertAlmostEqual(hi, 1.0 - lo)
            self.assertEqual(i, qnbinom(hi, size=5, mu=4, lower_tail=False))
            self.assertEqual(i, qnbinom(lo, size=5, mu=4, lower_tail=True))

        ctxt = self.assertRaisesRegex(
            ValueError, expected_regex="exactly one of prob and mu must be provided"
        )
        with ctxt:
            pnbinom(42, size=5, prob=0.5, mu=4)
        with ctxt:
            qnbinom(42, size=5, prob=0.5, mu=4)
        with ctxt:
            pnbinom(42, size=5)
        with ctxt:
            qnbinom(42, size=5)
