import unittest

import numpy as np

from inmoose.edgepy import q2qnbinom


class Test(unittest.TestCase):
    def test_q2qnbinom(self):
        x = np.arange(20, dtype=float)
        input_mean = 10.0
        output_mean = 20.0
        dispersion = 0.1

        ref = np.array(
            [
                1.339746,
                3.720337,
                5.706665,
                7.602321,
                9.448944,
                11.2636,
                13.05531,
                14.82956,
                16.58996,
                18.33906,
                20.07871,
                21.81033,
                23.53501,
                25.25363,
                26.96691,
                28.67544,
                30.37969,
                32.0801,
                33.777,
                35.47071,
            ]
        )

        res = q2qnbinom(
            x, input_mean=input_mean, output_mean=output_mean, dispersion=dispersion
        )
        self.assertTrue(np.allclose(res, ref))
