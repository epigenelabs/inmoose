import unittest
import numpy as np

from inmoose.utils import rnbinom
from inmoose.edgepy import addPriorCount, DGEList, glmFit, nbinomDeviance


class test_DGEGLM(unittest.TestCase):
    def test_constructor(self):
        from inmoose.edgepy import DGEGLM

        d = DGEGLM((1, 2, 3, 4, 5))

        self.assertIsNotNone(d.coefficients)
        self.assertIsNotNone(d.fitted_values)
        self.assertIsNotNone(d.deviance)
        self.assertIsNotNone(d.iter)
        self.assertIsNotNone(d.failed)

        self.assertIsNone(d.counts)
        self.assertIsNone(d.design)
        self.assertIsNone(d.offset)
        self.assertIsNone(d.dispersion)
        self.assertIsNone(d.weights)
        self.assertIsNone(d.prior_count)
        self.assertIsNone(d.unshrunk_coefficients)
        self.assertIsNone(d.method)


class test_nbinomDeviance(unittest.TestCase):
    def setUp(self):
        y = np.array(rnbinom(80, size=5, mu=20, seed=42)).reshape((20, 4))
        self.y = np.vstack(([0, 0, 0, 0], [0, 0, 2, 2], y))

    def test_nbinomDeviance(self):
        dev_ref = np.array(
            [
                7.999999,
                5.545177,
                321.818901,
                184.761910,
                355.515748,
                340.390858,
                414.784086,
                461.324010,
                235.746631,
                354.718673,
                410.643778,
                265.824737,
                422.344282,
                412.896876,
                392.316278,
                599.686176,
                242.088733,
                501.330767,
                608.676351,
                706.581453,
                430.474599,
                310.504051,
            ]
        )
        dev = nbinomDeviance(self.y, np.ones(self.y.shape))
        self.assertTrue(np.allclose(dev, dev_ref, atol=0, rtol=10 - 9))

        dev_ref = np.array([8.454617])
        dev = nbinomDeviance(np.array([1, 2, 3, 4]), np.ones(4), dispersion=np.zeros(4))
        self.assertTrue(np.allclose(dev, dev_ref, atol=0, rtol=10 - 9))

        with self.assertRaisesRegex(
            ValueError, expected_regex="y is a matrix but mean is not"
        ):
            nbinomDeviance(self.y, np.ones((22,)))

        with self.assertRaisesRegex(
            ValueError, expected_regex="mean is a matrix but y is not"
        ):
            nbinomDeviance(np.array([1, 2, 3, 4]), np.ones((4, 3)))

        with self.assertRaisesRegex(
            ValueError, expected_regex="length of mean differs from that of y"
        ):
            nbinomDeviance(np.array([1, 2, 3, 4]), np.ones(3))


class test_addPriorCount(unittest.TestCase):
    def setUp(self):
        y = np.array(rnbinom(80, size=5, mu=20, seed=42)).reshape((20, 4))
        self.y = np.vstack(([0, 0, 0, 0], [0, 0, 2, 2], y))

    def test_addPriorCount(self):
        ref = np.array(
            [
                [0.9275688, 1.042111, 1.053341, 0.9769792],
                [0.9275688, 1.042111, 3.053341, 2.9769792],
                [25.9275688, 20.042111, 12.053341, 23.9769792],
                [10.9275688, 18.042111, 11.053341, 17.9769792],
                [14.9275688, 36.042111, 28.053341, 2.9769792],
                [15.9275688, 8.042111, 29.053341, 29.9769792],
                [20.9275688, 16.042111, 34.053341, 25.9769792],
                [7.9275688, 36.042111, 29.053341, 28.9769792],
                [15.9275688, 15.042111, 20.053341, 16.9769792],
                [5.9275688, 25.042111, 23.053341, 30.9769792],
                [35.9275688, 22.042111, 15.053341, 22.9769792],
                [13.9275688, 23.042111, 16.053341, 19.9769792],
                [34.9275688, 23.042111, 8.053341, 29.9769792],
                [30.9275688, 33.042111, 18.053341, 13.9769792],
                [26.9275688, 16.042111, 24.053341, 26.9769792],
                [20.9275688, 26.042111, 39.053341, 38.9769792],
                [17.9275688, 21.042111, 14.053341, 15.9769792],
                [9.9275688, 49.042111, 36.053341, 8.9769792],
                [41.9275688, 22.042111, 43.053341, 17.9769792],
                [39.9275688, 34.042111, 41.053341, 25.9769792],
                [21.9275688, 21.042111, 23.053341, 33.9769792],
                [17.9275688, 20.042111, 24.053341, 18.9769792],
            ]
        )
        oref = np.full((22, 4), [6.027929, 6.144366, 6.155085, 6.079828])
        (yres, offsetres) = addPriorCount(self.y)
        self.assertTrue(np.allclose(yres, ref, atol=0, rtol=1e-6))
        self.assertTrue(np.allclose(offsetres, oref, atol=0, rtol=1e-6))

        with self.assertRaisesRegex(
            ValueError, expected_regex="count matrix must be numeric"
        ):
            addPriorCount(np.array(["foo", "bar"]))


class test_predFC(unittest.TestCase):
    def setUp(self):
        y = np.array(rnbinom(80, size=5, mu=20, seed=42)).reshape((20, 4))
        y = np.vstack(([0, 0, 0, 0], [0, 0, 2, 2], y))
        self.group = np.array([1, 1, 2, 2])
        self.d = DGEList(counts=y, group=self.group, lib_size=np.arange(1001, 1005))

    def test_predFC(self):
        design = np.array([[1, 0], [1, 0], [0, 1], [0, 2]])
        with self.assertWarnsRegex(Warning, expected_regex="dispersion set to zero"):
            res = self.d.predFC(design=design)
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


class test_glmFit(unittest.TestCase):
    def setUp(self):
        y = np.array(rnbinom(80, size=5, mu=20, seed=42)).reshape((20, 4))
        y = np.vstack(([0, 0, 0, 0], [0, 0, 2, 2], y))
        self.group = np.array([1, 1, 2, 2])
        self.d = DGEList(counts=y, group=self.group, lib_size=np.arange(1001, 1005))

    def test_glmFit(self):
        with self.assertRaisesRegex(
            ValueError, expected_regex="No dispersion values found in DGEList object"
        ):
            self.d.glmFit()
        # first estimate common dispersion
        self.d.estimateGLMCommonDisp()

        # test oneway method
        e = self.d.glmFit(prior_count=0)
        self.assertEqual(e.method, "oneway")
        coef_ref = np.array(
            [
                [-100000000, 0],
                [-100000000, 99999993.7818981],
                [-3.818158376, -0.2600061895],
                [-4.306653048, -0.001994836928],
                [-3.710751661, -0.5260476658],
                [-4.511242547, 0.9498904882],
                [-4.047000403, 0.5031039058],
                [-3.864988612, 0.2859441576],
                [-4.235093266, 0.1860767109],
                [-4.235334586, 0.5821202392],
                [-3.576947226, -0.4440123944],
                [-4.04714793, -0.03093068582],
                [-3.576961828, -0.4441419915],
                [-3.475280168, -0.7278715951],
                [-3.888726197, 0.1761259908],
                [-3.795782156, 0.5221193611],
                [-3.991513569, -0.2807027773],
                [-3.559630535, -0.2833225949],
                [-3.475132503, -0.05155169483],
                [-3.325699504, -0.1042127818],
                [-3.888819724, 0.2916754177],
                [-4.018902851, 0.1281254313],
            ]
        )
        self.assertTrue(np.allclose(e.coefficients, coef_ref, atol=1e-6, rtol=0))
        self.d.AveLogCPM = None
        e = self.d.glmFit(prior_count=0)
        self.assertTrue(np.allclose(e.coefficients, coef_ref, atol=1e-6, rtol=0))

        # test levenberg method
        design = np.array([[1, 0], [1, 0], [0, 1], [0, 2]])
        e = self.d.glmFit(design=design, prior_count=0)
        self.assertEqual(e.method, "levenberg")
        coef_ref = np.array(
            [
                [np.nan, np.nan],
                [-22.911238587272, -4.33912438169024],
                [-3.81815837585705, -2.1292021791932],
                [-4.30665304828008, -2.29993049236002],
                [-3.71075166096661, -3.34440560955535],
                [-4.51124254688311, -1.97663680068325],
                [-4.04700040309538, -2.04478756288351],
                [-3.86498861235895, -1.99475491866954],
                [-4.23509326552448, -2.31061055207476],
                [-4.23533458597804, -1.96831279885414],
                [-3.57694722597284, -2.1476132892472],
                [-4.04714792989704, -2.22576713012883],
                [-3.57696182768343, -2.00991843107384],
                [-3.47528016753611, -2.43228621567992],
                [-3.88872619677031, -2.04166124945388],
                [-3.79578215609008, -1.82484690967382],
                [-3.99151356863038, -2.36307176428644],
                [-3.55963053465756, -2.62529213670109],
                [-3.47513250262264, -2.22250430528328],
                [-3.32569950353846, -2.03272287830771],
                [-3.88881972448195, -1.91858334888063],
                [-4.0189028512534, -2.23681029760319],
            ]
        )
        self.assertTrue(
            np.allclose(e.coefficients, coef_ref, atol=1e-5, rtol=0, equal_nan=True)
        )

        with self.assertRaisesRegex(
            ValueError,
            expected_regex="design should have as many rows as y has columns",
        ):
            glmFit(self.d.counts, design=np.ones((5, 1)))

        with self.assertRaisesRegex(
            ValueError, expected_regex="No dispersion values provided"
        ):
            glmFit(self.d.counts, design=design)

        with self.assertRaisesRegex(
            ValueError,
            expected_regex="Dimensions of dispersion do not agree with dimensions of y",
        ):
            glmFit(self.d.counts, design=design, dispersion=np.ones((5, 1)))

        with self.assertRaisesRegex(
            ValueError,
            expected_regex="Dimensions of offset do not agree with dimensions of y",
        ):
            glmFit(
                self.d.counts, design=design, dispersion=0.05, offset=np.ones((5, 1))
            )

        with self.assertRaisesRegex(
            ValueError,
            expected_regex="lib_size has wrong length, should agree with ncol\(y\)",
        ):
            glmFit(
                self.d.counts, design=design, dispersion=0.05, lib_size=np.ones((2,))
            )

        e = glmFit(self.d.counts, dispersion=0.05, prior_count=0)
        coef_ref = np.array(
            [
                -1.000000e08,
                -6.099236e00,
                -3.120865e00,
                -3.495951e00,
                -3.138641e00,
                -3.114458e00,
                -2.953059e00,
                -2.908932e00,
                -3.325914e00,
                -3.096400e00,
                -2.953421e00,
                -3.252006e00,
                -2.951040e00,
                -2.960371e00,
                -2.980034e00,
                -2.691442e00,
                -3.309082e00,
                -2.895371e00,
                -2.686176e00,
                -2.563754e00,
                -2.917718e00,
                -3.142277e00,
            ]
        ).reshape(e.coefficients.shape)
        self.assertTrue(np.allclose(e.coefficients, coef_ref, atol=1e-6, rtol=0))
