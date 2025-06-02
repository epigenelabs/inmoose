import unittest

import numpy as np
import pandas as pd

from inmoose.edgepy import DGEList, glmFit, glmLRT, glmQLFTest, topTags
from inmoose.utils import rnbinom


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
        self.assertIsNone(d.AveLogCPM)


class test_glm(unittest.TestCase):
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

    def test_glmQLFit(self):
        with self.assertRaisesRegex(
            ValueError, expected_regex="No dispersion values found in DGEList object"
        ):
            self.d.glmQLFit()
        # first estimate common dispersion
        self.d.estimateGLMCommonDisp()

        e = self.d.glmQLFit()
        self.assertEqual(e.method, "oneway")
        coef_ref = np.array(
            [
                [-8.989943, 0.000000000],
                [-8.989943, 2.832275076],
                [-3.812746, -0.258338047],
                [-4.297698, -0.001976527],
                [-3.705921, -0.522525658],
                [-4.500199, 0.942978065],
                [-4.040138, 0.500298444],
                [-3.859316, 0.284480541],
                [-4.226767, 0.184626544],
                [-4.227016, 0.578352261],
                [-3.572745, -0.441539334],
                [-4.040290, -0.030706290],
                [-3.572760, -0.441672710],
                [-3.471510, -0.723582927],
                [-3.882900, 0.175143962],
                [-3.790498, 0.519873066],
                [-3.985036, -0.278532340],
                [-3.555513, -0.281880293],
                [-3.471359, -0.051338428],
                [-3.322486, -0.103831822],
                [-3.882996, 0.290140213],
                [-4.012239, 0.127298731],
            ]
        )
        self.assertTrue(np.allclose(e.coefficients, coef_ref, atol=1e-4, rtol=0))
        self.assertTrue(
            np.array_equal(
                [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                e.df_residual_zeros,
            )
        )
        self.assertAlmostEqual(e.df_prior, 4.672744, places=6)
        var_post_ref = np.array(
            [
                4.458200e-07,
                8.302426e-06,
                6.524738e-01,
                3.709944e-01,
                2.159878e00,
                6.208389e-01,
                5.878879e-01,
                1.325040e00,
                3.203313e-01,
                1.176470e00,
                6.941698e-01,
                4.743232e-01,
                1.205789e00,
                5.523578e-01,
                6.314750e-01,
                5.425240e-01,
                3.352875e-01,
                2.129557e00,
                9.924612e-01,
                5.895944e-01,
                5.991600e-01,
                4.650080e-01,
            ]
        )
        self.assertTrue(np.allclose(e.var_post, var_post_ref, atol=1e-6, rtol=0))
        var_prior_ref = np.array(
            [
                4.458200e-07,
                9.918589e-06,
                6.392936e-01,
                2.804475e-01,
                6.394279e-01,
                6.503286e-01,
                7.568130e-01,
                7.713810e-01,
                4.405790e-01,
                6.716242e-01,
                7.527023e-01,
                5.182746e-01,
                7.526737e-01,
                7.527380e-01,
                7.426247e-01,
                7.491848e-01,
                4.566134e-01,
                7.744945e-01,
                7.490555e-01,
                7.042350e-01,
                7.668183e-01,
                6.273684e-01,
            ]
        )
        self.assertTrue(np.allclose(e.var_prior, var_prior_ref, atol=1e-6, rtol=0))

    def test_glmQLFTest(self):
        # first estimate common dispersion
        self.d.estimateGLMCommonDisp()
        s = glmQLFTest(self.d.glmQLFit())
        table_ref = pd.DataFrame(
            {
                "log2FoldChange": [
                    0.00000,
                    4.086109e00,
                    -0.3727030,
                    -2.851525e-03,
                    -0.7538452,
                    1.36042978,
                    0.7217781,
                    0.4104187,
                    0.2663598,
                    0.8343859,
                    -0.6370066,
                    -0.04429981,
                    -0.6371990,
                    -1.04390950,
                    0.2526793,
                    0.7500183,
                    -0.4018372,
                    -0.4066673,
                    -0.07406570,
                    -0.14979766,
                    0.4185838,
                    0.1836532,
                ],
                "lfcSE": [
                    0.00000,
                    0.119306,
                    0.308323,
                    0.339974,
                    0.312279,
                    0.323978,
                    0.299188,
                    0.293219,
                    0.324064,
                    0.310603,
                    0.298936,
                    0.316756,
                    0.298942,
                    0.304455,
                    0.297726,
                    0.284142,
                    0.323617,
                    0.291959,
                    0.280812,
                    0.275328,
                    0.294518,
                    0.308360,
                ],
                "logCPM": [
                    10.95644,
                    1.154123e01,
                    14.3827970,
                    1.391051e01,
                    14.3829969,
                    14.39896457,
                    14.6144562,
                    14.6840429,
                    14.1263018,
                    14.4316968,
                    14.6005670,
                    14.22314458,
                    14.6004725,
                    14.60068510,
                    14.5711003,
                    14.9673456,
                    14.1463618,
                    14.7114142,
                    14.96790942,
                    15.13649877,
                    14.6566606,
                    14.3657820,
                ],
                "stat": [
                    0.000000e00,
                    5.998382e05,
                    4.845263e-01,
                    4.551851e-05,
                    5.906064e-01,
                    6.441791e00,
                    2.069610e00,
                    3.032625e-01,
                    4.811457e-01,
                    1.334095e00,
                    1.367280e00,
                    9.186695e-03,
                    7.875974e-01,
                    4.518866e00,
                    2.379445e-01,
                    2.538764e00,
                    1.047358e00,
                    1.860394e-01,
                    1.375613e-02,
                    9.649525e-02,
                    6.947198e-01,
                    1.651446e-01,
                ],
                "pvalue": [
                    1.00000000,
                    0.01861634,
                    0.50988854,
                    0.99481417,
                    0.46850481,
                    0.04035278,
                    0.19546361,
                    0.59978572,
                    0.51131874,
                    0.28775816,
                    0.28234748,
                    0.92645653,
                    0.40567730,
                    0.07303738,
                    0.64131120,
                    0.15720197,
                    0.34177834,
                    0.67982404,
                    0.91008476,
                    0.76555392,
                    0.43338141,
                    0.69717923,
                ],
            },
            index=[f"gene{i}" for i in range(22)],
        )
        pd.testing.assert_frame_equal(table_ref, s, check_frame_type=False, rtol=1e-4)

    def test_glmLRT(self):
        # first estimate common dispersion
        self.d.estimateGLMCommonDisp()
        s = glmLRT(self.d.glmFit())
        table_ref = pd.DataFrame(
            {
                "log2FoldChange": [
                    0.00000,
                    4.08610921,
                    -0.3727030,
                    -2.851525e-03,
                    -0.7538452,
                    1.36042978,
                    0.7217781,
                    0.4104187,
                    0.2663598,
                    0.8343859,
                    -0.6370066,
                    -0.044299812,
                    -0.6371990,
                    -1.043910,
                    0.2526793,
                    0.7500183,
                    -0.4018372,
                    -0.4066673,
                    -0.07406570,
                    -0.1497977,
                    0.4185838,
                    0.18365325,
                ],
                "lfcSE": [
                    0.00000,
                    0.119306,
                    0.308323,
                    0.339974,
                    0.312279,
                    0.323978,
                    0.299188,
                    0.293219,
                    0.324064,
                    0.310603,
                    0.298936,
                    0.316756,
                    0.298942,
                    0.304455,
                    0.297726,
                    0.284142,
                    0.323617,
                    0.291959,
                    0.280812,
                    0.275328,
                    0.294518,
                    0.308360,
                ],
                "logCPM": [
                    10.95644,
                    11.54122852,
                    14.3827970,
                    1.391051e01,
                    14.3829969,
                    14.39896457,
                    14.6144562,
                    14.6840429,
                    14.1263018,
                    14.4316968,
                    14.6005670,
                    14.223144579,
                    14.6004725,
                    14.600685,
                    14.5711003,
                    14.9673456,
                    14.1463618,
                    14.7114142,
                    14.96790942,
                    15.1364988,
                    14.6566606,
                    14.36578202,
                ],
                "stat": [
                    0.000000e00,
                    4.980113e00,
                    3.161407e-01,
                    1.688711e-05,
                    1.275638e00,
                    3.999315e00,
                    1.216698e00,
                    4.018350e-01,
                    1.541260e-01,
                    1.569523e00,
                    9.491243e-01,
                    4.357462e-03,
                    9.496767e-01,
                    2.496031e00,
                    1.502560e-01,
                    1.377340e00,
                    3.511659e-01,
                    3.961817e-01,
                    1.365242e-02,
                    5.689306e-02,
                    4.162483e-01,
                    7.679356e-02,
                ],
                "pvalue": [
                    1.00000000,
                    0.02564032,
                    0.57393623,
                    0.99672119,
                    0.25871164,
                    0.04551876,
                    0.27000955,
                    0.52614310,
                    0.69462318,
                    0.21027629,
                    0.32994229,
                    0.94736901,
                    0.32980161,
                    0.11413364,
                    0.69829083,
                    0.24055469,
                    0.55345388,
                    0.52906782,
                    0.90698400,
                    0.81147574,
                    0.51881503,
                    0.78169064,
                ],
            },
            index=[f"gene{i}" for i in range(22)],
        )
        pd.testing.assert_frame_equal(table_ref, s, check_frame_type=False, rtol=1e-4)

    def test_glmLRT_with_contrast(self):
        # first estimate common dispersion
        self.d.estimateGLMCommonDisp()
        s = glmLRT(self.d.glmFit(), contrast=np.array([1, 1]))
        table_ref = pd.DataFrame(
            {
                "log2FoldChange": [
                    -12.969746,
                    -8.883637,
                    -5.873327,
                    -6.203114,
                    -6.100376,
                    -5.131985,
                    -5.106911,
                    -5.157397,
                    -5.831578,
                    -5.263907,
                    -5.791384,
                    -5.873203,
                    -5.791591,
                    -6.052242,
                    -5.349160,
                    -4.718514,
                    -6.151028,
                    -5.536199,
                    -5.082184,
                    -4.943134,
                    -5.183392,
                    -5.604786,
                ],
                "lfcSE": [
                    0.000000,
                    0.238611,
                    0.457658,
                    0.509961,
                    0.458276,
                    0.506097,
                    0.456955,
                    0.444127,
                    0.490361,
                    0.476912,
                    0.441245,
                    0.474529,
                    0.441251,
                    0.444274,
                    0.449459,
                    0.432753,
                    0.479159,
                    0.433825,
                    0.420628,
                    0.411914,
                    0.446255,
                    0.464985,
                ],
                "logCPM": [
                    10.95644,
                    11.54124,
                    14.38279,
                    13.91052,
                    14.38299,
                    14.39898,
                    14.61446,
                    14.68405,
                    14.12630,
                    14.43171,
                    14.60056,
                    14.22315,
                    14.60047,
                    14.60068,
                    14.57110,
                    14.96735,
                    14.14636,
                    14.71141,
                    14.96790,
                    15.13650,
                    14.65667,
                    14.36578,
                ],
                "stat": [
                    0.00000,
                    107.96562,
                    72.73416,
                    77.43137,
                    75.98171,
                    61.75185,
                    61.37267,
                    62.13560,
                    72.12935,
                    63.73962,
                    71.54611,
                    72.73208,
                    71.54959,
                    75.29936,
                    65.01752,
                    55.46073,
                    76.69876,
                    67.80019,
                    60.99827,
                    58.88812,
                    62.52793,
                    68.81366,
                ],
                "pvalue": [
                    1.000000e00,
                    2.734813e-25,
                    1.483440e-17,
                    1.374153e-18,
                    2.863044e-18,
                    3.895845e-15,
                    4.723203e-15,
                    3.206031e-15,
                    2.015438e-17,
                    1.420001e-15,
                    2.708546e-17,
                    1.485003e-17,
                    2.703767e-17,
                    4.044912e-18,
                    7.423528e-16,
                    9.534463e-14,
                    1.991284e-18,
                    1.809322e-16,
                    5.712493e-15,
                    1.668972e-14,
                    2.626931e-15,
                    1.082204e-16,
                ],
            },
            index=[f"gene{i}" for i in range(22)],
        )
        pd.testing.assert_frame_equal(table_ref, s, check_frame_type=False, rtol=1e-4)

    def test_topTags(self):
        self.d.estimateGLMCommonDisp()
        s = glmLRT(self.d.glmFit())
        t = topTags(s)
        self.assertListEqual(
            list(t.index),
            [
                "gene1",
                "gene5",
                "gene13",
                "gene9",
                "gene15",
                "gene4",
                "gene6",
                "gene12",
                "gene10",
                "gene20",
            ],
        )
        self.assertListEqual(
            list(t.columns),
            [
                "log2FoldChange",
                "lfcSE",
                "logCPM",
                "stat",
                "pvalue",
                "FDR",
                "adj_pvalue",
            ],
        )
