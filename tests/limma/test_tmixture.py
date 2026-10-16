import unittest

import numpy as np
import pandas as pd
from scipy.stats import t

from inmoose.limma import MArrayLM, eBayes, tmixture_matrix, tmixture_vector


class Test(unittest.TestCase):
    def test_tmixture_matrix(self):
        self.assertTrue(
            np.allclose(
                tmixture_matrix(
                    np.arange(1, 7).reshape(2, 3),
                    np.arange(2, 8).reshape(2, 3),
                    np.arange(3, 5),
                    0.5,
                ),
                np.array([648.387, 1544.773, 3105.003]),
            )
        )

    def test_tmixture_vector(self):
        self.assertAlmostEqual(
            tmixture_vector(
                np.arange(1, 7),
                np.arange(2, 8),
                np.arange(3, 9),
                0.5,
            ),
            1124.83,
            places=2,
        )

    def test_tmixture_vector_preserves_inputs(self):
        # R 4.3.3 / limma 3.65.1:
        # limma:::tmixture.vector(c(4,-1), c(.5,1), df, .5, c(0,100))
        cases = [
            (np.array([5.67672502783233, 7.67672502783233]), False, 5.820694676031157),
            (np.array([7.0, 7.0]), False, 7.464298960213775),
            (np.array([5, 7]), False, 5.43374471290178),
            (np.array([5.67672502783233, 7.67672502783233]), True, 7.463045572722503),
        ]
        for df, missing, expected in cases:
            with self.subTest(df=df, missing=missing):
                tstat = np.array([4.0, np.nan if missing else -1.0])
                stdev = np.array([0.5, 1.0])
                limits = np.array([0.0, 100.0])
                inputs = (tstat, stdev, df, limits)
                before = [value.copy() for value in inputs]
                actual = tmixture_vector(tstat, stdev, df, 0.5, limits)
                for value, original in zip(inputs, before):
                    np.testing.assert_array_equal(value, original)
                np.testing.assert_allclose(actual, expected, rtol=1e-10)

    def test_tmixture_vector_readonly_views(self):
        # Strided views share their backing array. No numerical input needs
        # to be writable, even when the tail-probability mapping changes t.
        data = np.array(
            [[4.0, 0.5, 5.67672502783233], [-1.0, 1.0, 7.67672502783233]]
        )
        original = data.copy()
        data.setflags(write=False)
        actual = tmixture_vector(data[:, 0], data[:, 1], data[:, 2], 0.5)
        np.testing.assert_array_equal(data, original)
        np.testing.assert_allclose(actual, 5.820694676031157, rtol=1e-10)

    def test_tmixture_matrix_unequal_df(self):
        # limma:::tmixture.matrix(matrix(c(4,1,-5,2),2),
        #                        matrix(.5,2,2), df, .5)
        tstat = np.array([[4.0, -5.0], [1.0, 2.0]])
        stdev = np.full((2, 2), 0.5)
        df = np.array([5.67672502783233, 7.67672502783233])
        inputs = (tstat, stdev, df)
        before = [value.copy() for value in inputs]
        actual = tmixture_matrix(tstat, stdev, df, 0.5)
        for value, original in zip(inputs, before):
            np.testing.assert_array_equal(value, original)
        expected = [5.820694676031157, 8.856753630590356]
        np.testing.assert_allclose(actual, expected, rtol=1e-10)
        reversed_columns = tmixture_matrix(tstat[:, ::-1], stdev[:, ::-1], df, 0.5)
        np.testing.assert_allclose(reversed_columns, expected[::-1], rtol=1e-10)

    def test_ebayes_total_df_and_lods(self):
        # R 4.3.3 / limma 3.65.1: eBayes(fit, proportion=.5, legacy=TRUE).
        # A saved-fit example isolates this bug from lmFit missingness and
        # modern unequal-df hyperparameter estimation.
        index = ["g3", "g1", "g4", "g2"]
        columns = ["B-A", "C-A"]
        coefficients = pd.DataFrame(
            [[3.0, -2.0], [0.5, 1.0], [-1.0, 1.5], [2.0, -3.0]],
            index=index,
            columns=columns,
        )
        stdev = pd.DataFrame(
            [[0.5, 0.6], [0.75, 0.5], [1.0, 0.8], [0.6, 1.0]],
            index=index,
            columns=columns,
        )
        sigma = np.array([0.3, 1.0, 2.0, 4.0])
        df = np.array([4.0, 6.0, 6.0, 6.0])
        before = [value.copy() for value in (coefficients, stdev, sigma, df)]
        fit = MArrayLM(coefficients, stdev, sigma, df, np.eye(2))
        result = eBayes(fit, proportion=0.5)

        pd.testing.assert_frame_equal(coefficients, before[0])
        pd.testing.assert_frame_equal(stdev, before[1])
        np.testing.assert_array_equal(sigma, before[2])
        np.testing.assert_array_equal(df, before[3])
        np.testing.assert_allclose(
            result.df_total,
            [5.09304083726828, 7.09304083726828, 7.09304083726828, 7.09304083726828],
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            result.var_prior, [13.0063805328961, 7.34270378720564], rtol=1e-9
        )
        np.testing.assert_allclose(
            result.lods,
            [
                [7.4077715246757, 4.76323041745209],
                [-1.34122620001099, 0.118098493886717],
                [-1.17058374278518, -0.76841304252782],
                [-1.37939058518902, -0.745346996056226],
            ],
            rtol=1e-9,
        )
        np.testing.assert_allclose(
            result.p_value,
            [
                [3.59837862307161e-05, 0.000630070445753556],
                [0.513220335924056, 0.0773226533777667],
                [0.608120052590878, 0.34764920082012],
                [0.396164755269436, 0.442818039630208],
            ],
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            result.p_value,
            2 * t.sf(np.abs(result.t), result.df_total[:, None]),
            rtol=1e-12,
        )
        pd.testing.assert_index_equal(result.lods.index, coefficients.index)
        pd.testing.assert_index_equal(result.lods.columns, coefficients.columns)
