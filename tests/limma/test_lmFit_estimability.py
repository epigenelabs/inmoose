import unittest

import numpy as np
import pandas as pd
from patsy import DesignInfo, DesignMatrix, dmatrix

from inmoose.limma import contrasts_fit, eBayes, lmFit, makeContrasts


class Test(unittest.TestCase):
    def setUp(self):
        self.conditions = np.repeat(["A", "B", "C"], 4)
        self.batches = np.tile(["b1", "b1", "b2", "b2"], 3)
        self.design = dmatrix(
            "0 + condition + batch",
            {"condition": self.conditions, "batch": self.batches},
        )
        self.names = self.design.design_info.column_names
        self.contrasts = makeContrasts(
            ["condition[B]-condition[A]", "condition[C]-condition[A]"],
            levels=self.design,
        )
        self.contrasts.columns = ["B-A", "C-A"]
        rows = []
        for i, scale in enumerate([0.25, 0.5, 1, 2, 4, 1]):
            y = 10 + i + np.repeat([0, 2, -1], 4).astype(float)
            y += (self.batches == "b2") * 0.7
            y += np.tile([-1, 1], 6) * scale
            rows.append(y)
        self.y = pd.DataFrame(
            rows,
            index=["g4", "g1", "g3", "g0", "g2", "only_batch2"],
            columns=[f"sample{i}" for i in range(12)],
        )
        self.y.loc["only_batch2", self.batches == "b1"] = np.nan

    def test_batch_missing_level(self):
        # R 4.3.3 / limma 3.65.1, using these identical inputs:
        # design <- model.matrix(~0+condition+batch)
        # fit <- lmFit(y, design)
        # out <- contrasts.fit(fit, makeContrasts(conditionB-conditionA,
        #                       conditionC-conditionA, levels=design))
        with self.assertWarnsRegex(UserWarning, "Partial NA coefficients"):
            fit = lmFit(self.y, self.design)
        np.testing.assert_allclose(
            fit.coefficients.loc["only_batch2"], [15.7, 17.7, 14.7, np.nan]
        )
        np.testing.assert_allclose(
            fit.stdev_unscaled.loc["only_batch2"],
            [0.707106781186547, 0.707106781186547, 0.707106781186547, np.nan],
        )
        np.testing.assert_array_equal(fit.df_residual, [8, 8, 8, 8, 8, 3])
        np.testing.assert_allclose(
            fit.sigma,
            [
                0.306186217847897,
                0.612372435695795,
                1.224744871391589,
                2.449489742783178,
                4.898979485566357,
                1.414213562373095,
            ],
        )
        original = fit.coefficients.copy()
        result = contrasts_fit(fit, self.contrasts)
        pd.testing.assert_frame_equal(fit.coefficients, original)
        np.testing.assert_allclose(result.coefficients, np.tile([2, -1], (6, 1)))
        np.testing.assert_allclose(
            result.stdev_unscaled,
            np.repeat([[0.707106781186548]] * 5 + [[0.866025403784439]], 2, axis=1),
        )
        # Preserve R contrasts.fit's documented approximate SE for missing
        # values in a nonorthogonal design (direct refitting gives a different SE).
        np.testing.assert_allclose(
            result.stdev_unscaled.iloc[-1] * result.sigma[-1],
            [1.224744871391589, 1.224744871391589],
        )
        pd.testing.assert_index_equal(result.coefficients.index, self.y.index)
        pd.testing.assert_index_equal(result.coefficients.columns, self.contrasts.columns)
        reordered = contrasts_fit(fit, self.contrasts.iloc[::-1, ::-1])
        pd.testing.assert_frame_equal(
            reordered.coefficients, result.coefficients.iloc[:, ::-1]
        )
        pd.testing.assert_frame_equal(
            reordered.stdev_unscaled, result.stdev_unscaled.iloc[:, ::-1]
        )
        result = eBayes(result)
        self.assertTrue(np.isfinite(result.df_prior))
        self.assertTrue(np.isfinite(result.t).all().all())
        self.assertTrue(np.isfinite(result.p_value).all().all())

        # The contrast must not depend on which aliased column is dropped.
        order = [3, 0, 1, 2]
        design = DesignMatrix(
            self.design[:, order], DesignInfo([self.names[j] for j in order])
        )
        with self.assertWarnsRegex(UserWarning, "Partial NA coefficients"):
            fit = lmFit(self.y, design)
        result = contrasts_fit(fit, self.contrasts)
        np.testing.assert_allclose(result.coefficients, np.tile([2, -1], (6, 1)))

    def test_missing_condition(self):
        y = self.y.copy()
        y.loc["missing_B"] = self.y.loc["g3"]
        y.loc["missing_B", self.conditions == "B"] = np.nan
        design = dmatrix("0 + condition", {"condition": self.conditions})
        with self.assertWarnsRegex(UserWarning, "Partial NA coefficients"):
            fit = lmFit(y, design)
        result = contrasts_fit(fit, self.contrasts.loc[design.design_info.column_names])
        np.testing.assert_allclose(
            result.coefficients.loc["missing_B"], [np.nan, -1], atol=1e-12
        )
        self.assertTrue(np.isnan(result.stdev_unscaled.loc["missing_B", "B-A"]))
        self.assertTrue(np.isfinite(result.stdev_unscaled.loc["missing_B", "C-A"]))
        result = eBayes(result)
        self.assertTrue(np.isnan(result.p_value.loc["missing_B", "B-A"]))
        self.assertTrue(np.isfinite(result.p_value.loc["missing_B", "C-A"]))

    def test_alias_cancellation_and_nonestimable_contrasts(self):
        with self.assertWarnsRegex(UserWarning, "Partial NA coefficients"):
            fit = lmFit(self.y, self.design)
        contrasts = pd.DataFrame(
            {
                "A_at_batch2": [1, 0, 0, 1],
                "batch_effect": [0, 0, 0, 1],
                "A_at_batch1": [1, 0, 0, 0],
                "B-A": [-1, 1, 0, 0],
            },
            index=self.names,
        )
        result = contrasts_fit(fit, contrasts)
        # The observed A mean is estimable although A and the batch effect
        # separately are not. R lmFit on the observed group-means design gives
        # mean=15.7 and stdev.unscaled=1/sqrt(2).
        np.testing.assert_allclose(
            result.coefficients.loc["only_batch2"], [15.7, np.nan, np.nan, 2]
        )
        np.testing.assert_allclose(
            result.stdev_unscaled.loc["only_batch2"],
            [np.sqrt(0.5), np.nan, np.nan, np.sqrt(0.75)],
        )

        # Retaining a coefficient in a reduced fit is also insufficient to
        # prove estimability: here B and batch are confounded for this feature.
        y = self.y.copy()
        y.loc["confounded"] = self.y.loc["g3"]
        obs = ((self.conditions != "B") & (self.batches == "b1")) | (
            (self.conditions == "B") & (self.batches == "b2")
        )
        y.loc["confounded", ~obs] = np.nan
        with self.assertWarnsRegex(UserWarning, "Partial NA coefficients"):
            fit = lmFit(y, self.design)
        result = contrasts_fit(fit, self.contrasts)
        np.testing.assert_allclose(
            result.coefficients.loc["confounded"], [np.nan, -1], atol=1e-12
        )
        self.assertTrue(np.isnan(result.stdev_unscaled.loc["confounded", "B-A"]))

    def test_estimability_survives_subsetting_and_contrasts(self):
        with self.assertWarnsRegex(UserWarning, "Partial NA coefficients"):
            fit = lmFit(self.y, self.design)
        # Column selection must retain the nuisance alias directions even when
        # the nuisance coefficient itself is no longer in the fitted object.
        selected = contrasts_fit(fit, coefficients=self.names[:3])
        result = contrasts_fit(selected, self.contrasts.loc[self.names[:3]])
        direct = contrasts_fit(fit, self.contrasts)
        pd.testing.assert_frame_equal(result.coefficients, direct.coefficients)
        pd.testing.assert_frame_equal(result.stdev_unscaled, direct.stdev_unscaled)

        rows = self.y.index.isin(["g3", "only_batch2"])
        selected = fit[rows, self.names[:3]]
        result = contrasts_fit(selected, self.contrasts.loc[self.names[:3]])
        pd.testing.assert_frame_equal(result.coefficients, direct.coefficients.loc[rows])
        pd.testing.assert_frame_equal(
            result.stdev_unscaled, direct.stdev_unscaled.loc[rows]
        )

        # An intermediate non-estimable coefficient must not destroy the
        # information needed by a subsequent estimable linear combination.
        identity = pd.DataFrame(np.eye(4), index=self.names, columns=self.names)
        intermediate = contrasts_fit(fit, identity)
        self.assertTrue(intermediate.coefficients.loc["only_batch2"].isna().all())
        result = contrasts_fit(intermediate, self.contrasts)
        pd.testing.assert_frame_equal(result.coefficients, direct.coefficients)
        pd.testing.assert_frame_equal(result.stdev_unscaled, direct.stdev_unscaled)
        second = pd.DataFrame({"B-C": [1, -1]}, index=self.contrasts.columns)
        np.testing.assert_allclose(contrasts_fit(direct, second).coefficients, 3)

    def test_complete_data_control(self):
        complete = self.y.iloc[:5]
        fit = lmFit(complete, self.design)
        self.assertEqual(fit._rank_deficient_fits, {})
        result = contrasts_fit(fit, self.contrasts)
        with self.assertWarnsRegex(UserWarning, "Partial NA coefficients"):
            mixed = lmFit(self.y, self.design)
        mixed = contrasts_fit(mixed, self.contrasts)
        pd.testing.assert_frame_equal(result.coefficients, mixed.coefficients.iloc[:5])
        pd.testing.assert_frame_equal(
            result.stdev_unscaled, mixed.stdev_unscaled.iloc[:5]
        )
        np.testing.assert_allclose(fit.sigma, mixed.sigma[:5], rtol=1e-12)
        np.testing.assert_array_equal(fit.df_residual, mixed.df_residual[:5])
