import json
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from inmoose.limma import MArrayLM, eBayes, fitFDist, fitFDistUnequalDF1, squeezeVar
from inmoose.limma.fitFDistUnequalDF1 import _loess_fit


def read_reference():
    text = (Path(__file__).parent / "data/unequal_df/reference.json").read_text()
    # jsonlite encodes R's missing/nonfinite numbers as strings.
    for old, new in [
        ('"NA"', "NaN"),
        ('"NaN"', "NaN"),
        ('"Inf"', "Infinity"),
        ('"-Inf"', "-Infinity"),
    ]:
        text = text.replace(old, new)
    return json.loads(text)


class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = read_reference()

    def assert_reference(self, actual, expected, boundary=False):
        # Both implementations use bounded Brent with R's default x tolerance.
        # Their sqrt(eps) constants differ slightly. Near the upper bound this
        # is amplified by df2 = 2*p/(1-p); variance estimates remain much closer.
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=5e-7 if boundary else 1e-9,
            atol=1e-12,
            equal_nan=True,
        )

    def test_estimator_R_references(self):
        for name, case in self.reference["cases"].items():
            with self.subTest(case=name):
                actual = fitFDistUnequalDF1(
                    case["x"],
                    case["df"],
                    covariate=case.get("covariate"),
                    prior_weights=case.get("prior_weights"),
                )
                self.assert_reference(actual["scale"], case["fit"]["scale"])
                self.assert_reference(
                    actual["df2"], case["fit"]["df2"], boundary=name == "boundary"
                )

    def test_squeezeVar_R_references(self):
        for name, case in self.reference["cases"].items():
            for mode, kwargs in [
                ("default", {}),
                ("legacy", {"legacy": True}),
                ("modern", {"legacy": False}),
                ("span", {"legacy": True, "span": 0.6}),
            ]:
                if mode not in case:
                    continue
                with self.subTest(case=name, mode=mode):
                    actual = squeezeVar(
                        case["x"],
                        case["df"],
                        covariate=case.get("covariate"),
                        **kwargs,
                    )
                    for key, value in actual.items():
                        self.assert_reference(
                            value,
                            case[mode][key.replace("_", ".")],
                            boundary=name == "boundary" and key == "df_prior",
                        )

    def test_missing_reference_moderation(self):
        # Saved R lmFit/contrasts.fit output isolates moderation from #184.
        case = self.reference["missing"]
        coefficients = pd.DataFrame(
            case["coefficients"], index=case["genes"], columns=case["contrasts"]
        )
        stdev = pd.DataFrame(
            case["stdev_unscaled"], index=case["genes"], columns=case["contrasts"]
        )
        for mode, kwargs in [
            ("default", {}),
            ("default", {"legacy": False}),
            ("legacy", {"legacy": True}),
            ("trend", {"trend": True}),
            ("trend", {"trend": case["Amean"]}),
            ("span", {"trend": True, "legacy": True, "span": 0.6}),
        ]:
            with self.subTest(
                mode=mode, numeric_trend=isinstance(kwargs.get("trend"), list)
            ):
                fit = MArrayLM(
                    coefficients.copy(),
                    stdev.copy(),
                    np.array(case["sigma"]),
                    np.array(case["df"]),
                    np.eye(2),
                )
                fit.Amean = np.array(case["Amean"])
                result = eBayes(fit, **kwargs)
                for key in ("df_prior", "s2_prior", "s2_post", "t", "p_value"):
                    self.assert_reference(getattr(result, key), case[mode][key])
                pd.testing.assert_index_equal(result.t.index, coefficients.index)
                pd.testing.assert_index_equal(result.t.columns, coefficients.columns)
                # tmixture_vector's separate df mutation bug is out of scope.
                # Check the correct total df and t/p directly from squeezeVar.
                out = squeezeVar(
                    np.square(case["sigma"]),
                    case["df"],
                    covariate=case["Amean"] if mode in ("trend", "span") else None,
                    legacy=kwargs.get("legacy"),
                    span=kwargs.get("span"),
                )
                total = np.minimum(
                    np.array(case["df"]) + out["df_prior"], sum(case["df"])
                )
                self.assert_reference(total, case[mode]["df_total"])
                moderated_t = coefficients / stdev / np.sqrt(out["var_post"][:, None])
                self.assert_reference(moderated_t, case[mode]["t"])
                self.assert_reference(
                    2 * t.cdf(-np.abs(moderated_t), total[:, None]),
                    case[mode]["p_value"],
                )

    def test_selection_and_input_preservation(self):
        case = self.reference["cases"]["equal"]
        x = np.array(case["x"])
        for df in (6, np.float64(6), [6], np.full(len(x), 6)):
            actual = squeezeVar(x, df)
            self.assert_reference(actual["df_prior"], case["default"]["df.prior"])
            self.assert_reference(actual["var_prior"], fitFDist(x, 6)["scale"])
            self.assertEqual(actual["df_prior"], fitFDist(x, 6)["df2"])
        for case in self.reference["cases"].values():
            x, df = np.array(case["x"]), np.array(case["df"])
            original_x, original_df = x.copy(), df.copy()
            fitFDistUnequalDF1(x, df)
            squeezeVar(x, df)
            np.testing.assert_array_equal(x, original_x)
            np.testing.assert_array_equal(df, original_df)
        for legacy in (None, True, False):
            out = squeezeVar([1, 9], [2, 6], legacy=legacy)
            np.testing.assert_array_equal(out["var_post"], [1, 9])
            np.testing.assert_array_equal(out["var_prior"], [1, 9])
            self.assertEqual(out["df_prior"], 0)
        # R's automatic min/max comparison selects legacy when df contains NA.
        out = squeezeVar([0.1, 0.2, 0.4, 0.8, 1.5, 3.2], [1, 2, np.nan, 4, 5, 6])
        self.assertEqual(out["df_prior"], np.inf)
        np.testing.assert_allclose(out["var_post"], 1.16)

    def test_lowess_R_references(self):
        cases = [self.reference["smoother"], *self.reference["small_trends"].values()]
        for case in cases:
            self.assert_reference(
                _loess_fit(
                    np.array(case["y"]), np.array(case["x"]),
                    np.array(case["weights"]), 0.6,
                ),
                case["fitted"],
            )

    def test_uninformative_and_nonfinite_R_references(self):
        for name, case in self.reference["edge_cases"].items():
            with self.subTest(case=name), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                actual = fitFDistUnequalDF1(case["x"], case["df"])
                self.assert_reference(actual["scale"], case["fit"]["scale"])
                self.assert_reference(actual["df2"], case["fit"]["df2"], boundary=True)

    def test_invalid_arguments_and_robust_unsupported(self):
        with self.assertRaisesRegex(ValueError, "different lengths"):
            fitFDistUnequalDF1([1, 2, 3], [2, 3])
        with self.assertRaisesRegex(ValueError, "NA df1"):
            fitFDistUnequalDF1([1, 2, 3], [2, np.nan, 3])
        with self.assertRaisesRegex(ValueError, "NA covariate"):
            fitFDistUnequalDF1([1, 2, 3], 4, covariate=[1, np.nan, 2])
        with self.assertRaisesRegex(ValueError, "negative"):
            fitFDistUnequalDF1([1, 2, 3], 4, prior_weights=[1, -1, 1])
        with self.assertRaisesRegex(NotImplementedError, "Robust"):
            fitFDistUnequalDF1([1, 2, 3], [2, 3, 4], robust=True)
        for legacy in (None, True, False):
            with self.assertRaisesRegex(NotImplementedError, "Robust"):
                squeezeVar([1, 2, 3], [2, 3, 4], robust=True, legacy=legacy)
