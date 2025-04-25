import unittest
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pandas as pd
from sklearn.decomposition import PCA

from inmoose.cohort_qc.cohort_metric import CohortMetric


class TestCohortMetric(unittest.TestCase):
    def setUp(self):
        """Set up mock data for testing."""
        self.clinical_df = pd.DataFrame(
            {
                "SampleID": ["Sample1", "Sample2", "Sample3", "Sample4"],
                "Covariate1": [1, 1, 2, 2],
                "Covariate2": ["X", "Y", "X", "Y"],
                "Covariate3": ["D", "D", "D", "D"],
                "batch": [
                    "Dataset1",
                    "Dataset1",
                    "Dataset2",
                    "Dataset2",
                ],
            }
        ).set_index("SampleID")

        self.data_expression_df = pd.DataFrame(
            {
                "Sample1": [1.1, 1.2, 1.3],
                "Sample2": [2.1, 2.2, 2.3],
                "Sample3": [3.1, 3.2, 3.3],
                "Sample4": [4.1, 4.2, 4.3],
            },
            index=["Gene1", "Gene2", "Gene3"],
        )

        self.data_expression_df_before = self.data_expression_df.copy()

        self.covariates = ["Covariate1", "Covariate2"]

        self.qc = CohortMetric(
            clinical_df=self.clinical_df,
            batch_column="batch",
            data_expression_df=self.data_expression_df,
            data_expression_df_before=self.data_expression_df_before,
            covariates=self.covariates,
            n_components=2,
            n_neighbors=2,
        )

    def test_cohort_qc_only_one_batch(self):
        self.clinical_df["batch"] = "unique_batch"
        with self.assertRaises(ValueError):
            CohortMetric(
                clinical_df=self.clinical_df,
                batch_column="batch",
                data_expression_df=self.data_expression_df,
                data_expression_df_before=self.data_expression_df_before,
                covariates=self.covariates,
                n_components=2,
                n_neighbors=2,
            )

    def test_covariate_corner_cases(self):
        """
        make sure the code runs without error when passing zero or one covariate
        """
        CohortMetric(
            clinical_df=self.clinical_df,
            batch_column="batch",
            data_expression_df=self.data_expression_df,
            data_expression_df_before=self.data_expression_df_before,
            n_components=2,
            n_neighbors=2,
        )
        CohortMetric(
            clinical_df=self.clinical_df,
            batch_column="batch",
            data_expression_df=self.data_expression_df,
            data_expression_df_before=self.data_expression_df_before,
            covariates=[],
            n_components=2,
            n_neighbors=2,
        )
        CohortMetric(
            clinical_df=self.clinical_df,
            batch_column="batch",
            data_expression_df=self.data_expression_df,
            data_expression_df_before=self.data_expression_df_before,
            covariates=["Covariate1"],
            n_components=2,
            n_neighbors=2,
        )

    def test_CohortMetric_missing_covariates(self):
        missing_covariates = ["covariate1", "nonexistent_covariate"]

        with self.assertRaises(ValueError) as context:
            CohortMetric(
                clinical_df=self.clinical_df,
                batch_column="batch",
                data_expression_df=self.data_expression_df,
                covariates=missing_covariates,
                n_components=2,
                n_neighbors=2,
            )
        self.assertEqual(
            str(context.exception),
            "Covariates covariate1, nonexistent_covariate are not present in the clinical dataframe.",
        )

    def test_CohortMetric_without_before_data(self):
        """Test CohortMetric class."""
        qc_without_before_data = CohortMetric(
            clinical_df=self.clinical_df,
            batch_column="batch",
            data_expression_df=self.data_expression_df,
            covariates=["Covariate1", "Covariate2", "Covariate3"],
            n_components=2,
            n_neighbors=2,
        )
        assert qc_without_before_data.data_expression_df_before is None
        assert qc_without_before_data.covariates == ["Covariate1", "Covariate2"]

    def test_identify_mixed_datasets(self):
        """Test identify_mixed_datasets method."""
        mixed_datasets = self.qc.identify_mixed_datasets()
        self.assertEqual(mixed_datasets, ["Dataset1", "Dataset2"])

    @mock.patch("seaborn.scatterplot")
    def test_pca_plot(self, mock_scatterplot):
        # Simulate PCA output (2D points)
        pcs = np.random.rand(4, 2)
        labels = self.clinical_df["batch"]

        # Call the private method _pca_plot
        result_fig = self.qc.pca_plot(pcs=pcs, labels=labels, title="Test PCA")

        # Check that the method returns a matplotlib figure
        self.assertIsInstance(result_fig, plt.Figure)

        # Check that seaborn scatterplot was called
        self.assertTrue(mock_scatterplot.called)

        # Extract the actual arguments passed to the scatterplot
        _, kwargs = mock_scatterplot.call_args

        # Verify the numpy arrays pcs[:, 0] and pcs[:, 1] match
        npt.assert_array_equal(kwargs["x"], pcs[:, 0])
        npt.assert_array_equal(kwargs["y"], pcs[:, 1])

    def test_plot_pca_variance(self):
        # Example explained variance data for 5 principal components
        explained_variance = np.array([0.4, 0.3, 0.2, 0.05, 0.05])

        # Call the private method _plot_pca_variance
        result_fig = self.qc.plot_pca_variance(
            explained_variance=explained_variance, ylim=0.5
        )

        # Check that the method returns a matplotlib figure
        self.assertIsInstance(result_fig, plt.Figure)

    def test_create_correlation_matrix_with_pc(self):
        clinical_df_with_pc = self.clinical_df.copy()
        clinical_df_with_pc["Data Element With Spaces"] = [
            "Some Value",
            "other-value",
            "otherValue",
            "other-value",
        ]
        clinical_df_with_pc["PC1"] = [1, 2, 3, 1]
        clinical_df_with_pc["PC2"] = [2, 2, 3, 5]
        result_matrix = self.qc.create_correlation_matrix_with_pc(clinical_df_with_pc)

        # Pearson correlation should be done with numeric_var
        self.assertEqual(
            result_matrix.at["Covariate1", "PC1"],
            ("3.02e-01", "6.98e-01", "Pearson correlation", 4),
        )
        self.assertEqual(
            result_matrix.at["Covariate1", "PC2"],
            ("8.16e-01", "1.84e-01", "Pearson correlation", 4),
        )

        # T-test should be done for binary categorical_var
        self.assertEqual(
            result_matrix.at["Covariate2", "PC1"], ("4.47e-01", "6.98e-01", "T-test", 4)
        )
        self.assertEqual(
            result_matrix.at["Covariate2", "PC2"],
            ("-6.32e-01", "5.92e-01", "T-test", 4),
        )

    def test_pca_analysis(self):
        """Test pca_analysis method."""
        result = self.qc.pca_analysis()
        # Ensure the result is a tuple of length 6
        self.assertEqual(len(result), 6)
        self.assertIsInstance(result[0], pd.DataFrame)
        self.assertIsInstance(result[1], pd.DataFrame)
        self.assertIsInstance(result[2], PCA)
        self.assertIsInstance(result[3], PCA)
        self.assertIsInstance(result[4], np.ndarray)
        self.assertIsInstance(result[5], np.ndarray)

    def test_cohort_summary(self):
        """Test cohort_summary method."""
        summary = self.qc.cohort_summary()
        self.assertEqual(summary["Number of Samples"], 4)
        self.assertEqual(summary["Number of Datasets"], 2)
        self.assertEqual(summary["Number of Genes"], 3)

    def test_quantify_correction_effect(self):
        """Test quantify_correction_effect method."""
        mad_before, mad_after, effect_metric = self.qc.quantify_correction_effect()
        self.assertIsInstance(mad_before, float)
        self.assertIsInstance(mad_after, float)
        self.assertIsInstance(effect_metric, float)

    @mock.patch("inmoose.cohort_qc.cohort_metric.silhouette_score")
    def test_silhouette_score(self, mock_silhouette_score):
        """Test silhouette_score method."""
        mock_silhouette_score.side_effect = [0.5, 0.3]  # Mock silhouette scores
        score_before, score_after = self.qc.silhouette_score()
        self.assertEqual(score_before, 0.3)
        self.assertEqual(score_after, 0.5)

    @mock.patch("sklearn.neighbors.NearestNeighbors.kneighbors")
    def test_compute_entropy(self, mock_kneighbors):
        """Test compute_entropy method."""
        mock_kneighbors.return_value = (None, np.array([[0, 1], [1, 0]]))
        entropy = self.qc.compute_entropy(self.qc.data_expression_df)
        self.assertIsInstance(entropy, float)

    def test_entropy_batch_mixing(self):
        """Test entropy_batch_mixing method."""
        entropy_before, entropy_after = self.qc.entropy_batch_mixing()
        self.assertIsInstance(entropy_before, float)
        self.assertIsInstance(entropy_after, float)

    @mock.patch("inmoose.cohort_qc.cohort_metric.sns.violinplot")
    def test_compare_sample_distribution_by_covariates(self, mock_violinplot):
        fig = self.qc.compare_sample_distribution_by_covariates(("1", "X"))
        self.assertIsNotNone(fig)

    @mock.patch("inmoose.cohort_qc.cohort_metric.sns.violinplot")
    def test_compare_sample_distribution_total(self, mock_violinplot):
        # Test compare_sample_distribution_total
        fig = self.qc.compare_sample_distribution_total()
        self.assertIsNotNone(fig)

    def test_process(self):
        """Test process method."""
        # Mock the return values of each method that is called within `process`
        with (
            mock.patch.object(self.qc, "pca_analysis") as mock_pca_analysis,
            mock.patch.object(
                self.qc, "quantify_correction_effect"
            ) as mock_quantify_correction_effect,
            mock.patch.object(self.qc, "silhouette_score") as mock_silhouette_score,
            mock.patch.object(
                self.qc, "entropy_batch_mixing"
            ) as mock_entropy_batch_mixing,
            mock.patch.object(
                self.qc, "summarize_and_compare_mixed_datasets"
            ) as mock_summarize_and_compare_mixed_datasets,
        ):
            # Define mock return values for each method
            mock_pca_analysis.return_value = (
                pd.DataFrame(),  # association_matrix_before
                pd.DataFrame(),  # association_matrix
                PCA(),  # pca_before
                PCA(),  # pca_after
                np.array([]),  # pcs_before
                np.array([]),  # pcs_after
            )
            mock_quantify_correction_effect.return_value = (0.6, 0.4, 0.2)
            mock_silhouette_score.return_value = (0.5, 0.3)
            mock_entropy_batch_mixing.return_value = (0.7, 0.6)
            mock_summarize_and_compare_mixed_datasets.return_value = {"some": "summary"}

            # Call the process method
            self.qc.process()

            # Assert that each method was called once
            mock_pca_analysis.assert_called_once()
            mock_quantify_correction_effect.assert_called_once()
            mock_silhouette_score.assert_called_once()
            mock_entropy_batch_mixing.assert_called_once()
            mock_summarize_and_compare_mixed_datasets.assert_called_once()
