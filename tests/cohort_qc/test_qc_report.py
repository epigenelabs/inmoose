import unittest
from io import BytesIO
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd

from inmoose.cohort_qc import CohortQC, QCReport


class TestQCReport(unittest.TestCase):
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

        qc = CohortQC(
            clinical_df=self.clinical_df,
            batch_column="batch",
            data_expression_df=self.data_expression_df,
            data_expression_df_before=self.data_expression_df_before,
            covariates=self.covariates,
            n_components=2,
            n_neighbors=2,
        )

        # Initialize the QCReport object with the mock CohortQC
        self.qc_report = QCReport(qc)

    @patch("inmoose.cohort_qc.qc_report.plt.Figure.savefig")
    def test_plot_html(self, mock_savefig):
        # Simulate saving the plot to a buffer
        buffer = BytesIO()
        mock_savefig.side_effect = lambda *args, **kwargs: buffer.write(
            b"fake_image_data"
        )

        # Call the method
        fig = plt.figure()
        html_img = self.qc_report.plot_html(fig, file_name="test_plot")

        # Check if the generated HTML contains the correct base64 image data
        self.assertIn('<img src="data:image/png;base64,', html_img)
        self.assertIn('alt="test_plot"', html_img)

    def test_generate_html_report(self):
        """Test the generation of the full HTML report."""
        html_report = self.qc_report.generate_html_report()
        self.assertIn("<h1>Cohort Quality Control Report</h1>", html_report)
        self.assertIn("<h2>PCA Analysis and Variance Explained", html_report)

    def test_pca_analysis_html_report(self):
        """Test PCA analysis HTML report generation."""
        pca_report = self.qc_report.pca_analysis_html_report()
        self.assertIn("<h2>PCA Analysis and Variance Explained", pca_report)

    def test_cohort_summary_html_report(self):
        """Test cohort summary HTML report generation."""
        cohort_summary_report = self.qc_report.cohort_summary_html_report()
        self.assertIn("<h2>Cohort Summary", cohort_summary_report)

    def test_sample_distribution_html_report(self):
        """Test sample distribution HTML report generation."""
        sample_dist_report = self.qc_report.sample_distribution_html_report()
        self.assertIn("<h2>Correction Effect Metric", sample_dist_report)

    def test_silhouette_html_report(self):
        """Test silhouette score HTML report generation."""
        silhouette_report = self.qc_report.silhouette_html_report()
        self.assertIn("<h2>Silhouette Score", silhouette_report)

    def test_entropy_html_report(self):
        """Test entropy batch mixing HTML report generation."""
        entropy_report = self.qc_report.entropy_html_report()
        self.assertIn("<h2>Entropy of Batch Mixing (EBM)", entropy_report)

    def test_mixed_dataset_html_report(self):
        """Test mixed dataset summary HTML report generation."""
        mixed_dataset_report = self.qc_report.mixed_dataset_html_report()
        self.assertIn("<h1>Mixed Dataset Summary Report", mixed_dataset_report)

    def test_save_html_report_local(self):
        """Test saving the HTML report to a local file."""
        with patch("builtins.open", unittest.mock.mock_open()) as mock_file:
            self.qc_report.save_html_report_local(output_path=".")
            mock_file.assert_called_once_with("./cohort_qc_report.html", "w")
            mock_file().write.assert_called_once()
