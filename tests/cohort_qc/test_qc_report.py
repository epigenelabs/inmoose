import os
import unittest

import pandas as pd

from inmoose.cohort_qc.cohort_metric import CohortMetric
from inmoose.cohort_qc.qc_report import QCReport


class TestQCReport(unittest.TestCase):
    def setUp(self):
        import os

        self.output_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "html_report_output",
            "test_qc_report_output.html",
        )
        output_dir = os.path.dirname(self.output_path)
        os.makedirs(output_dir, exist_ok=True)

        self.clinical_df = pd.DataFrame(
            {
                "SampleID": ["S1", "S2", "S3", "S4"],
                "batch": ["B1", "B1", "B2", "B2"],
                "Cov1": [1, 1, 2, 2],
                "Cov2": ["X", "Y", "X", "Y"],
            }
        ).set_index("SampleID")

        self.expression_df = pd.DataFrame(
            {
                "S1": [1.1, 1.2, 1.3],
                "S2": [2.1, 2.2, 2.3],
                "S3": [3.1, 3.2, 3.3],
                "S4": [4.1, 4.2, 4.3],
            },
            index=["G1", "G2", "G3"],
        )

        self.qc = CohortMetric(
            clinical_df=self.clinical_df,
            batch_column="batch",
            data_expression_df=self.expression_df,
            data_expression_df_before=self.expression_df.copy(),
            covariates=["Cov1", "Cov2"],
            n_components=2,
            n_neighbors=2,
        )
        self.qc.process()

        self.report = QCReport(self.qc)
        self.report.save_report(output_path=self.output_path)

    def test_html_report_created(self):
        """Vérifie que le fichier HTML est bien créé."""
        self.assertTrue(os.path.isfile(self.output_path))

    def test_html_report_not_empty(self):
        """Vérifie que le fichier HTML généré n'est pas vide."""
        with open(self.output_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertGreater(len(content.strip()), 100)

    def test_html_report_contains_expected_sections(self):
        """Vérifie que le HTML contient les titres de section attendus."""
        with open(self.output_path, "r", encoding="utf-8") as f:
            html = f.read()

        expected_sections = [
            "Cohort Summary",
            "PCA Analysis and Variance Explained",
            "Correction Effect Metric",
            "Silhouette Score",
            "Entropy of Batch Mixing",
            "Mixed Dataset Summary Report",
        ]

        for section in expected_sections:
            with self.subTest(section=section):
                self.assertIn(section, html)

    def test_html_contains_images(self):
        """Vérifie qu’il y a bien des images encodées base64 dans le HTML."""
        with open(self.output_path, "r", encoding="utf-8") as f:
            html = f.read()

        self.assertIn("data:image/png;base64,", html)


if __name__ == "__main__":
    unittest.main()
