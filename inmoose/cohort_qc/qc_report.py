# -----------------------------------------------------------------------------
# Copyright (C) 2022-2023 M. Colange

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------

import base64
from io import BytesIO

import matplotlib.pyplot as plt
from yattag import Doc, indent

from .cohort_metric import CohortMetric


class QCReport:
    """
    A class for generating an HTML report from a CohortMetric object.

    Attributes
    ----------
    cohort_qc : CohortMetric
        A CohortMetric object.

    Methods
    -------
    pca_analysis_html_report()
        Generates an HTML report summarizing the PCA analysis.
    cohort_summary_html_report()
        Generates an HTML report summarizing the cohort.
    sample_distribution_html_report()
        Generates an HTML report on sample distribution by covariates.
    silhouette_html_report()
        Generates an HTML report summarizing the Silhouette Scores.
    entropy_html_report()
        Generates an HTML report summarizing entropy of batch mixing.
    mixed_dataset_html_report()
        Generates an HTML report on mixed datasets.
    generate_html_report()
        Combines all report sections into a full HTML report.
    save_html_report_local(output_path='.')
        Saves the full HTML report to a local file.
    """

    def __init__(self, cohort_qc: CohortMetric) -> None:
        """
        Parameters
        ----------
        cohort_qc : CohortMetric
            A CohortMetric object.
        """
        self.cohort_qc = cohort_qc
        self.html_report = self.generate_html_report()

    @staticmethod
    def plot_html(plot: plt.Figure, file_name: str = "") -> str:
        """
        Convert a matplotlib plot to a base64-encoded string suitable for embedding in HTML.

        Parameters
        ----------
        plot : matplotlib.figure.Figure
            A matplotlib figure object.
        file_name : str
            Name of the file to be displayed in the alt text of the image. Default is an empty string.

        Returns
        -------
        str
            A base64-encoded string representing the plot.
        """
        # Convert the plot to a PNG image and then to a base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        encoded_image = base64.b64encode(image_png).decode("utf-8")
        plt.close()

        return f'<img src="data:image/png;base64,{encoded_image}" alt="{file_name}">'

    def pca_analysis_html_report(self) -> str:
        """
        Generate an HTML report summarizing the PCA analysis and variance explained before and after batch correction.

        Returns
        -------
        str
            HTML report as a string.
        """
        # Generate an HTML report using yattag
        doc, tag, text = Doc().tagtext()

        with tag("h2"):
            text("PCA Analysis and Variance Explained\n")
        with tag("p"):
            text(
                "Principal Component Analysis (PCA) helps to identify the main sources of variation in the data.\n"
            )
            text(
                "We perform PCA before and after batch effect correction to see if the batch effects dominate the first few principal components (PCs).\n"
            )
            text(
                "The variance explained by each PC indicates how much of the data's variation is captured by that PC. After correction, the variance should ideally be more evenly distributed across PCs.\n"
            )

        with tag("h3"):
            text("PCA Plots (PC1 vs PC2) Before and After Batch Correction\n")
        plot_data_elements_list = [self.cohort_qc.batch_column]
        plot_data_elements_list.extend(self.cohort_qc.covariates)
        for covariate in plot_data_elements_list:
            if self.cohort_qc.data_expression_df_before is not None:
                with tag("h4"):
                    text(f"PCA Plot Before Correction - Colored by {covariate}\n")
                doc.asis(
                    self.plot_html(
                        plot=self.cohort_qc._pca_plot(
                            self.cohort_qc.pcs_before,
                            self.cohort_qc.clinical_df[covariate],
                            title=f"PCA before correction - {covariate}",
                        ),
                        file_name=f"PCA_before_correction_{covariate}",
                    )
                )
            else:
                with tag("h4"):
                    text("Data before correction not available\n")
            with tag("h4"):
                text(f"PCA Plot After Correction - Colored by {covariate}\n")
            doc.asis(
                self.plot_html(
                    plot=self.cohort_qc._pca_plot(
                        self.cohort_qc.pcs_after,
                        self.cohort_qc.clinical_df[covariate],
                        title=f"PCA after correction - {covariate}",
                    ),
                    file_name=f"PCA_after_correction_{covariate}",
                )
            )

        with tag("h3"):
            text("PC Variance Explained Before and After Batch Correction\n")
        with tag("p"):
            text(
                "Variance explained by each principal component before and after batch effect correction. Lower variance in the first few PCs suggests successful batch effect correction.\n"
            )
        with tag("h4"):
            text("Variance explained by Principal Components\n")
        with tag("div", style="display: flex; justify-content: space-around;"):
            if self.cohort_qc.data_expression_df_before is not None:
                ylim = max(
                    self.cohort_qc.pca_before.explained_variance_ratio_.max(),
                    self.cohort_qc.pca_after.explained_variance_ratio_.max(),
                )
                # First plot (PCA Variance before correction)
                with tag("div", style="flex: 1; padding: 10px;"):
                    with tag("h4"):
                        text(
                            "Variance explained by Principal Components before correction"
                        )
                    with tag("p"):
                        doc.asis(
                            self.plot_html(
                                plot=self.cohort_qc._plot_pca_variance(
                                    self.cohort_qc.pca_before.explained_variance_ratio_,
                                    ylim=ylim,
                                ),
                                file_name="PCA_variance_before_correction",
                            )
                        )
            else:
                ylim = self.cohort_qc.pca_after.explained_variance_ratio_.max()

            # Second plot (PCA Variance after correction)
            with tag("div", style="flex: 1; padding: 10px;"):
                with tag("h4"):
                    text("Variance explained by Principal Components after correction")
                with tag("p"):
                    doc.asis(
                        self.plot_html(
                            plot=self.cohort_qc._plot_pca_variance(
                                self.cohort_qc.pca_after.explained_variance_ratio_,
                                ylim=ylim,
                            ),
                            file_name="PCA_variance_after_correction",
                        )
                    )

        with tag("h3"):
            text("Association between PCs and clinical annotations")
        with tag("h4"):
            text("Interpretation")
        with tag("p"):
            text(
                " - High correlation between PCs and batch information before correction indicates batch effects. Lower correlation after correction suggests successful batch effect removal."
            )
        with tag("p"):
            text(
                " - However, it is important to note that even after batch correction, a strong association between datasets and PCs might still persist, especially in cases where:"
            )
        with tag("ul"):
            with tag("li"):
                text(
                    "The cohort is highly diverse: When dealing with a highly diverse cohort, the first few principal components might capture this diversity, leading to a natural association between the PCs and the datasets. This diversity could be in terms of biological variation, differences in sample types, or varying conditions across the datasets."
                )
            with tag("li"):
                text(
                    "Datasets contain only specific kinds of samples (e.g., normal or tumor): If datasets are composed of samples that are biologically distinct, such as normal versus tumor samples, the PCs might reflect these inherent biological differences. In such cases, the association between datasets and PCs is not due to batch effects, but rather due to the underlying biological differences that the PCA is capturing."
                )

        with tag("p"):
            text(
                "In these scenarios, the observed associations are expected and reflect meaningful biological or experimental differences rather than technical artifacts."
            )

            if self.cohort_qc.data_expression_df_before is not None:
                with tag("h4"):
                    text("Before batch effect :\n")
                with tag("p"):
                    text(
                        "Results format: statistics, pvalue, test performed and number of samples used for the test."
                    )
                    doc.asis(self.cohort_qc.association_matrix_before.to_html())
            with tag("h4"):
                text("After batch effect :\n")
                doc.asis(self.cohort_qc.association_matrix.to_html())

        return doc.getvalue()

    def cohort_summary_html_report(self) -> str:
        """
        Generate an HTML report summarizing the cohort, including details about the samples and covariates.

        Returns
        -------
        str
            HTML report as a string.
        """
        doc, tag, text = Doc().tagtext()

        # Cohort summary
        summary = self.cohort_qc.cohort_summary()
        with tag("h2"):
            text("Cohort Summary")

        with tag("ul"):
            for key, value in summary.items():
                if key == "Samples by Covariate Combination":
                    with tag("li"):
                        text(f"{key}:")
                        with tag(
                            "table",
                            border="1",
                            cellpadding="5",
                            cellspacing="0",
                            style="border-collapse: collapse;",
                        ):
                            # Header row
                            with tag("tr"):
                                for covariate in self.cohort_qc.covariates:
                                    with tag("th"):
                                        text(covariate)
                                with tag("th"):
                                    text("Number of Samples")

                            # Data rows
                            for combination in value:
                                with tag("tr"):
                                    for covariate in self.cohort_qc.covariates:
                                        with tag("td"):
                                            text(combination[covariate])
                                    with tag("td"):
                                        text(str(combination["Number of Samples"]))
                else:
                    with tag("li"):
                        text(f"{key}: {value}")

        return doc.getvalue()

    def sample_distribution_html_report(self) -> str:
        """
        Generate an HTML report comparing the distribution of samples by dataset before and after correction.

        Returns
        -------
        str
            HTML report as a string.
        """
        doc, tag, text = Doc().tagtext()
        with tag("h2"):
            text("Correction Effect Metric")

        with tag("p"):
            text(
                "This metric quantifies the impact of the batch effect correction on the variability of gene expression data. "
                "The metric is calculated as follows:"
            )

        with tag("ul"):
            with tag("li"):
                with tag("b"):
                    text("Step 1:")
                text(
                    "Calculate the Median Absolute Deviation (MAD) for each gene before correction. "
                    "This involves computing the median expression level for each gene across all samples, then calculating the "
                    "absolute deviations from this median, and finally taking the median of these deviations."
                )
                if self.cohort_qc.data_expression_df_before is not None:
                    with tag("ul"):
                        with tag("li"):
                            text("Median absolute deviation before correction =")
                        with tag("li"):
                            with tag("b"):
                                text(f"{self.cohort_qc.mad_before:.4f}")
                else:
                    with tag("ul"):
                        with tag("li"):
                            text("Data before correction not available")
            with tag("li"):
                with tag("b"):
                    text("Step 2:")
                text(
                    "Repeat the same process to calculate the MAD for each gene after the batch effect correction."
                )
                with tag("ul"):
                    with tag("li"):
                        text("Median absolute deviation after correction =")
                    with tag("li"):
                        with tag("b"):
                            text(f"{self.cohort_qc.mad_after:.4f}")
            with tag("li"):
                with tag("b"):
                    text("Step 3:")
                text(
                    "Compute the correction effect metric as the ratio of the mean MAD after correction to the mean MAD before correction."
                )
                with tag("ul"):
                    with tag("li"):
                        text("The correction effect metric calculated is:")
                    if self.cohort_qc.data_expression_df_before is not None:
                        with tag("li"):
                            with tag("b"):
                                text(f"{self.cohort_qc.effect_metric:.4f}")
                    else:
                        with tag("li"):
                            text("Data before correction not available")

        with tag("h4"):
            text("Interpretation:")
        with tag("p"):
            text(
                "This ratio help to quantifies how much variability remains after correction."
            )
        with tag("ul"):
            with tag("li"):
                text(
                    "If the ratio is close to 1, the correction may not have reduced batch effects much."
                )
            with tag("li"):
                text(
                    "A ratio significantly less than 1 indicates that the correction has reduced the batch effect (lower variability). A metric around 0.5 might indicate that the batch correction was strong. While this reduction in variability could be due to effective correction, it also raises concerns about potentially removing biological variability."
                )
            with tag("li"):
                text(
                    "A metric much higher than 1 could indicate that the correction process introduced additional variability. This could be due to overcorrection, which might have removed important biological signals along with the batch effects."
                )
            with tag("li"):
                text(
                    "The correction effect metric can depend on the number of datasets or batches. With more datasets, batch effects might be stronger, necessitating more aggressive correction, potentially leading to a lower metric. Fewer datasets might result in a correction effect closer to 1, as the variability due to batch effects could be less significant."
                )

        with tag("h3"):
            text("Sample Distribution by Covariate Combination")
        covariate_combinations = (
            self.cohort_qc.clinical_df.groupby(self.cohort_qc.covariates).size().index
        )

        with tag("p"):
            text(
                "The following boxplots compare gene expression across datasets (batches) before and after correction for different covariate combinations."
            )
        with tag("ul"):
            with tag("li"):
                with tag("b"):
                    text("Before Correction: ")
                text(
                    "Look for variability across batches, which may indicate batch effects. "
                )
            with tag("li"):
                with tag("b"):
                    text("After Correction: ")
                text(
                    "Ideally, the distributions should become more consistent across batches, suggesting effective correction."
                )

        with tag("p"):
            text(
                "These plots help assess whether the correction process has successfully reduced batch-related variability without masking important biological differences."
            )

        for combination in covariate_combinations:
            plot_tmp = self.cohort_qc.compare_sample_distribution_by_covariates(
                combination
            )

            with tag("h4"):
                text(f"Sample distribution for covariate combination: {combination}")
            with tag("p"):
                doc.asis(
                    self.plot_html(
                        plot=plot_tmp,
                        file_name=f"Sample_distribution_for_covariate_combination_{combination}",
                    )
                )

        with tag("h4"):
            text("Total Sample Distribution by Covariate Combination")
        with tag("p"):
            plot_tmp = self.cohort_qc.compare_sample_distribution_total()
            doc.asis(
                self.plot_html(
                    plot=plot_tmp,
                    file_name="Sample_distribution_for_all_covariate_combination",
                )
            )

        return doc.getvalue()

    def silhouette_html_report(self) -> str:
        """
        Generate an HTML report summarizing the Silhouette Scores before and after batch correction.

        Returns
        -------
        str
            HTML report as a string.
        """

        doc, tag, text = Doc().tagtext()
        with tag("h2"):
            text("Silhouette Score")
        with tag("p"):
            text(
                "The Silhouette Score measures how similar each sample is to its own batch compared to other batches."
            )
            text(
                "A high Silhouette Score before correction indicates strong batch effects, as samples from the same batch cluster together."
            )
            text("A lower score after correction indicates reduced batch effects.")
        with tag("ul"):
            with tag("li"):
                if self.cohort_qc.data_expression_df_before is not None:
                    text(
                        f"Silhouette Score Before Correction: {self.cohort_qc.silhouette_before:.4f}"
                    )
                else:
                    text("Data before correction not available")
        with tag("ul"):
            with tag("li"):
                text(
                    f"Silhouette Score After Correction: {self.cohort_qc.silhouette_after:.4f}"
                )
        with tag("h4"):
            text("Interpretation:")
        with tag("p"):
            text(
                " - A decrease in the Silhouette Score after correction suggests that the batch effect has been successfully mitigated."
            )

        return doc.getvalue()

    def entropy_html_report(self) -> str:
        """
        Generate an HTML report summarizing the entropy of batch mixing before and after correction.

        Returns
        -------
        str
            HTML report as a string.
        """
        doc, tag, text = Doc().tagtext()
        with tag("h2"):
            text("Entropy of Batch Mixing (EBM)")
        with tag("p"):
            text(
                "The Entropy of Batch Mixing (EBM) measures how well samples from different batches are mixed after correction."
            )
            text(
                "Higher entropy indicates better mixing, meaning that the batch effect has been reduced."
            )
        with tag("ul"):
            with tag("li"):
                if self.cohort_qc.data_expression_df_before is not None:
                    text(
                        f"Entropy Before Correction: {self.cohort_qc.entropy_before:.4f}"
                    )
                else:
                    text("Data before correction not available")
        with tag("ul"):
            with tag("li"):
                text(f"Entropy After Correction: {self.cohort_qc.entropy_after:.4f}")
        with tag("h4"):
            text("Interpretation:")
        with tag("p"):
            text(
                " - An increase in entropy after correction indicates improved mixing of batches, suggesting successful batch effect correction."
            )
        return doc.getvalue()

    def mixed_dataset_html_report(self) -> str:
        """
        Generate an HTML report summarizing the characteristics of mixed datasets and their impact on batch effect correction.

        Returns
        -------
        str
            HTML report as a string.
        """

        doc, tag, text = Doc().tagtext()

        # Title
        with tag("h1"):
            text("Mixed Dataset Summary Report")

        # Confidence in Batch Effect Correction
        with tag("h2"):
            text(
                "Confidence in Batch Effect Correction regarding Mixed Datasets and Samples"
            )

        with tag("ul"):
            with tag("li"):
                text(
                    f"Total Mixed Datasets: {self.cohort_qc.summary['total_mixed_datasets']}"
                )
            with tag("li"):
                text(
                    f"Total Non-Mixed Datasets: {self.cohort_qc.summary['total_non_mixed_datasets']}"
                )
            with tag("li"):
                text(
                    f"Total Mixed Samples: {self.cohort_qc.summary['total_mixed_samples']}"
                )
            with tag("li"):
                text(
                    f"Total Non-Mixed Samples: {self.cohort_qc.summary['total_non_mixed_samples']}"
                )

        with tag("h4"):
            mixed_dataset_ratio = self.cohort_qc.summary["total_mixed_datasets"] / (
                self.cohort_qc.summary["total_mixed_datasets"]
                + self.cohort_qc.summary["total_non_mixed_datasets"]
            )
            mixed_sample_ratio = self.cohort_qc.summary["total_mixed_samples"] / (
                self.cohort_qc.summary["total_mixed_samples"]
                + self.cohort_qc.summary["total_non_mixed_samples"]
            )

            text(
                f"Proportion of mixed datasets in the cohort: {mixed_dataset_ratio:.2%}"
            )
        with tag("h4"):
            text(f"Proportion of mixed samples in the cohort: {mixed_sample_ratio:.2%}")

        # Mixed samples by covariate
        with tag("h3"):
            text("Mixed Samples by Covariate")
        with tag("h4"):
            text("Mixed Samples")
        with tag("ul"):
            for covariate, count in self.cohort_qc.summary[
                "mixed_samples_by_covariate"
            ].items():
                with tag("li"):
                    text(f"{covariate}: {count} samples")

        with tag("h4"):
            text("Non-Mixed Samples")
        with tag("ul"):
            for covariate, count in self.cohort_qc.summary[
                "non_mixed_samples_by_covariate"
            ].items():
                with tag("li"):
                    text(f"{covariate}: {count} samples")

        # Overall proportion by covariate combination
        with tag("h3"):
            text(
                f"Overall Proportion of Mixed Samples by Covariate Combination - {', '.join(self.cohort_qc.covariates)}"
            )
        with tag("table", border="1", cellpadding="5", cellspacing="0"):
            with tag("tr"):
                with tag("th"):
                    text(
                        f"Covariate combination {', '.join(self.cohort_qc.covariates)}"
                    )
                with tag("th"):
                    text("Proportion of Mixed Samples")
            for combination, proportion in self.cohort_qc.summary[
                "overall_proportion_by_covariate_combination"
            ].items():
                with tag("tr"):
                    with tag("td"):
                        text(str(combination))
                    with tag("td"):
                        text(f"{proportion:.2%}")

            # Detailed interpretation based on the proportion
        with tag("h4"):
            text("Interpretation:")
        with tag("ul"):
            with tag("li"):
                with tag("strong"):
                    text(
                        "Very High Confidence: >50% mixed datasets and >50% mixed samples"
                    )
                with tag("p"):
                    text(
                        "Confidence is very high in the batch effect correction due to a substantial proportion of mixed datasets and samples. "
                        "This suggests that the correction algorithm was applied across a highly diverse set of conditions, "
                        "minimizing the risk that batch effects confound the biological signals. The variability across different conditions "
                        "was well-represented, leading to more reliable results."
                    )
            with tag("li"):
                with tag("strong"):
                    text(
                        "High Confidence: 30-50% mixed datasets or 30-50% mixed samples"
                    )
                with tag("p"):
                    text(
                        "Confidence is high in the batch effect correction due to a substantial proportion of mixed datasets and samples. "
                        "This indicates that the correction algorithm was applied across a diverse range of conditions, "
                        "reducing the likelihood that batch effects are confounded with biological signals. "
                        "A higher representation of mixed datasets means that the variability across different conditions "
                        "was well-represented during the correction, leading to more reliable and robust results."
                    )
            with tag("li"):
                with tag("strong"):
                    text(
                        "Moderate Confidence: 15-30% mixed datasets or 15-30% mixed samples"
                    )
                with tag("p"):
                    text(
                        "Confidence is moderate in the batch effect correction. There is a reasonable proportion of mixed datasets and samples, "
                        "suggesting that the correction was performed on a fairly diverse dataset. "
                        "However, there's still a possibility that some batch effects might not have been fully corrected if certain covariate combinations were underrepresented. "
                        "While the results are likely to be reliable, some caution is advised in interpreting the findings."
                    )
            with tag("li"):
                with tag("strong"):
                    text("Low Confidence: <15% mixed datasets and <15% mixed samples")
                with tag("p"):
                    text(
                        "Confidence is low in the batch effect correction. The mixed datasets and samples form a small proportion of the cohort, "
                        "which indicates that the correction may have been applied under limited conditions. "
                        "This can lead to insufficient representation of the variability across different conditions, "
                        "increasing the risk that batch effects may still confound the biological signals. "
                        "In such cases, the reliability of the corrected data could be compromised, and further validation might be necessary."
                    )

        # Detailed summary for each mixed dataset
        with tag("h2"):
            text("Detailed Summary for Each Mixed Dataset")
        for dataset, details in self.cohort_qc.summary["mixed_dataset_details"].items():
            with tag("h3"):
                text(f"Dataset: {dataset}")
            with tag("ul"):
                with tag("li"):
                    text(f"Total Samples: {details['total_samples']}")
                with tag("li"):
                    text("Samples by Covariate Combination:")
                    with tag("ul"):
                        for combination, count in details[
                            "samples_by_covariate_combination"
                        ].items():
                            with tag("li"):
                                text(
                                    f"{combination}: {count} samples ({details['proportion_by_covariate_combination'][combination]:.2%})"
                                )

        # Generate the HTML content
        return indent(doc.getvalue())

    def generate_html_report(self) -> str:
        """
        Combine all report sections into a full HTML report.

        Returns
        -------
        str
            Full HTML report as a string.
        """
        cohort_summary_report_text = self.cohort_summary_html_report()
        pca_report_text = self.pca_analysis_html_report()
        sample_distribution_html_report_text = self.sample_distribution_html_report()
        silhouette_report_text = self.silhouette_html_report()
        entropy_report_text = self.entropy_html_report()
        mixed_dataset_report_text = self.mixed_dataset_html_report()

        # Generate an HTML report using yattag
        doc, tag, text = Doc().tagtext()

        with tag("h1"):
            text("Cohort Quality Control Report")

        if self.cohort_qc.data_expression_df_before is None:
            with tag("h4"):
                text(
                    "Without data before correction, some metrics are not available, use data_expression_df_before parameter of the CohortMetric class to get all metrics."
                )

        doc.asis(cohort_summary_report_text)
        doc.asis(pca_report_text)
        doc.asis(sample_distribution_html_report_text)
        doc.asis(silhouette_report_text)
        doc.asis(entropy_report_text)
        doc.asis(mixed_dataset_report_text)
        return doc.getvalue()

    def save_html_report_local(self, output_path: str = ".") -> None:
        """
        Save the full HTML report to a local file.

        Parameters
        ----------
        output_path : str, optional
            Directory path to save the report, by default '.'.
        """

        # Save the HTML to a file
        with open(f"{output_path}/cohort_qc_report.html", "w") as f:
            f.write(self.html_report)
