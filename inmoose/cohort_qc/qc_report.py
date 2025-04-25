# -----------------------------------------------------------------------------
# Copyright (C) 2024-2025 L. Meunier, T. Tavernier

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
import os
from datetime import datetime
from io import BytesIO

import jinja2
import matplotlib.pyplot as plt

from .cohort_metric import CohortMetric


class QCReport:
    """
    Generate a cohort quality control report using Jinja2.
    """

    def __init__(self, cohort_qc: CohortMetric, template_dir: str = None):
        self.cohort_qc = cohort_qc

        if template_dir is None:
            template_dir = os.path.dirname(os.path.abspath(__file__))

        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.template_env.filters["percentage"] = lambda v: f"{v * 100:.2f}%"
        self.template_env.filters["format_float"] = lambda v, p=4: f"{v:.{p}f}"

    @staticmethod
    def plot_to_base64(plot: plt.Figure) -> str:
        buffer = BytesIO()
        plot.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        image_png = buffer.read()
        buffer.close()
        plt.close(plot)
        return base64.b64encode(image_png).decode("utf-8")

    def generate_report(self, template_name: str = "qc_report.html") -> str:
        covariates = [self.cohort_qc.batch_column] + self.cohort_qc.covariates

        pca_plots = []
        for covariate in covariates:
            before = None
            if self.cohort_qc.data_expression_df_before is not None:
                fig_before = self.cohort_qc.pca_plot(
                    self.cohort_qc.pcs_before,
                    self.cohort_qc.clinical_df[covariate],
                    title=f"PCA before correction - {covariate}",
                )
                before = self.plot_to_base64(fig_before)

            fig_after = self.cohort_qc.pca_plot(
                self.cohort_qc.pcs_after,
                self.cohort_qc.clinical_df[covariate],
                title=f"PCA after correction - {covariate}",
            )
            after = self.plot_to_base64(fig_after)

            pca_plots.append(
                {
                    "covariate": covariate,
                    "before": before,
                    "after": after,
                }
            )

        ylim = self.cohort_qc.pca_after.explained_variance_ratio_.max()
        if self.cohort_qc.data_expression_df_before is not None:
            ylim = max(ylim, self.cohort_qc.pca_before.explained_variance_ratio_.max())

        pca_variance_before = (
            self.plot_to_base64(
                self.cohort_qc.plot_pca_variance(
                    self.cohort_qc.pca_before.explained_variance_ratio_, ylim
                )
            )
            if self.cohort_qc.data_expression_df_before is not None
            else None
        )

        pca_variance_after = self.plot_to_base64(
            self.cohort_qc.plot_pca_variance(
                self.cohort_qc.pca_after.explained_variance_ratio_, ylim
            )
        )

        summary = self.cohort_qc.cohort_summary()
        samples_by_comb = summary.get("Samples by Covariate Combination", [])
        covariates = self.cohort_qc.covariates

        mad_before = self.cohort_qc.mad_before
        mad_after = self.cohort_qc.mad_after
        effect_metric = self.cohort_qc.effect_metric

        if len(self.cohort_qc.covariates) > 0:
            covariate_combinations = (
                self.cohort_qc.clinical_df.groupby(self.cohort_qc.covariates)
                .size()
                .index
            )
        else:
            covariate_combinations = [()]

        distribution_plots = []
        for combination in covariate_combinations:
            fig = self.cohort_qc.compare_sample_distribution_by_covariates(combination)
            img = self.plot_to_base64(fig)
            distribution_plots.append(
                {
                    "combination": combination,
                    "img": img,
                }
            )

        global_distribution_plot = self.plot_to_base64(
            self.cohort_qc.compare_sample_distribution_total()
        )

        silhouette_before = self.cohort_qc.silhouette_before
        silhouette_after = self.cohort_qc.silhouette_after

        entropy_before = self.cohort_qc.entropy_before
        entropy_after = self.cohort_qc.entropy_after

        summary_mixed = self.cohort_qc.summary
        mixed_proportions_table = [
            {**{f"cov{i + 1}": v for i, v in enumerate(comb)}, "proportion": proportion}
            for comb, proportion in summary_mixed[
                "overall_proportion_by_covariate_combination"
            ].items()
        ]

        mixed_dataset_ratio = summary_mixed["total_mixed_datasets"] / (
            summary_mixed["total_mixed_datasets"]
            + summary_mixed["total_non_mixed_datasets"]
        )
        mixed_sample_ratio = summary_mixed["total_mixed_samples"] / (
            summary_mixed["total_mixed_samples"]
            + summary_mixed["total_non_mixed_samples"]
        )

        def format_association_matrix(matrix):
            formatted = matrix.copy()
            for col in matrix.columns:
                formatted[col] = formatted[col].apply(
                    lambda v: f"<strong>statistics</strong>={v[0]}<br><strong>p-value</strong>={v[1]}<br><strong>test performed</strong>={v[2]}<br><strong>number of samples used</strong>={v[3]}"
                )
            return formatted.to_html(escape=False)

        context = {
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pca_plots": pca_plots,
            "pca_variance_before": pca_variance_before,
            "pca_variance_after": pca_variance_after,
            "association_matrix_before": format_association_matrix(
                self.cohort_qc.association_matrix_before
            ),
            "association_matrix_after": format_association_matrix(
                self.cohort_qc.association_matrix
            ),
            "summary": summary,
            "samples_by_comb": samples_by_comb,
            "covariates": covariates,
            "mad_before": mad_before,
            "mad_after": mad_after,
            "effect_metric": effect_metric,
            "distribution_plots": distribution_plots,
            "global_distribution_plot": global_distribution_plot,
            "silhouette_before": silhouette_before,
            "silhouette_after": silhouette_after,
            "entropy_before": entropy_before,
            "entropy_after": entropy_after,
            "summary_mixed": summary_mixed,
            "mixed_dataset_ratio": mixed_dataset_ratio,
            "mixed_sample_ratio": mixed_sample_ratio,
            "mixed_proportions_table": mixed_proportions_table,
            "covariate_names": self.cohort_qc.covariates,
        }

        template = self.template_env.get_template(template_name)
        return template.render(**context)

    def save_report(
        self,
        output_path: str = "cohort_qc_report.html",
        template_name: str = "qc_report.html",
    ):
        html = self.generate_report(template_name)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
