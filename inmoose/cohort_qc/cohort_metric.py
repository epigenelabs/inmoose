# -----------------------------------------------------------------------------
# Copyright (C) 2024-2025 L. Meunier

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

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols

from ..utils import round_scientific_notation, truncate_name


class CohortMetric:
    """
    A class for performing quality control (QC) on cohort datasets, including PCA analysis,
    batch effect correction assessment.

    Attributes
    ----------
    clinical_df : pd.DataFrame
        Clinical data frame containing patient/sample metadata.
    data_expression_df : pd.DataFrame
        Gene expression data frame after batch effect correction.
    data_expression_df_before : pd.DataFrame
        Gene expression data frame before batch effect correction.
    covariates : list
        List of covariate names used for analysis.
    mixed_datasets : list
        List of datasets identified as mixed datasets based on covariate combinations.
    n_components : int
        Number of principal components to retain.
    n_neighbors : int
        Number of nearest neighbors to consider in the entropy calculation.

    Methods
    -------
    identify_mixed_datasets()
        Identifies datasets that contain several combinations of covariates.
    pca_analysis()
        Performs PCA and computes associations between PCs and clinical annotations.
    cohort_summary()
        Generates a summary of the cohort.
    quantify_correction_effect()
        Computes a metric quantifying the effect of batch correction.
    compare_sample_distribution_by_covariates(combination)
        Compares sample distribution by covariates.
    compare_sample_distribution_total()
        Compares overall sample distribution before and after correction.
    silhouette_score()
        Computes the Silhouette Score for batch mixing before and after correction.
    compute_entropy(data_expression_df)
        Computes entropy of batch mixing.
    entropy_batch_mixing()
        Computes entropy of batch mixing before and after correction.
    summarize_and_compare_mixed_datasets()
        Summarizes and compares mixed datasets.
    process()
        Runs all QC checks.
    """

    def __init__(
        self,
        clinical_df: pd.DataFrame,
        batch_column: str,
        data_expression_df: pd.DataFrame,
        data_expression_df_before: pd.DataFrame = None,
        covariates: list = None,
        clinical_columns_of_interest: list = None,
        n_components: int = 10,
        n_neighbors: int = 5,
    ) -> None:
        """
        Initialize the CohortMetric object with clinical and expression data.

        Parameters
        ----------
        clinical_df : pandas.DataFrame
            Clinical data containing patient/sample metadata.
        data_expression_df : pandas.DataFrame
            Gene expression data after batch effect correction.
        data_expression_df_before : pandas.DataFrame
            Gene expression data before batch effect correction.
        covariates : list, optional
            List of covariate names used for analysis, by default None.
        clinical_columns_of_interest : list, optional
            List of clinical columns of interest to include in the analysis, by default None.
        n_components : int, optional
            Number of principal components to retain, by default 10.
        n_neighbors : int, optional
            Number of nearest neighbors to consider in the entropy calculation, by default 5.

        Raises
        ------
        ValueError
            If any specified covariates are not present in the clinical data frame.
        """
        common_samples = np.intersect1d(data_expression_df.columns, clinical_df.index)
        if clinical_columns_of_interest is None:
            clinical_columns_of_interest = clinical_df.columns.tolist()
        self.clinical_df = clinical_df.loc[common_samples, clinical_columns_of_interest]

        if covariates is None:
            covariates = []
        # Check if covariates are in clinical_df
        missing_covariates = [
            covariate
            for covariate in covariates
            if covariate not in clinical_df.columns
        ]
        if len(missing_covariates) > 0:
            raise ValueError(
                f"Covariates {', '.join(missing_covariates)} are not present in the clinical dataframe."
            )
        self.batch_column = batch_column
        if self.clinical_df[self.batch_column].nunique() <= 1:
            raise ValueError(
                "Cohort QC is not applicable to cohorts with only one batch."
            )
        # remove covariates with only one unique value
        self.covariates = [
            cov for cov in covariates if clinical_df[cov].unique().size > 1
        ]
        if len(self.covariates) < len(covariates):
            logging.warning(
                f"Since covariates {', '.join(np.setdiff1d(covariates, self.covariates))} have only one unique value, they are removed."
            )
        self.mixed_datasets = self.identify_mixed_datasets()

        self.data_expression_df = data_expression_df.loc[:, common_samples]
        # Check if data_expression_df_before is provided, if not, handle accordingly
        if data_expression_df_before is not None:
            self.data_expression_df_before = data_expression_df_before.loc[
                :, common_samples
            ]
        else:
            self.data_expression_df_before = None
        self.n_components = n_components
        self.n_neighbors = n_neighbors

    def identify_mixed_datasets(self) -> list:
        """
        Identify datasets that contain several combinations of covariates, indicating mixed datasets.

        Returns
        -------
        mixed_datasets: list
            A list of dataset names that contain multiple unique combinations of covariates.
        """

        # Group by dataset and covariates, then count unique combinations per dataset
        grouped = (
            self.clinical_df.groupby([self.batch_column] + self.covariates)
            .size()
            .reset_index(name="Counts")
        )

        # Count the number of unique combinations per dataset
        combination_counts = grouped.groupby(self.batch_column).size()

        # Identify datasets with more than one unique combination
        mixed_datasets = combination_counts[combination_counts > 1].index.tolist()

        return mixed_datasets

    def pca_plot(
        self,
        pcs: np.ndarray,
        labels: pd.Series,
        title: str = "PCA Plot",
        PC_list: list = [0, 1],
    ) -> plt.Figure:
        """
        Generates a PCA plot and returns it as a matplotlib figure object.

        Parameters
        ----------
        pcs: np.ndarray
            Array of principal components (e.g., from PCA).
        labels: pd.Series
            Series containing the labels to color the points by.
        title: str
            Title of the plot.

        Returns
        -------
        fig: plt.Figure
            The PCA plot as a figure object.
        """
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(12, 7))

        # Generate the scatter plot
        sns.scatterplot(
            x=pcs[:, PC_list[0]],
            y=pcs[:, PC_list[1]],
            hue=labels,
            palette="Set1",
            ax=ax,
        )

        # Set plot title and labels
        ax.set_title(title)
        ax.set_xlabel(f"PC{PC_list[0]}")
        ax.set_ylabel(f"PC{PC_list[1]}")

        # Position the legend outside the plot
        ax.legend(title=labels.name, bbox_to_anchor=(1.05, 1), loc="upper left")

        # Tight layout to adjust the plot
        plt.tight_layout()

        # Return the figure object
        return fig

    def plot_pca_variance(
        self, explained_variance: np.ndarray, ylim: float = None
    ) -> plt.Figure:
        """
        Generate a PCA variance explained barplot and return the figure object.

        Parameters
        ----------
        explained_variance: np.ndarray
            Array of variance explained ratios by each principal component.

        Returns
        -------
        fig: plt.Figure
            The PCA variance barplot as a figure object.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(range(1, len(explained_variance) + 1), explained_variance * 100)
        # Set y-axis limit to 10
        ax.set_ylim(0, ylim * 100)
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Variance Explained (%)")
        ax.set_title("PCA Variance Explained")

        # Tight layout to ensure everything fits within the figure
        plt.tight_layout()

        return fig

    def create_correlation_matrix_with_pc(
        self, clinical_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create a correlation matrix between PC columns and other clinical data elements.

        - If elements are both numeric, we use Pearson correlation.
        - If one element is categorical and the other numeric, we use a t-test or ANOVA depending on the number of categories.

        Parameters
        ----------
        clinical_data: pd.DataFrame
            Clinical data containing PCs and other data elements.

        Returns
        -------
        pd.DataFrame
            A correlation matrix between PC columns and other clinical data elements.
        """

        # Filter PC columns (assuming continuous) and other data elements
        pc_columns = [col for col in clinical_data.columns if col.startswith("PC")]
        other_columns = [
            col for col in clinical_data.columns if not col.startswith("PC")
        ]

        # Initialize an empty correlation matrix
        correlation_matrix = pd.DataFrame(index=pc_columns, columns=other_columns).map(
            lambda x: (np.nan, np.nan, "Not computed", 0)
        )

        # Numerical data elements excluding PCs
        numerical_columns = np.intersect1d(
            clinical_data.select_dtypes(include=[float, int]).columns.tolist(),
            other_columns,
        )
        cat_columns = np.setdiff1d(other_columns, numerical_columns)

        # Pearson correlation between PC and other continuous numerical data
        for pc in pc_columns:
            for num_col in numerical_columns:
                if num_col != pc:
                    temp_data = clinical_data[[pc, num_col]].dropna()
                    if temp_data.shape[0] >= 2:
                        correlation_matrix.at[pc, num_col] = tuple(
                            [
                                round_scientific_notation(val)
                                for val in stats.pearsonr(
                                    temp_data[pc], temp_data[num_col]
                                )[:2]
                            ]
                            + ["Pearson correlation", temp_data.shape[0]]
                        )

        # T-test or ANOVA between PC and categorical data elements
        for pc in pc_columns:
            for cat_col in cat_columns:
                temp_data = clinical_data[[pc, cat_col]].dropna()
                if temp_data.empty:
                    continue
                categories = temp_data[cat_col].unique()
                if len(categories) == 2:
                    # T-test for binary categorical data
                    correlation_matrix.at[pc, cat_col] = tuple(
                        [
                            round_scientific_notation(val)
                            for val in stats.ttest_ind(
                                temp_data.loc[temp_data[cat_col] == categories[0], pc],
                                temp_data.loc[temp_data[cat_col] == categories[1], pc],
                            )[:2]
                        ]
                        + ["T-test", temp_data.shape[0]]
                    )
                elif len(categories) > 2:
                    # ANOVA for multi-category categorical data
                    try:
                        model = ols(f"{pc} ~ C(Q('{cat_col}'))", data=temp_data).fit()
                        anova_table = sm.stats.anova_lm(model, typ=2)
                        correlation_matrix.at[pc, cat_col] = tuple(
                            [
                                round_scientific_notation(val)
                                for val in anova_table.loc[
                                    f"C(Q('{cat_col}'))", ["F", "PR(>F)"]
                                ].values
                            ]
                            + ["ANOVA", temp_data.shape[0]]
                        )
                    except Exception as e:
                        if (
                            "r_matrix performs f_test for using dimensions that are asymptotically non-normal"
                            in str(e)
                        ):
                            correlation_matrix.at[pc, cat_col] = (
                                np.nan,
                                np.nan,
                                "Convergence error",
                                temp_data.shape[0],
                            )
                        else:
                            raise e

        # Mirror the matrix to fill both upper and lower triangle
        return correlation_matrix.T

    def _compute_pc_associations(self, pcs: np.ndarray) -> pd.DataFrame:
        """
        Compute the correlation between principal components and clinical annotations.

        Parameters
        ----------
        pcs: np.ndarray
            Array of principal components (e.g., from PCA).

        Returns
        -------
        correlation_matrix: pd.DataFrame
            A DataFrame containing the correlation between PCs and clinical annotations.
        """
        # Add all PCs to the clinical data at once
        pc_data = self.clinical_df.copy()
        for i in range(pcs.shape[1]):
            pc_data[f"PC{i + 1}"] = pcs[:, i]

        # Generate the correlation matrix once with all PCs
        correlation_matrix = self.create_correlation_matrix_with_pc(pc_data)
        return correlation_matrix

    def pca_analysis(self) -> tuple:
        """
        Perform PCA and compute associations between principal components and clinical annotations.

        Returns
        -------
        tuple
            Tuple containing the association matrices, PCA models, and principal components before and after correction.
        """
        # Perform PCA for data after batch effect correction
        pca_after = PCA(n_components=self.n_components)
        scaler = StandardScaler()
        pcs_after = pca_after.fit_transform(
            scaler.fit_transform(self.data_expression_df.T)
        )

        # If data_expression_df_before is not None, compute PCA for it
        if self.data_expression_df_before is not None:
            pca_before = PCA(n_components=self.n_components)
            pcs_before = pca_before.fit_transform(
                scaler.fit_transform(self.data_expression_df_before.T)
            )
            association_matrix_before = self._compute_pc_associations(pcs_before)
        else:
            pca_before = None
            pcs_before = None
            association_matrix_before = None

        # Compute association between PCs and clinical annotations for data after correction
        association_matrix = self._compute_pc_associations(pcs_after)

        return (
            association_matrix_before,
            association_matrix,
            pca_before,
            pca_after,
            pcs_before,
            pcs_after,
        )

    def cohort_summary(self) -> dict:
        """
        Generate a summary of the cohort, including the number of samples, datasets, genes, and covariate combinations.

        Returns
        -------
        dict
            Summary of the cohort with details on samples, datasets, genes, and covariate combinations.
        """
        # Generate a summary of the cohort
        num_samples = self.clinical_df.shape[0]
        num_datasets = self.clinical_df[self.batch_column].nunique()
        num_genes = self.data_expression_df.shape[0]
        covariates_used = ", ".join(self.covariates)

        # Calculate the number of samples by covariate combinations
        if len(self.covariates) > 0:
            covariate_combinations = (
                self.clinical_df.groupby(self.covariates)
                .size()
                .reset_index(name="Number of Samples")
            )
        else:
            covariate_combinations = pd.DataFrame(
                {"Number of Samples": [len(self.clinical_df)]}
            )

            # Convert the covariate combinations and their counts to a readable format
        covariate_combination_summary = covariate_combinations.to_dict(orient="records")

        summary = {
            "Number of Samples": num_samples,
            "Number of Datasets": num_datasets,
            "Number of Genes": num_genes,
            "Covariates Used": covariates_used,
            "Samples by Covariate Combination": covariate_combination_summary,
        }

        return summary

    def quantify_correction_effect(self) -> tuple:
        """
        Compute a metric quantifying the effect of batch correction on the variability of gene expression data.

        Returns
        -------
        tuple
            A tuple containing the mean median absolute deviation before and after correction, and the correction effect metric.
        """
        # Compute MAD for after correction
        mad_after = stats.median_abs_deviation(
            self.data_expression_df.to_numpy(), axis=1
        )

        if self.data_expression_df_before is not None:
            # Compute MAD for before correction if data is available
            mad_before = stats.median_abs_deviation(
                self.data_expression_df_before.to_numpy(), axis=1
            )
            effect_metric = np.mean(mad_after) / np.mean(mad_before)
            return mad_before.mean(), mad_after.mean(), effect_metric
        else:
            # Handle the case where there is no before-correction data
            return None, mad_after.mean(), None

    def compare_sample_distribution_by_covariates(
        self, combination: tuple
    ) -> plt.Figure:
        """
        Compare the distribution of the samples in the batch groups by a specific covariate combination.

        Parameters
        ----------
        combination : tuple
            A tuple of covariate values representing a unique covariate combination.

        Returns
        -------
        matplotlib.figure.Figure
            The generated plot object.
        """
        # Filter the clinical data to select samples corresponding to the current covariate combination
        filter_condition = self.clinical_df[self.covariates].eq(combination).all(axis=1)
        selected_samples = self.clinical_df[filter_condition].index

        data_after = self.data_expression_df[selected_samples]
        data_after_melted = data_after.melt(var_name="Sample", value_name="Expression")
        data_after_melted["Dataset"] = self.clinical_df.loc[
            data_after_melted["Sample"], self.batch_column
        ].values
        data_after_melted["Type"] = data_after_melted["Dataset"].apply(
            lambda x: "Mixed" if x in self.mixed_datasets else "Non-Mixed"
        )
        data_after_melted["Dataset"] = (
            data_after_melted["Dataset"].astype(str).apply(truncate_name, args=(11,))
        )

        # Plot the boxplot
        # Calculate the number of unique datasets
        num_datasets = len(data_after_melted["Dataset"].unique())

        # Dynamically adjust the width of the plot based on the number of datasets
        base_width = 3  # base width for a small number of datasets
        width_per_dataset = 0.65  # additional width per dataset
        fig_width = base_width + width_per_dataset * max(num_datasets - 1, 0)

        if self.data_expression_df_before is not None:
            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(fig_width, 8))

            data_before = self.data_expression_df_before[selected_samples]
            data_before_melted = data_before.melt(
                var_name="Sample", value_name="Expression"
            )
            data_before_melted["Dataset"] = self.clinical_df.loc[
                data_before_melted["Sample"], self.batch_column
            ].values
            data_before_melted["Type"] = data_before_melted["Dataset"].apply(
                lambda x: "Mixed" if x in self.mixed_datasets else "Non-Mixed"
            )
            data_before_melted["Dataset"] = (
                data_before_melted["Dataset"]
                .astype(str)
                .apply(truncate_name, args=(11,))
            )
            # Plot before correction
            sns.violinplot(
                ax=axs[0],
                x="Dataset",
                y="Expression",
                hue="Type",
                data=data_before_melted,
                palette={"Mixed": "red", "Non-Mixed": "grey"},
            )
            axs[0].set_title("Gene expression distribution before correction")
            axs[0].set_xlabel("Dataset (Batch)")
            axs[0].set_ylabel("Gene Expression")
            axs[0].tick_params(axis="x", rotation=45)
            axs[0].legend(title="Dataset Type", loc="upper right")
            ax = axs[1]
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_width, 5))

        # Plot after correction
        sns.violinplot(
            ax=ax,
            x="Dataset",
            y="Expression",
            hue="Type",
            data=data_after_melted,
            palette={"Mixed": "red", "Non-Mixed": "grey"},
        )
        ax.set_title("Gene expression distribution after correction")
        ax.set_xlabel("Dataset (Batch)")
        ax.set_ylabel("Gene Expression")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Dataset Type", loc="upper right")

        # Return the plot as a figure object
        fig.tight_layout()
        return fig

    def compare_sample_distribution_total(self) -> plt.Figure:
        """
        Compare the overall distribution of samples by dataset before and after correction across all covariate combinations.

        Returns
        -------
        matplotlib.figure.Figure
            The generated plot object.
        """
        self.clinical_df["Covariate_Combination"] = (
            self.clinical_df[self.covariates].astype(str).agg("-".join, axis=1)
        )

        data_after = self.data_expression_df.copy()
        mean_after = data_after.mean(axis=0).reset_index()
        mean_after.columns = ["Sample", "Mean_Expression"]
        mean_after["Covariate_Combination"] = self.clinical_df.loc[
            mean_after["Sample"], "Covariate_Combination"
        ].values

        width = (11 / (12 - min(11, mean_after["Covariate_Combination"].nunique()))) + 4
        heigth = max(
            10,
            (
                max(
                    len(item)
                    for item in mean_after["Covariate_Combination"].unique()
                    if isinstance(item, str)
                )
            )
            / 3,
        )

        if self.data_expression_df_before is not None:
            data_before = self.data_expression_df_before.copy()
            mean_before = data_before.mean(axis=0).reset_index()
            mean_before.columns = ["Sample", "Mean_Expression"]
            mean_before["Covariate_Combination"] = self.clinical_df.loc[
                mean_before["Sample"], "Covariate_Combination"
            ].values

            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(width, heigth))

            # Plot before correction
            sns.violinplot(
                ax=axs[0],
                x="Covariate_Combination",
                y="Mean_Expression",
                data=mean_before,
            )
            axs[0].set_title(
                "Mean of sample gene expression distribution before correction"
            )
            axs[0].set_xlabel("Covariate combination")
            axs[0].set_ylabel("Mean gene expression by sample")
            axs[0].tick_params(axis="x", rotation=45)
            ax = axs[1]
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width, heigth / 2))

        # Plot after correction
        sns.violinplot(
            ax=ax,
            x="Covariate_Combination",
            y="Mean_Expression",
            data=mean_after,
        )
        ax.set_title("Mean of sample gene expression distribution after correction")
        ax.set_xlabel("Covariate combination")
        ax.set_ylabel("Mean gene expression by sample")
        ax.tick_params(axis="x", rotation=45)

        # Return the plot as a figure object
        fig.tight_layout()
        return fig

    def silhouette_score(self) -> tuple[float, float]:
        """
        Compute the Silhouette Score for batch mixing before and after batch correction.

        Returns
        -------
        tuple[float, float]
            Silhouette scores before and after correction.
        """
        # Compute silhouette score for after batch correction
        score_after = silhouette_score(
            self.data_expression_df.T, self.clinical_df[self.batch_column]
        )

        if self.data_expression_df_before is not None:
            # Compute silhouette score before correction if available
            score_before = silhouette_score(
                self.data_expression_df_before.T,
                self.clinical_df[self.batch_column],
            )
        else:
            score_before = None

        return score_before, score_after

    def compute_entropy(self, data_expression_df: pd.DataFrame) -> float:
        """
        Compute the entropy of batch mixing to evaluate the effectiveness of batch correction.

        Parameters
        ----------
        data_expression_df : pandas.DataFrame
            Gene expression data frame to evaluate.
        n_neighbors : int
            Number of nearest neighbors to consider in the entropy calculation.

        Returns
        -------
        float
            Mean entropy value indicating the level of batch mixing.
        """
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, metric="euclidean").fit(
            data_expression_df.T
        )
        _, indices = nbrs.kneighbors(data_expression_df.T)
        batch_labels = self.clinical_df[self.batch_column].values

        entropies = []
        for index in indices:
            neighbors_labels = batch_labels[index]
            label_counts = np.unique(neighbors_labels, return_counts=True)[1]
            probabilities = label_counts / len(neighbors_labels)
            entropy = -np.sum(probabilities * np.log2(probabilities))
            entropies.append(entropy)

        return np.mean(entropies)

    def entropy_batch_mixing(self) -> tuple[float, float]:
        """
        Compute the entropy of batch mixing before and after batch correction.

        Returns
        -------
        tuple[float, float]
            Entropy values before and after correction.
        """
        # Compute entropy of batch mixing after batch correction
        entropy_after = self.compute_entropy(self.data_expression_df)

        if self.data_expression_df_before is not None:
            # Compute entropy of batch mixing before batch correction
            entropy_before = self.compute_entropy(self.data_expression_df_before)
        else:
            entropy_before = None

        return entropy_before, entropy_after

    def summarize_and_compare_mixed_datasets(self) -> dict:
        """
        Summarize and compare mixed datasets to non-mixed datasets to evaluate their representativeness.

        Returns
        -------
        dict
            Summary with details about each mixed dataset and all mixed datasets in general.
        """

        summary = {}

        # Identify mixed and non-mixed datasets
        non_mixed_datasets = np.setdiff1d(
            self.clinical_df[self.batch_column].unique(), self.mixed_datasets
        )

        summary["total_mixed_datasets"] = len(self.mixed_datasets)
        summary["total_non_mixed_datasets"] = len(non_mixed_datasets)

        # Overall summary for mixed and non-mixed datasets
        mixed_samples = self.clinical_df[
            self.clinical_df[self.batch_column].isin(self.mixed_datasets)
        ]
        non_mixed_samples = self.clinical_df[
            self.clinical_df[self.batch_column].isin(non_mixed_datasets)
        ]

        summary["total_mixed_samples"] = mixed_samples.shape[0]
        summary["total_non_mixed_samples"] = non_mixed_samples.shape[0]

        # Summarize by covariates (e.g., biopsy site)
        covariate_tuple = (
            self.covariates if isinstance(self.covariates, list) else [self.covariates]
        )

        if len(covariate_tuple) > 0:
            summary["mixed_samples_by_covariate"] = (
                mixed_samples.groupby(covariate_tuple).size().to_dict()
            )
            summary["non_mixed_samples_by_covariate"] = (
                non_mixed_samples.groupby(covariate_tuple).size().to_dict()
            )
        else:
            summary["mixed_samples_by_covariate"] = {(): len(mixed_samples)}
            summary["non_mixed_samples_by_covariate"] = {(): len(non_mixed_samples)}

        # Compare clinical annotations between mixed and non-mixed datasets
        comparison_results = {}
        for column in self.clinical_df.columns:
            if column not in [self.batch_column] + self.covariates:
                mixed_values = mixed_samples[column].value_counts(normalize=True)
                non_mixed_values = non_mixed_samples[column].value_counts(
                    normalize=True
                )
                comparison_results[column] = {
                    "mixed": mixed_values.to_dict(),
                    "non_mixed": non_mixed_values.to_dict(),
                }

        summary["clinical_annotation_comparison"] = comparison_results

        # Summary by each mixed dataset with composition and proportions
        mixed_dataset_details = {}
        for dataset in self.mixed_datasets:
            dataset_samples = self.clinical_df[
                self.clinical_df[self.batch_column] == dataset
            ]
            samples_by_combination = (
                dataset_samples.groupby(covariate_tuple).size().to_dict()
            )
            total_samples_by_combination = (
                self.clinical_df.groupby(covariate_tuple).size().to_dict()
            )

            proportion_by_combination = {
                comb: samples_by_combination.get(comb, 0)
                / total_samples_by_combination[comb]
                for comb in total_samples_by_combination
            }

            details = {
                "total_samples": dataset_samples.shape[0],
                "samples_by_covariate_combination": samples_by_combination,
                "proportion_by_covariate_combination": proportion_by_combination,
            }
            mixed_dataset_details[dataset] = details

        summary["mixed_dataset_details"] = mixed_dataset_details

        # Compute the overall proportion of mixed samples by covariate combination across all datasets
        if len(covariate_tuple) > 0:
            total_samples_by_combination = (
                self.clinical_df.groupby(covariate_tuple).size().to_dict()
            )
            mixed_samples_by_combination = (
                mixed_samples.groupby(covariate_tuple).size().to_dict()
            )
        else:
            total_samples_by_combination = {(): len(self.clinical_df)}
            mixed_samples_by_combination = {(): len(mixed_samples)}

        overall_proportion_by_combination = {
            comb: mixed_samples_by_combination.get(comb, 0)
            / total_samples_by_combination[comb]
            for comb in total_samples_by_combination
        }

        summary["overall_proportion_by_covariate_combination"] = (
            overall_proportion_by_combination
        )

        return summary

    def process(self) -> None:
        """
        Run all QC checks including PCA analysis, batch correction assessment.
        """

        # Run all QC checks
        (
            self.association_matrix_before,
            self.association_matrix,
            self.pca_before,
            self.pca_after,
            self.pcs_before,
            self.pcs_after,
        ) = self.pca_analysis()
        self.mad_before, self.mad_after, self.effect_metric = (
            self.quantify_correction_effect()
        )
        self.silhouette_before, self.silhouette_after = self.silhouette_score()
        self.entropy_before, self.entropy_after = self.entropy_batch_mixing()
        self.summary = self.summarize_and_compare_mixed_datasets()
