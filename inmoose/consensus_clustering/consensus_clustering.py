# -----------------------------------------------------------------------------
# Copyright (C) 2019-2021 Ž. Sajovic
# Copyright (C) 2023 S. Weill and G. Appé

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

# This file is based on the file 'consensusClustering.py' of the repository
# github.com/ZigaSajovic/Consensus_Clustering (as of July 30, 2021).

import bisect
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

from ..utils import LOGGER


class consensusClustering:
    """
    Implementation of Consensus clustering, following the paper https://link.springer.com/content/pdf/10.1023%2FA%3A1023949509487.pdf

    Arguments
    ---------
    cluster: sklearn clustering class
        Clustering algorithm to use for consensus clustering
        NOTE: the class is to be instantiated with parameter `n_clusters`, and possess a `fit_predict` method, which is invoked on data.
    mink: int
        smallest number of clusters to try, default = 2
    maxk: int
        biggest number of clusters to try, default = 10
    nb_resampling_iteration: int
        number of resamplings for each cluster number, default = 50
    resample_proportion: float
        percentage to sample. Number between 0 and 1, default = 0.5
    n_bins: int
        Number of bins used to compute histogram in compute_area_under_curve, default = 10
    consensus_matrices: ndarray[float]
        consensus matrices for each k
        NOTE: every consensus matrix is retained, like specified in the paper
    Ak: array[float]
        area under CDF for each number of clusters (see paper: section 3.3.1. Consensus distribution.)
    deltaK: array[float]
        changes in areas under CDF (see paper: section 3.3.1. Consensus distribution.)
    bestK: int
        number of clusters that was found to be best
    """

    def __init__(
        self,
        cluster,
        mink=2,
        maxk=10,
        nb_resampling_iteration=50,
        resample_proportion=0.5,
        n_bins=10,
    ):
        assert 0 <= resample_proportion <= 1, "proportion has to be between 0 and 1"
        self.cluster_ = cluster
        self.resample_proportion = resample_proportion
        self.min_k = mink
        self.max_k = maxk
        self.nb_iteration = nb_resampling_iteration
        self.nbins = n_bins
        self.consensus_matrices = None
        self.Ak = None
        self.deltaK = None
        self.bestK = None

    def _internal_resample(self, data, proportion, rand_state):
        """
        Sampling data based on the proportion of samples

        Arguments
        ---------
        data: ndarray
            numpy matrix to run the cluster on samples*features
        proportion: float
            percentage to sample
        rand_state: numpy.random.Generator
            Numpy random generator instance

        Returns
        -------
        resampled_indices
            sample's indices to use for the iteration
        """

        resampled_indices = rand_state.choice(
            range(data.shape[0]), size=int(data.shape[0] * proportion), replace=False
        )
        return resampled_indices

    def compute_consensus_clustering(self, data, random_state, verbose=False):
        """
        Fits a consensus matrix for each number of clusters

        Arguments
        ---------
        data: ndarray
            numpy matrix to run the cluster on samples*features
        random_state: int
            seed to use to generate a numpy generator instance. Default is None
        verbose: bool (default=False)
            If True, print the number of clusters for which the consensus matrix is computed
        """
        rand_state = np.random.default_rng(random_state)
        consensus_mats = np.zeros(
            (self.max_k - self.min_k + 1, data.shape[0], data.shape[0])
        )

        for k in range(self.min_k, self.max_k + 1):  # for each number of clusters
            if verbose:
                LOGGER.info(f"Computing consensus matrix for {k} clusters")
            indicator_matrix = np.zeros((data.shape[0],) * 2)
            connectivity_matrix = np.zeros((data.shape[0],) * 2)

            for _ in range(self.nb_iteration):  # resample H times
                resampled_indices = self._internal_resample(
                    data, self.resample_proportion, rand_state
                )

                Mh = self.cluster_(n_clusters=k).fit_predict(data[resampled_indices, :])

                connectivity_matrix += self.compute_iteration_connectivity_matrix(
                    resampled_indices, Mh, k, data.shape[0]
                )

                indicator_matrix += self.compute_iteration_indicator_mat(
                    resampled_indices, data.shape[0]
                )

            consensus_mats[k - self.min_k] = self.compute_consensus_mat(
                connectivity_matrix, indicator_matrix
            )

        self.consensus_matrices = consensus_mats

        self.bestK = self.compute_bestk()

    def compute_iteration_connectivity_matrix(
        self, resampled_indices, clust_res, k, nb_samples
    ):
        """
        Compute connectivity matrix

        The connectivity matrix allows to track how many times 2 elements of the matrix where sampled and clustered together

        Arguments
        ---------
        resampled_indices: array
            indices of the elements selected for the current iteration
        clust_res: array
            Clustering results
        k: int
            number of clusters
        nb_samples: int
            total number of elements of the input matrix


        Returns
        -------
        conn_mat
            Connectivity matrix for the iteration
        """
        # find indices of elements from same clusters with bisection
        # on sorted array => this is more efficient than brute force search
        conn_mat = np.zeros((nb_samples,) * 2)
        index_mapping = np.array((clust_res, resampled_indices)).T
        index_mapping = index_mapping[index_mapping[:, 0].argsort()]
        sorted_ = index_mapping[:, 0]
        id_clusts = index_mapping[:, 1]
        for i in range(k):  # for each cluster
            ia = bisect.bisect_left(sorted_, i)
            ib = bisect.bisect_right(sorted_, i)
            is_ = np.sort(id_clusts[ia:ib])
            ids_ = np.array(list(combinations(is_, 2))).T
            # sometimes only one element is in a cluster (no combinations)
            if ids_.size != 0:
                conn_mat[ids_[0], ids_[1]] += 1
        return conn_mat

    def compute_iteration_indicator_mat(self, resampled_indices, nb_samples):
        """
        Compute indicator matrix for one iteration

        The indicator matrix allows to track how many time 2 elements of the matrix where sampled together

        Arguments
        ---------
        resampled_indices: array
            sample's indices to use for the iteration
        nb_samples: int
            Number of samples in the initial matrix

        Returns
        -------
        indic_mat
            indicator matrix for the current iteration
        """
        indic_mat = np.zeros((nb_samples,) * 2)
        ids_2 = np.array(list(combinations(np.sort(resampled_indices), 2))).T
        indic_mat[ids_2[0], ids_2[1]] += 1
        return indic_mat

    def compute_consensus_mat(self, connectivity_mat, indicator_mat):
        """
        Compute consensus matrix defined as the normalized sum of the connectivity matrices of all the resampled datasets

        Arguments
        ---------
        connectivity_mat: ndarray
            connectivity matrix for k clusters. Sum of the iteration connectivity matrix
        indicator_mat: ndarray
            indicator matrix for k clusters. Sum of the iteration indicator matrix

        Returns
        -------
        consensus_mat
            Consensus matrix for k clusters
        """
        consensus_mat = connectivity_mat / (indicator_mat + 1e-8)
        # consensus_mat is upper triangular (with zeros on diagonal), we now make it symmetric
        consensus_mat += consensus_mat.T
        consensus_mat[range(consensus_mat.shape[0]), range(consensus_mat.shape[0])] = (
            1  # always with self
        )
        return consensus_mat

    def compute_bestk(self):
        """
        Get best number of clusters

        Returns
        -------
        best number of clusters
        """
        assert self.consensus_matrices is not None, "First compute consensus clustering"
        self.Ak = self.compute_area_under_curve()
        self.deltaK = self.compute_area_delta()
        return (
            np.argmax(self.deltaK) + self.min_k if self.deltaK.size > 0 else self.min_k
        )

    def compute_area_delta(self):
        """
        Compute the differences between areas under CDFs

        Returns
        -------
        Array containing the difference between the area under the CDFs for each number of clusters
        """
        assert self.consensus_matrices is not None, "First compute consensus clustering"
        return np.array(
            [
                (Ab - Aa) / Aa if i > 2 else Aa / self.nbins
                for Ab, Aa, i in zip(
                    self.Ak[1:], self.Ak[:-1], range(self.min_k, self.max_k + 1)
                )
            ]
        )

    def compute_area_under_curve(self):
        """
        Compute area under the CDFs curve

        Returns
        -------
        area_under_curve
            array of the area under the CDF for each cluster number
        """
        assert self.consensus_matrices is not None, "First compute consensus clustering"
        area_under_curve = np.zeros(self.max_k - self.min_k + 1)
        for i, m in enumerate(self.consensus_matrices):
            hist, bins = np.histogram(m.ravel(), density=True, bins=self.nbins)
            area_under_curve[i] = sum(
                h * (b - a) for b, a, h in zip(bins[1:], bins[:-1], np.cumsum(hist))
            )
        return area_under_curve

    def compute_summary_statistics(self, k):
        """
        For one prediction, compute a summary statistics, cluster consensus and item consensus, showing cluster stability and most representative cluster items.

        Cluster consensus is defined as the average consensus index between all pairs of items belonging to the cluster.
        Item consensus is defined as the average consensus index between item ei and all the (other) items in a cluster.


        Arguments
        ---------
        k: int
            Number of clusters

        Returns
        -------
        predictions
            Array of the predicted cluster
        clusters_consensus
            Array of cluster consensus
        items_consensus
            Array of the item consensus (nb_items * k)
        """
        assert self.consensus_matrices is not None, "First compute consensus clustering"
        assert self.min_k <= k <= self.max_k, (
            "Number of clusters must be between min_k and max_k"
        )
        predictions = self.predict(k)
        clusters_consensus = self.compute_clusters_consensus(predictions, k)
        items_consensus = self.compute_items_consensus(predictions, k)
        return predictions, clusters_consensus, items_consensus

    def compute_clusters_consensus(self, prediction, k):
        """
        For one prediction, compute clusters consensus, showing cluster stability.

        Cluster consensus is defined as the average consensus index between all pairs of items belonging to the cluster.

        Arguments
        ---------
        prediction: ndarray
            Array of the predicted cluster
        k: int
            Number of clusters

        Returns
        -------
        clusters_consensus
            Array of cluster consensus
        """
        assert self.min_k <= k <= self.max_k, (
            "Number of clusters must be between min_k and max_k"
        )
        clusters_consensus = np.zeros(k)
        for clust in range(k):
            ids = np.where(prediction == clust)[0]
            clust_size = len(ids)
            ids_ = np.array(list(combinations(np.sort(ids), 2))).T
            if ids_.size == 0:
                clusters_consensus[clust] = np.nan
                LOGGER.warning(
                    f"Single sample cluster for cluster {str(clust)} of k={k}. Setting cluster consensus to NaN."
                )
                continue
            clusters_consensus[clust] = self.consensus_matrices[
                k - self.min_k, ids_[0], ids_[1]
            ].sum() / (clust_size * (clust_size - 1) / 2)
        return clusters_consensus

    def compute_items_consensus(self, prediction, k):
        """
        For one prediction, compute item consensus, showing most representative cluster items.

        Item consensus is defined as the average consensus index between item ei and all the (other) items in a cluster.

        Arguments
        ---------
        prediction: ndarray
            Array of the predicted cluster
        k: int
            Number of clusters

        Returns
        -------
        items_consensus
            Array of the item consensus (nb_items * k)
        """
        assert self.min_k <= k <= self.max_k, (
            "Number of clusters must be between min_k and max_k"
        )
        items_consensus = np.zeros(
            (self.consensus_matrices[k - self.min_k].shape[0], k)
        )

        clusters, sizes = np.unique(prediction, return_counts=True)

        for id in range(items_consensus.shape[0]):
            for clust, siz in zip(clusters, sizes):
                clust_elem = np.where(prediction == clust)[0]
                if id in clust_elem:
                    cols = clust_elem[clust_elem != id]
                    items_consensus[id, clust] = np.sum(
                        self.consensus_matrices[k - self.min_k, id, cols]
                    ) / (siz - 1)
                else:
                    items_consensus[id, clust] = (
                        np.sum(self.consensus_matrices[k - self.min_k, id, clust_elem])
                        / siz
                    )
        return items_consensus

    def predict(self, k):
        """
        Predicts clusters on the consensus matrix, for k clusters using the consensus matrix

        Arguments
        ---------
        k: int
            Number of clusters

        Returns
        -------
        predicted cluster for k clusters
        """
        assert self.consensus_matrices is not None, "First compute consensus clustering"
        assert self.min_k <= k <= self.max_k, (
            "Number of clusters must be between min_k and max_k"
        )
        return self.cluster_(n_clusters=k).fit_predict(
            1 - self.consensus_matrices[k - self.min_k]
        )

    def predict_data(self, data):
        """
        Predicts clusters on the data, for best found cluster number

        Arguments
        ---------
        data: ndarray
            input matrix (samples * attributes)

        Returns
        -------
        predicted cluster for best number of clusters
        """
        assert self.consensus_matrices is not None, "First compute consensus clustering"
        return self.cluster_(n_clusters=self.bestK).fit_predict(data)

    def plot_clustermap(self, k, saving_path, col_color=None):
        """
        Compute and save a consensus clustering heatmap and dendrogram showing the consensus clustering stability.

        Arguments
        ---------
        k: int
            The number of cluster to plot the clustermap for
        saving_path: str
            Path to file where to store the result plot
        col_color: dataframe
            dataframe of color to annotate the clustermap.
            NOTE: the colors must be ordered in the same way as the initial data matrix and be in hexadecimal or rgb format (cf clustermap seaborn documentation)

        """
        assert self.consensus_matrices is not None, "First compute consensus clustering"
        assert self.min_k <= k <= self.max_k, (
            "Number of clusters must be between min_k and max_k"
        )
        cluster_map = sn.clustermap(
            pd.DataFrame(self.consensus_matrices[k - self.min_k]),
            method="ward",
            metric="euclidean",
            cmap=sn.color_palette("Blues", as_cmap=True),
            yticklabels=False,
            xticklabels=False,
            figsize=(10, 10),
            col_colors=col_color,
        )
        cluster_map.ax_col_dendrogram.set_title(
            f"Consensus clustering heatmap and dendrogram: n_clusters={k}"
        )
        cluster_map.figure.savefig(saving_path)
        return cluster_map

    def build_clusters_consensus_df(self):
        """
        Compute cluster consensus for each k from min_k to max_k and return a dataframe to use in the plot_clusters_consensus
        """
        consensus_clusters = []
        for k in range(self.min_k, self.max_k + 1):
            prediction = self.predict(k)
            consensus_clust = self.compute_clusters_consensus(prediction, k)
            consensus_clusters.append(consensus_clust)

        return pd.DataFrame(
            consensus_clusters,
            index=[f"k={i}" for i in range(self.min_k, self.max_k + 1)],
        )

    def plot_clusters_consensus(self, cons_clust_df, fig_path):
        """
        Plot cluster consensus results showing the statibility of clusters

        To plot cluster consensus you first need to compute clusters consensus
        for each K between mi_k and max_k.

        Arguments
        ---------
        cons_clust_df: dataframe
            clusters consensus results formatted in a dataframe max_k-min_k+1*max_k
        fig_path: str
            Path to store the plot
        """
        fig, ax = plt.subplots(figsize=(13, 8))

        cons_clust_df.plot(
            kind="bar",
            stacked=False,
            width=1,
            edgecolor="white",
            linewidth=1,
            align="center",
            ax=ax,
        )
        plt.tick_params(rotation=45)
        plt.ylim(0, 1)
        plt.title("Cluster consensus measuring cluster stability")
        plt.xlabel("Number of clusters")
        plt.ylabel("Cluster consensus measuring cluster stability")
        ax.legend(bbox_to_anchor=(1, 1), title="Cluster")
        plt.savefig(fig_path)

    def line_plots_cluster_consensus(self, cons_clust_df, fig_path, threshold=0.75):
        """
        Line plots of the % and number of cluster with cluster consensus > threshold

        To plot cluster consensus you first need to compute clusters consensus
        for each K between mi_k and max_k.

        Arguments
        -----------
        cons_clust_df: dataframe
            clusters consensus results formatted in a dataframe max_k-min_k+1*max_k
        fig_path: str
            Path to store the plot
        threshold: float
            cluster consensus minimum value to be considered as stable
            default=0.75
        """
        ratio_stable_cluster = cons_clust_df.apply(
            lambda x: (x > threshold).sum() / (~x.isna()).sum(), axis=1
        )
        nb_stable_cluster = cons_clust_df.apply(lambda x: (x > threshold).sum(), axis=1)

        fig, (ax1, ax2) = plt.subplots(2)
        sn.lineplot(nb_stable_cluster, ax=ax1)
        ax1.set_title("Cluster stability")
        ax1.set(ylabel=f"number of cluster \n with stability>{threshold}", xlabel=" ")
        sn.lineplot(ratio_stable_cluster, ax=ax2, color="orange")
        ax2.set(
            ylabel=f"% of cluster \n with stability>{threshold}",
            xlabel="total nb of cluster",
        )
        plt.savefig(fig_path)

    def plot_deltak(self, fig_path):
        """
        Plot the curve showing the relative change in area under the curve

        Arguments
        ---------
        fig_path: str
            Path to store the plot
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.plot(range(self.min_k, self.max_k), self.deltaK, marker="o", color="black")
        plt.title(
            "Relative change in area under the curve, showing the best number of clusters"
        )
        plt.xlabel("Number of clusters")
        plt.ylabel("Relative change in area under the curve")
        plt.savefig(fig_path)
