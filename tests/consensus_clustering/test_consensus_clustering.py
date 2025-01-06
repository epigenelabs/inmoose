import importlib.resources
import logging
import os
import unittest

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from inmoose.consensus_clustering.consensus_clustering import consensusClustering


class test_consensusClustering(unittest.TestCase):
    def setUp(self) -> None:
        this_dir = importlib.resources.files(__package__)
        mock_file = this_dir.joinpath("mocked_data_consensus_clustering.csv")

        # Fix random seeds to always get the same results in the tests
        self.CC = consensusClustering(
            cluster=AgglomerativeClustering,
            mink=2,
            maxk=4,
            nb_resampling_iteration=50,
            resample_proportion=0.5,
        )
        self.mocked_data = pd.read_csv(mock_file, index_col=0)

        self.deltak_plot = this_dir.joinpath("deltak_plot.png")
        self.clustermap_plot = this_dir.joinpath("clustermap_plot.png")
        self.consensus_plot = this_dir.joinpath("clusters_consensus_plot.png")
        self.consensus_line_plot = this_dir.joinpath("clusters_consensus_line_plot.png")

    def tearDown(self) -> None:
        if os.path.exists(self.deltak_plot):
            os.remove(self.deltak_plot)

        if os.path.exists(self.clustermap_plot):
            os.remove(self.clustermap_plot)

        if os.path.exists(self.consensus_plot):
            os.remove(self.consensus_plot)

        if os.path.exists(self.consensus_line_plot):
            os.remove(self.consensus_line_plot)

    def test_internal_resample(self):
        resampled_indices = self.CC._internal_resample(
            self.mocked_data, 0.5, np.random.default_rng()
        )

        assert len(resampled_indices) == len(self.mocked_data) / 2

    def test_compute_consensus_clustering(self):
        self.CC.compute_consensus_clustering(
            self.mocked_data.to_numpy(), random_state=0
        )

        # test consensus matrix is symetric
        for i in range(len(self.CC.consensus_matrices)):
            assert np.allclose(
                self.CC.consensus_matrices[i], self.CC.consensus_matrices[i].T
            )

        # test max consensus matrices = 1 and min = 0
        assert np.max(self.CC.consensus_matrices) == 1
        assert np.min(self.CC.consensus_matrices) == 0

        # assert bestK
        assert self.CC.bestK == 3

    def test_plot_clustermap(self):
        self.CC.compute_consensus_clustering(
            self.mocked_data.to_numpy(), random_state=0
        )
        self.CC.plot_clustermap(3, self.clustermap_plot)
        assert os.path.exists(self.clustermap_plot)

    def test_plot_deltak(self):
        self.CC.compute_consensus_clustering(
            self.mocked_data.to_numpy(), random_state=0
        )
        self.CC.plot_deltak(self.deltak_plot)
        assert os.path.exists(self.deltak_plot)

    def test_plot_clusters_consensus(self):
        self.CC.compute_consensus_clustering(
            self.mocked_data.to_numpy(), random_state=0
        )
        cons_clust_df = self.CC.build_clusters_consensus_df()
        self.CC.plot_clusters_consensus(cons_clust_df, self.consensus_plot)
        assert os.path.exists(self.consensus_plot)

    def test_clusters_consensus_single_sample(self):
        np.random.seed(0)
        self.CC.compute_consensus_clustering(
            self.mocked_data.iloc[:8].to_numpy(), random_state=0
        )
        with self.assertLogs("inmoose", level=logging.WARNING) as log:
            cons_clust_df = self.CC.build_clusters_consensus_df()
            self.assertIn(
                "Single sample cluster for cluster 1 of k=4. Setting cluster consensus to NaN.",
                log.output[0],
            )
            self.assertIn(
                "Single sample cluster for cluster 2 of k=4. Setting cluster consensus to NaN.",
                log.output[1],
            )
            self.assertIn(
                "Single sample cluster for cluster 3 of k=4. Setting cluster consensus to NaN.",
                log.output[2],
            )

        assert cons_clust_df.iloc[2].isna().sum() == 3

    def test_line_plots_cluster_consensus(self):
        self.CC.compute_consensus_clustering(
            self.mocked_data.to_numpy(), random_state=0
        )
        cons_clust_df = self.CC.build_clusters_consensus_df()
        self.CC.line_plots_cluster_consensus(cons_clust_df, self.consensus_line_plot)
        assert os.path.exists(self.consensus_line_plot)
