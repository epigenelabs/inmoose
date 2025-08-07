================================
Consensus Clustering
================================

InMoose implements consensus clustering [Monti2003]_, an unsupervised cluster
discovery algorithm. InMoose implementation is based on a previous
`implementation by Žiga Sajovic <https://github.com/ZigaSajovic/Consensus_Clustering>`_

Cohort Stratification
=====================

.. currentmodule:: inmoose.consensus_clustering

We illustrate the clustering-based stratification capabilities of InMoose.

We start by simulating RNA-Seq data, using the :mod:`sim` module of InMoose.

.. repl::
   import numpy as np
   import pandas as pd
   from inmoose.sim import sim_rnaseq

   # number of genes
   N = 1000
   # number of samples
   M = 1000
   assert M % 10 == 0
   P = M // 10  # 10% of M, helper variable

   # 3 batches: 20% 30% 50% of the samples
   batch = (2 * P) * [0] + (3 * P) * [1] + (5 * P) * [2]
   batch = np.array([f"batch{b}" for b in batch])
   batch0 = batch == "batch0"
   batch1 = batch == "batch1"
   batch2 = batch == "batch2"

   # 2 condition groups
   #   - group 1: 50% batch 1, 33% batch 2, 60% batch 3
   #   - group 2: 50% batch 1, 67% batch 2, 40% batch 3
   group = P * [0] + P * [1] + P * [0] + (2 * P) * [1] + (2 * P) * [0] + (3 * P) * [1]
   group = np.array([f"group{g}" for g in group])
   assert len(batch) == M and len(group) == M

   # store clinical metadata (i.e. batch and group) as a DataFrame
   clinical = pd.DataFrame({"batch": batch, "group": group})
   clinical.index = [f"sample{i}" for i in range(M)]

   # simulate data
   # random_state passes a seed to the PRNG for reproducibility
   counts = sim_rnaseq(N, M, batch=batch, group=group, random_state=42).T


We then run the consensus clustering algorithm.

.. repl::
   from inmoose.consensus_clustering.consensus_clustering import consensusClustering
   from sklearn.cluster import AgglomerativeClustering

   cc = consensusClustering(AgglomerativeClustering)
   cc.compute_consensus_clustering(counts, random_state=None)


We can now look at the clusters found.

.. repl-quiet::
   from matplotlib import rcParams
   # repl default config replaces '.' by '-' in the savefig.directory :/
   rcParams['savefig.directory'] = rcParams['savefig.directory'].replace("readthedocs-org", "readthedocs.org")


.. repl::
   from anndata import AnnData
   from inmoose.utils import Factor
   import scanpy as sc


   ad = AnnData(counts, obs=clinical)
   for k in range(2, 11):
       # Factor ensures that cluster ID are interpreted as categorical data
       ad.obs[f"k={k}"] = Factor(cc.predict(k))

   # compute the PCA
   sc.tl.pca(ad)
   # plot the PCA
   sc.pl.pca(ad, color=[f"k={k}" for k in range(2, 11)], return_fig=True).show()



References
==========

.. [Monti2003] S. Monti, P. Tamayo, J. Mesirov, T. Golub. 2003. Consensus
   Clustering: A Resampling-Based Method for Class Discovery and Visualization
   of Gene Expression Microarray Data. *Machine Learning* 52(1).
   :doi:`https://doi.org/10.1023/A:1023949509487`


Code documentation
==================

.. autosummary::
   :toctree: generated/

   ~consensus_clustering.consensusClustering
