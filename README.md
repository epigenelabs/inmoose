[![Documentation Status](https://readthedocs.org/projects/inmoose/badge/?version=latest)](https://inmoose.readthedocs.io/en/latest/?badge=latest)

# InMoose

InMoose is the **In**tegrated **M**ulti **O**mic **O**pen **S**ource **E**nvironment.
It is a collection of tools for the analysis of omic data.

# Installation

You can install InMoose directly with:

```
pip install inmoose
```

# Documentation

Documentation is hosted on [readthedocs.org](https://inmoose.readthedocs.io/en/latest/).

# Batch Effect Correction

InMoose provides features to correct technical biases, also called batch
effects, in transcriptomic data:
- for microarray data, InMoose supersedes pyCombat [1], a Python 3
  implementation of ComBat [2], one of the most widely used tool for batch effect
  correction on microarray data.
- for RNASeq, InMoose features a port to Python3 of ComBat-Seq [3], one of the
  most widely used tool for batch effect correction on RNASeq data.

To use these functions, simply import them and call them with default
parameters:
```python
from inmoose.pycombat import pycombat_norm, pycombat_seq

microarray_corrected = pycombat_norm(microarray_data, microarray_batches)
rnaseq_corrected = pycombat_seq(rnaseq_data, rnaseq_batches)
```

* `microarray_data`, `rnaseq_data`: the expression matrices, containing the
  information about the gene expression (rows) for each sample (columns).
* `microarray_batches`, `rnaseq_batches`: list of batch indices, describing the
  batch for each sample. The list of batches should contain as many elements as
  the number of samples in the expression matrix.

# Consensus clustering
InMoose provides features to compute consensus clustering, a resampling based algorithm compatible with any clustering algorithms which class implementation is instantiated with parameter `n_clusters`, and possess a `fit_predict` method, which is invoked on data.
Consensus clustering helps determining the best number of clusters to use and output confidence metrics and plots.


To use these functions, import the consensusClustering class and a clustering algorithm class:
```python
from inmoose.consensus_clustering.consensus_clustering import consensusClustering
from sklearn.cluster import AgglomerativeClustering

CC = consensusClustering(AgglomerativeClustering)
CC.compute_consensus_clustering(numpy_ndarray)
```

# How to contribute

Please refer to [CONTRIBUTING.md](https://github.com/epigenelabs/inmoose/blob/master/CONTRIBUTING.md) to learn more about the contribution guidelines.

# References

[1] Behdenna A, Colange M, Haziza J, Gema A, Appé G, Azencot CA and Nordor A. (2023) pyComBat, a Python tool for batch effects correction in high-throughput molecular data using empirical Bayes methods. BMC Bioinformatics 7;24(1):459. https://doi.org/10.1186/s12859-023-05578-5.

[2] Johnson W E, et al. (2007) Adjusting batch effects in microarray expression data using empirical Bayes methods. Biostatistics, 8, 118–12. https://doi.org/10.1093/biostatistics/kxj037

[3] Zhang Y, et al. (2020) ComBat-Seq: batch effect adjustment for RNASeq count
data. NAR Genomics and Bioinformatics, 2(3). https://doi.org/10.1093/nargab/lqaa078

