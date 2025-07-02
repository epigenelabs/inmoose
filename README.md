<p align="center">
  
<img src="docs/source/inmoose.png" width="500">
  
</p>

<h1 align="center">

[![pypi version](https://img.shields.io/pypi/v/inmoose)](https://pypi.org/project/inmoose)
[![PyPI Downloads](https://static.pepy.tech/badge/inmoose)](https://pepy.tech/project/inmoose)
[![PyPI Downloads](https://static.pepy.tech/badge/inmoose/month)](https://pepy.tech/projects/inmoose)
[![coverage](https://img.shields.io/coverallsCoverage/github/epigenelabs/inmoose.svg)](https://coveralls.io/github/epigenelabs/inmoose)
[![Documentation Status](https://readthedocs.org/projects/inmoose/badge/?version=latest)](https://inmoose.readthedocs.io/en/latest/?badge=latest)
[![license](https://img.shields.io/pypi/l/inmoose)](LICENSE)

</h1>

<h1 align="center">
  
InMoose

</h1>

InMoose is the **In**tegrated **M**ulti **O**mic **O**pen **S**ource **E**nvironment.
It is a collection of tools for the analysis of omic data.

InMoose is developed and maintained by <img src="docs/source/epigenelogo.png" width="20"> [Epigene Labs](https://www.epigenelabs.com/).

# Installation

You can install InMoose directly with:

```
pip install inmoose
```

# Documentation

Documentation is hosted on [readthedocs.org](https://inmoose.readthedocs.io/en/latest/).

# Citing

When using InMoose in research projects, please cite:
- Colange M, Appé G, Meunier L, Weill S, Johnson WE, Nordor A, Behdenna A. (2025)
  Bridging the gap between R and Python in bulk transcriptomic data analysis with InMoose. Nature Scientific Reports 15;18104. https://doi.org/10.1038/s41598-025-03376-y.

Depending on the features you use, you may also cite one of the following papers:
- Behdenna A, Colange M, Haziza J, Gema A, Appé G, Azencot CA and Nordor A. (2023)
  pyComBat, a Python tool for batch effects correction in high-throughput molecular data using empirical Bayes methods. BMC Bioinformatics 24;459. https://doi.org/10.1186/s12859-023-05578-5.
- Colange M, Appé G, Meunier L, Weill S, Nordor A, Behdenna A. (2025)
  Differential Expression Analysis with InMoose, the Integrated Multi-Omic Open-Source Environment in Python. BMC Bioinformatics 26;160. https://doi.org/10.1186/s12859-025-06180-7.

# Batch Effect Correction

InMoose provides features to correct technical biases, also called batch
effects, in transcriptomic data:
- for microarray data, InMoose supersedes
  [pyCombat](https://github.com/epigenelabs/pycombat/), a Python3 implementation
  of [ComBat](https://doi.org/10.1093/biostatistics/kxj037), one of the most
  widely used tool for batch effect correction on microarray data.
- for RNASeq data, InMoose features a port to Python3 of
  [ComBat-Seq](https://doi.org/10.1093/nargab/lqaa078), one of the most widely
  used tool for batch effect correction on RNASeq data.

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


# Cohort QC
InMoose provides classes `CohortMetric` and `QCReport` to help to perform quality control (QC) on cohort datasets after batch effect correction.

`CohortMetric`: This class handles the analysis and provides methods for performing quality control on cohort datasets.

**Description**
The `CohortMetric` class performs a range of quality control analyses, including:
- Principal Component Analysis (PCA) to assess data variation.
- Comparison of sample distributions across different datasets or batches.
- Quantification of the effect of batch correction.
- Silhouette Score calculation to assess how well batches are separated.
- Entropy calculation to evaluate the mixing of samples from different batches.

**Usage Example**
```python
from inmoose.cohort_qc.cohort_metric import CohortMetric

cohort_quality_control = CohortMetric(
    clinical_df=clinical_data,
    batch_column="batch",
    data_expression_df=gene_expression_after_correction,
    data_expression_df_before=gene_expression_before_correction,
    covariates=["biopsy_site", "sample_type"]
)
cohort_quality_control.process()
```

`QCReport`: This class takes a CohortMetric argument, and generates an HTML report summarizing the QC results.

**Description**
The `QCReport` class extends `CohortMetric` and generates a comprehensive HTML report based on the quality control analysis. It includes visualizations and summaries of PCA, batch correction, Silhouette Scores, entropy, and more.

**Usage Example**
```python
from inmoose.cohort_qc.qc_report import QCReport

# Generate and save the QC report
qc_report = QCReport(cohort_quality_control)
qc_report.save_report(output_path='reports')
```

# Differential Expression Analysis

InMoose provides features to analyse diffentially expressed genes in bulk
transcriptomic data:
- for microarray data, InMoose features a port of
  [limma](https://doi.org/10.1093/nar/gkv007), the *de facto* standard tool
  for differential expression analysis on microarray data.
- for RNASeq data, InMoose features a ports to Python3 of
  [edgeR](https://doi.org/10.12688/f1000research.8987.2) and
  [DESeq2](https://doi.org/10.1186/s13059-014-0550-8), two of the most widely
  used tools for differential expression analysis on RNASeq data.

See the dedicated sections of the
[documentation](https://inmoose.readthedocs.io/en/latest/).

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

