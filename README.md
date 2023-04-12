# InMoose

InMoose is the **In**tegrated **M**ulti **O**mic **O**pen **S**ource **E**nvironment.
It is a collection of tools for the analysis of omic data.

Currently it focuses on transcriptomic data.

# Installation

You can install InMoose directly with:

```
pip install inmoose
```

# Batch Effect Correction

InMoose provides features to correct technical biases, also called batch
effects, in transcriptomic data:
- for microarray data, InMoose supersedes pyCombat [1], a Python 3
  implementation of ComBat [2], one of the most widely used tool for batch effect
  correction on microarray data.
- for RNASeq, InMoose features a port to Python3 of ComBat-Seq [3], one the most
  widely used tool for batch effect correction on RNASeq data.

To use these functions, simply import them and call them with default
parameters:
```python
from inmoose.batch import pycombat, pycombat_seq

microarray_corrected = pycombat(microarray_data, microarray_batches)
rnaseq_corrected = pycombat_seq(rnaseq_data, rnaseq_batches)
```

* `microarray_data`, `rnaseq_data`: the expression matrices, containing the
  information about the gene expression (rows) for each sample (columns).
* `microarray_batches`, `rnaseq_batches`: list of batch indices, describing the
  batch for each sample. The list of batches should contain as many elements as
  the number of samples in the expression matrix.

# How to contribute

Please refer to [CONTRIBUTING.md](https://github.com/epigenelabs/inmoose/blob/master/CONTRIBUTING.md) to learn more about the contribution guidelines.

# References

[1] Behdenna A, Haziza J, Azencot CA and Nordor A. (2020) pyComBat, a Python tool for batch effects correction in high-throughput molecular data using empirical Bayes methods. bioRxiv. https://doi.org/10.1101/2020.03.17.995431

[2] Johnson W E, et al. (2007) Adjusting batch effects in microarray expression data using empirical Bayes methods. Biostatistics, 8, 118–12. https://doi.org/10.1093/biostatistics/kxj037

[3] Zhang Y, et al. (2020) ComBat-Seq: batch effect adjustment for RNASeq count
data. NAR Genomics and Bioinformatics, 2(3). https://doi.org/10.1093/nargab/lqaa078

