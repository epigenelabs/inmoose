pycombat_norm: batch effect corrections for microarray data
===========================================================

.. currentmodule:: inmoose.pycombat

ComBat [Johnson2007]_ is one of the most widely used tool for batch
effect correction for microarray transcriptomic data.
:func:`pycombat_norm` is a Python implementation of ComBat. Strictly following
the same mathematical framework, :func:`pycombat_norm` has results similar to
those of ComBat in terms of batch effects correction. Additionally,
:func:`pycombat_norm` is as fast, if not faster, than the original
implementation in R.

Minimal usage example
---------------------

This minimal usage example illustrates how to use pycombat_norm in a default setting,
and shows some results on ovarian cancer data. The data we use is freely available on
NCBI `Gene Expression Omnibus <https://www.ncbi.nlm.nih.gov/geo/>`_, namely:

  * GSE18520
  * GSE66957
  * GSE69428

The corresponding expression files are stored on InMoose repository in the
`data` subfolder.

.. code-block:: Python

   # import libraries
   from inmoose.pycombat import pycombat_norm
   import pandas as pd
   import matplotlib.pyplot as plt

   # prepare data
   # the datasets are dataframes where:
      # the indices correspond to the gene names
      # the column names correspond to the sample names
   # Any number (>=2) of datasets can be treated at once
   dataset_1 = pd.read_pickle("data/GSE18520.pickle")
   dataset_2 = pd.read_pickle("data/GSE66957.pickle")
   dataset_3 = pd.read_pickle("data/GSE69428.pickle")

   # merge all three datasets into a single one, keeping only common genes
   df_expression = pd.concat([dataset_1,dataset_2,dataset_3],join="inner",axis=1)

   # plot raw data
   plt.boxplot(df_expression)
   plt.show()

.. figure:: pycombat/distrib_raw.png
   :width: 600
   :alt: Distribution of raw data

   Gene expression by sample in the raw data (colored by dataset).

.. code-block:: Python

   # generate the list of batches
   datasets = [dataset_1, dataset_2, dataset_3]
   batch = [j for j,ds in enumerate(datasets) for _ in range(len(ds.columns))]

   # run pycombat_norm
   df_corrected = pycombat_norm(df_expression, batch)

   # visualize results
   plt.boxplot(df_corrected)
   plt.show()

.. figure:: pycombat/distrib_corrected.png
   :width: 600
   :alt: Distribution of corrected data

   Gene expression by sample in the batch effect-corrected data (colored by dataset).

Biological Insight
------------------

The data we used for the example above contain tumor samples and healthy
samples. A simple PCA on the raw expression data shows that, instead of
clustering by sample type, data cluster by dataset.

.. figure:: pycombat/pca_raw.png
   :width: 600
   :alt: PCA for the raw data

   PCA on the raw expression data, colored by tumor sample (blue and yellow) and healthy sample (pink).

However, after correcting batch effects with pycombat_norm, the same PCA shows
two clusters, corresponding respectively to tumor and healthy samples.

.. figure:: pycombat/pca_corrected.png
   :width: 600
   :alt: PCA for data corrected for batch effects

   PCA on the batch effect-corrected expression data, colored by tumor sample (blue and yellow) and healthy sample (pink).

Code documentation
------------------

.. autofunction:: pycombat_norm
