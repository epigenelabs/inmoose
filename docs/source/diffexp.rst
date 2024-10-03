================================
Differential Expression Analysis
================================

InMoose offers a Python port of the well-known R Bioconductor packages:

- `DESeq2 package <https://bioconductor.org/packages/release/bioc/html/DESeq2.html>`_ [Love2014]_ in module :doc:`deseq`.
- `edgeR package <https://bioconductor.org/packages/release/bioc/html/edgeR.html>`_ [Chen2016]_ in module :doc:`edgepy`.
- `limma package <https://bioconductor.org/packages/release/bioc/html/limma.html>`_ [Ritchie2015]_. in module :doc:`limma`.

Note that not all features of the R packages are necessarily ported. Extending
the functionality of these modules will be based on user requests, so do not
hesitate to open an issue if your favorite feature is missing.

In addition, InMoose provides a meta-analysis feature to combine the results
from different differential expression analysis tools.

Differential Expression Meta-Analysis
=====================================

.. currentmodule:: inmoose.diffexp

We illustrate the differential expression meta-analysis capabilities of InMoose
along two approaches:

- the Aggregate Data (AD) approach consists in running classical differential
  expression tools on individual cohorts then combining the results through
  *e.g.* random-effect models.
- the Individual Sample Data (ISD) consists in merging individual cohorts into a
  large meta-cohort, accounting for batch effects to eliminate inter-cohort
  biases, then running a classical differential expression analysis on the
  resulting meta-cohort.

We start by simulating RNA-Seq data, using the :mod:`sim` module of InMoose.

.. repl::
   import numpy as np
   import pandas as pd
   from inmoose.sim import sim_rnaseq

   # number of genes
   N = 10000
   # number of samples
   M = 2000
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

We then run the two meta-analysis approaches on the obtained data.

.. repl::
   from inmoose.deseq2 import DESeq, DESeqDataSet
   from inmoose.diffexp import meta_de
   from inmoose.pycombat import pycombat_seq

   # run AD meta-analysis on batches 1, 2 and 3
   # first run the differential expression analysis on each batch individually
   cohorts = [DESeqDataSet(counts[b], clinical.loc[b], design="~ group")
              for b in [batch0, batch1, batch2]]
   individual_de = [DESeq(c).results() for c in cohorts]
   # then aggregate the results
   ad = meta_de([de for de in individual_de])

   # run ISD meta-analysis on merged batches
   # first correct batch effects to properly merge batches
   # transpositions account for the difference of formats for pycombat_seq and deseq2
   harmonized_counts = pycombat_seq(counts.T, batch).T
   # then run the differential expression analysis on the merged batches
   isd = DESeq(DESeqDataSet(harmonized_counts, clinical, design="~ group")).results()

We can now compare the results obtained by the two approaches.

.. repl-quiet::
   from matplotlib import rcParams
   # repl default config replaces '.' by '-' in the savefig.directory :/
   rcParams['savefig.directory'] = rcParams['savefig.directory'].replace("readthedocs-org", "readthedocs.org")

.. repl::
   import matplotlib.pyplot as plt
   from scipy.stats import pearsonr

   ax = plt.gca()
   ax.axline((0,0), slope=1, ls="--", lw=0.3, c=".4")
   ax.scatter(isd["log2FoldChange"], ad["combined logFC"], s=3)
   corr = pearsonr(isd["log2FoldChange"], ad["combined logFC"]).statistic
   ax.annotate(f"Pearson correlation: {100*corr:.1f}%", (2, -4), size="small")
   ax.set_title("ISD", loc="center")
   ax.set_ylabel("AD", loc="center")

   plt.show()

It is possible to combine results obtained from different tools, as long as the
results of the differential expression analysis are stored as
:class:`DEResults`.  All three modules :doc:`limma`, :doc:`edgepy` and
:doc:`deseq` return sub-classes of :class:`DEResults`, thus allowing users to
perform cross-technology meta-analysis (*e.g.* by combining results from
:doc:`limma` with results from :doc:`deseq`).

Code documentation
==================

.. autosummary::
   :toctree: generated/

   DEResults

   meta_de
