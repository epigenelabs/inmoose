Differential Expression Analysis
================================

.. currentmodule:: inmoose.deseq2

InMoose offers a Python port of the well-known R Bioconductor `DESeq2 package
<https://bioconductor.org/packages/release/bioc/html/DESeq2.html>`_ [1]_.

Note that not all features of the R package are necessarily ported. Extending
the functionality of this module will be based on user requests, so do not
hesitate to open an issue if your favorite DESeq2 is missing.

.. [1] M. I. Love, W. Huber, S. Anders. 2014. Moderated estimation of fold
   change and dispersion for RNA-seq data with DESeq2. *Genome Biology*, 15(12),
   550.  :doi:`10.1186/s13059-014-0550-8`

Code documentation
------------------

.. autosummary::
   :toctree: generated/

   ~DESeqDataSet.DESeqDataSet
   ~results.DESeqResults

   DESeq
   estimateBetaPriorVar
   estimateDispersionsFit
   estimateDispersionsGeneEst
   estimateDispersionsMAP
   estimateDispersionsPriorVar
   estimateSizeFactorsForMatrix
   ~results.filtered_p
   lfcShrink
   makeExampleDESeqDataSet
   nbinomLRT
   nbinomWaldTest
   ~results.p_adjust
   replaceOutliers
   varianceStabilizingTransformation
   ~Hmisc.wtd_quantile

