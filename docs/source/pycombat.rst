=============================
Batch Effect Correction Tools
=============================

Variability in datasets not only results from biological processes, but also
from technical bias [Lander1999]_.
InMoose offers a collection of tools for the correction of such technical bias,
also called batch effects.

.. toctree::
   :maxdepth: 1
   :caption: Batch effect correction per type of data:

   for microarray data <pycombatnorm>
   for RNASeq data <pycombatseq>


References
==========

.. [Johnson2007] W. E. Johnson, C. Li, A. Rabinovic. 2007. Adjusting batch
   effects in microarray expression data using empirical Bayes methods.
   *Biostatistics*, 8, 118–12.  :doi:`10.1093/biostatistics/kxj037`

.. [Lander1999] E. S. Lander. 1999. Array of hope. *Nature Genetics*, 21(1
   Suppl), 3-4.  :doi:`10.1038/4427`

.. [Zhang2020] Y. Zhang, G. Parmigiani, W. E. Johnson. 2020. ComBat-Seq: batch
   effect adjustment for RNASeq count data. *NAR Genomics and Bioinformatics*,
   2(3).  :doi:`10.1093/nargab/lqaa078`

