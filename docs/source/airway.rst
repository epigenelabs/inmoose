airway
======

.. currentmodule:: inmoose.data.airway

This module is a port of the R Bioconductor `airway package
<https://bioconductor.org/packages/release/data/experiment/html/airway.html>`_,
version 1.23.0.

This module provides an :class:`AnnData` object of read counts in
genes for an RNA-Seq experiment on four human airway smooth muscle cell lines
treated with dexamethasone.  Details on the gene model and read counting
procedure are provided in the `R package vignette
<https://bioconductor.org/packages/release/data/experiment/vignettes/airway/inst/doc/airway.html>`.
The original publication for the data is [Himes2014]_.

Code documentation
------------------

.. autofunction:: airway

References
----------

.. [Himes2014] B.E. Himes, X. Jiang, P. Wagner, R. Hu, Q. Wang, B. Klanderman,
   R.M. Whitaker, Q. Duan, J. Lasky-Su, C. Nikolos, W. Jester, M. Johnson, R.
   Panettieri Jr, K.G. Tantisira, S.T. Weiss, Q. Lu. 2014. RNA-Seq Transcriptome
   Profiling Identifies CRISPLD2 as a Glucocorticoid Responsive Gene that
   Modulates Cytokine Function in Airway Smooth Muscle Cells. *PLoS One*.
   9(6):e99625. :doi:`10.1371/journal.pone.0099625`.
