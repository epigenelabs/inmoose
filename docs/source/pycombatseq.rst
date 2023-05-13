pycombat_seq: batch effect correction for RNASeq data
=====================================================

.. currentmodule:: inmoose.pycombat

ComBat-Seq (Zhang et al, 2020) [1]_ follows on the steps of ComBat, but targets
specifically RNA-Seq data. Conceptually, ComBat-Seq is based on the same
mathematical framework as ComBat, except that its replaces the normal
distribution of microarray data by a negative binomial distribution to account
for the specificities of RNA-Seq expression data.
:func:`pycombat_seq` is a direct port of ComBat-Seq to Python. Since ComBat-Seq
relies on the Bioconductor :code:`edgeR` package, the relevant parts of
:code:`edgeR` have been ported along.  Closely following the original
implementation in R, :func:`pycombat_seq` has results very similar to those of
ComBat-Seq in terms of batch effects correction.  Additionally,
:func:`pycombat_seq` is as fast, if not faster, than the original implementation
in R. It also features additional capabilities, such as fixing a given batch as
reference.

.. [1] Y. Zhang, G. Parmigiani, W. E. Johnson. 2020. ComBat-Seq: batch effect
   adjustment for RNASeq count data. *NAR Genomics and Bioinformatics*, 2(3).
   :doi:`10.1093/nargab/lqaa078`

Code documentation
------------------

.. autofunction:: pycombat_seq
