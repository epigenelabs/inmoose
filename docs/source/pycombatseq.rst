pycombat-seq
============

ComBat-seq (TODO) follows on the steps of ComBat, but targets specifically
RNA-Seq data. Conceptually, ComBat-seq is based on the same mathematical
framework as ComBat, except that its replaces the normal distribution of
microarray data by a negative binomial distribution to account for the
specificities of RNA-seq expression data.
pycombat-seq is a direct port of ComBat-seq to Python. Since ComBat-seq relies
on the Bioconductor "edgeR" package, the relevant parts of edgeR have been
ported along.
Closely following the original implementation in R, pycombat-seq has results
very similar to those of ComBat-Seq in terms of batch effects correction.
Additionally, pycombat is as fast, if not faster, than the original
implementation in R. It also features additional capabilities, such as fixing a
given batch as reference.

Code documentation
------------------

.. autofunction:: inmoose.batch.pycombat_seq
