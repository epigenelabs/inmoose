pycombat_seq
============

ComBat-Seq (Zhang et al, 2020) follows on the steps of ComBat, but targets
specifically RNA-Seq data. Conceptually, ComBat-Seq is based on the same
mathematical framework as ComBat, except that its replaces the normal
distribution of microarray data by a negative binomial distribution to account
for the specificities of RNA-Seq expression data.
pycombat_seq is a direct port of ComBat-Seq to Python. Since ComBat-Seq relies
on the Bioconductor "edgeR" package, the relevant parts of edgeR have been
ported along.
Closely following the original implementation in R, pycombat_seq has results
very similar to those of ComBat-Seq in terms of batch effects correction.
Additionally, pycombat_seq is as fast, if not faster, than the original
implementation in R. It also features additional capabilities, such as fixing a
given batch as reference.

Code documentation
------------------

.. autofunction:: inmoose.pycombat.pycombat_seq
