pasilla
=======

.. currentmodule:: inmoose.data.pasilla

This module is a port of the R Bioconductor `pasilla package
<https://bioconductor.org/packages/release/data/experiment/html/pasilla.html>`_,
version 1.31.0.

This module provides per-exon and per-gene read counts computed for selected
genes from RNA-seq data that were presented in [Brooks2011]_.  The experiment
studied the effect of RNAi knockdown of Pasilla, the Drosophila melanogaster
ortholog of mammalian NOVA1 and NOVA2, on the transcriptome.  The `R package
vignette
<https://bioconductor.org/packages/release/data/experiment/vignettes/pasilla/inst/doc/create_objects.html>`_
describes how the data provided here were derived from the RNA-Seq read sequence
data that are provided by NCBI Gene Expression Omnibus under accession numbers
GSM461176 to GSM461181.

We describe below how to load the data to build an :class:`AnnData` object (NB:
the snippet below is wrapped in the :func:`pasilla` function for convenience)::

  import importlib.resources
  import pandas as pd
  import anndata as ad

  data_dir = importlib.resources.files("inmoose.data.pasilla")
  cts = pd.read_csv(data_dir.joinpath("pasilla_gene_counts.tsv"), sep='\t', index_col=0)
  anno = pd.read_csv(data_dir.joinpath("pasilla_sample_annotation.csv"), index_col=0)

  # The columns of `cts` and the rows of `anno` use different labels and are
  # not in the same order. We first need to harmonize them before building the
  # AnnData object.

  # first get rid of the "fb" suffix
  anno.index = [i[:-2] for i in anno.index]

  # second reorder the index
  anno = anno.reindex(cts.columns)

  # we are now ready to build the AnnData object
  adata = ad.AnnData(cts.T, anno)
  adata


Code documentation
------------------

.. autofunction:: pasilla

References
----------

.. [Brooks2011] A.N. Brooks, L. Yang, M.O. Duff, K.D. Hansen, J.W. Park, S.
   Dudoit, S.E. Brenner, B.R. Graveley. 2011. Conservation of an RNA regulatory
   map between Drosophila and mammals. *Genome Research*
   :doi:`10.1101/gr.108662.110`
