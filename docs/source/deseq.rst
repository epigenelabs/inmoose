======
deseq2
======

.. currentmodule:: inmoose.deseq2

This module is a partial port of the R Bioconductor `DESeq2 package
<https://bioconductor.org/packages/release/bioc/html/DESeq2.html>`_ [Love2014]_.

.. repl::
   from inmoose.deseq2 import *
   import patsy
   import numpy as np
   import matplotlib.pyplot as plt
   import pandas as pd
   import scipy


.. repl-quiet::
   pd.options.display.max_columns = 10
   from matplotlib import rcParams
   # repl default config replaces '.' by '-' in the savefig.directory :/
   rcParams['savefig.directory'] = rcParams['savefig.directory'].replace("readthedocs-org", "readthedocs.org")

Quick start
===========

Here we show the most basic steps for a differential expression analysis. There
are a variety of steps upstream of DESeq2 that result in the generation of
counts or estimated counts for each sample, which we will discuss in the
sections below. This code chunk assumes that you have a count matrix called
:code:`cts` and a table of sample information called :code:`coldata`.  The
:code:`design` indicates how to model the samples, here, that we want to measure
the effect of the condition, controlling for batch differences. The two factor
variables :code:`batch` and :code:`condition` should  be columns of
:code:`coldata`:

>>> dds = DESeqDataSet(countData = cts,
...                    clinicalData = coldata,
...                    design = "~ batch + condition")
>>> dds = DESeq(dds)
>>> # list the coefficients
>>> dds.resultsNames()
>>> res = dds.results(name = "condition_trt_vs_untrt")
>>> # or to shrink the log fold changes association with condition
>>> res = dds.lfcShrink(coef = "condition_trt_vs_untrt", type = "apeglm")


Input data
==========

Why un-normalized counts?
-------------------------

As input, the DESeq2 package expects count data as obtained, *e.g.* from RNA-seq
or another high-throughput sequencing experiment, in the form of a matrix of
integer values. The value in the :math:`i`-th row and the :math:`j`-th column of
the matrix tells how many reads can be assigned to gene :math:`i` in sample
:math:`j`.  Analogously, for other types of assays, the rows of the matrix might
correspond *e.g.* to binding regions (with ChIP-Seq) or peptide sequences (with
quantitative mass spectrometry). We will list method for obtaining count
matrices in sections below.

The values in the matrix should be un-normalized counts or estimated counts of
sequencing reads (for single-end RNA-seq) or fragments (for paired-end RNA-seq).
The `RNA-seq workflow <http://www.bioconductor.org/help/workflows/rnaseqGene/>`_
describes multiple techniques for preparing such count matrices.  It is
important to provide count matrices as input for DESeq2's statistical model
[Love2014]_ to hold, as only the count values allow assessing the measurement
precision correctly. The DESeq2 model internally corrects for library size, so
transformed or normalized values such as counts scaled by library size should
not be used as input.

The DESeqDataSet
----------------

The object class used by the DESeq2 package to store the read counts and the
intermediate estimated quantities during statistical analysis is the
:class:`~DESeqDataSet.DESeqDataSet`, which will usually be represented in the
code here as an object :code:`dds`.

A technical detail is that the :class:`~DESeqDataSet.DESeqDataSet` class extends
the :class:`AnnData` class of the `anndata
<https://github.com/scverse/anndata>`_ package.

A :class:`~DESeqDataSet.DESeqDataSet` object must have an associated **design
matrix**.  The design matrix expresses the variables which will be used in
modeling. The matrix is built from a formula starting with a tilde (~) followed
by the variables with plus signs between them (it will be coerced into a formula
if it is not already). The design can be changed later, however then all
differential analysis steps should be repeated, as the design formula is used to
estimate the dispersions and to estimate the log2 fold changes of the model.

.. note::
   In order to benefit from the default settings of the package, you should put
   the variable of interest at the end of the formula and make sure the control
   level is the first level.

We will now show 2 ways of constructing a :class:`~DESeqDataSet.DESeqDataSet`,
depending on what pipeline was used upstream of DESeq2 to generated counts or
estimated counts:

  1. From a count matrix :ref:`countmat`
  2. From an :class:`AnnData` object :ref:`ad`

.. note::
   The original R package allows to build a :class:`~DESeqDataSet.DESeqDataSet`
   from transcript abundance files and htseq count files, but those features
   have not yet been ported into `inmoose`.

.. _countmat:

Count matrix input
------------------

The constructor :meth:`.DESeqDataSet.__init__` can be used if you already have a
matrix of read counts prepared from another source. Another method for quickly
producing count matrices from alignment files is the :code:`featureCounts`
function [Liao2013]_ in the `Rsubread
<http://bioconductor.org/packages/Rsubread>`_ package.  To use the constructor
of :meth:`.DESeqDataSet.__init__`, the user should provide the counts matrix,
the information about the samples (the rows of the count matrix) as a data
frame, and the design formula.

To demonstrate the construction of :class:`~DESeqDataSet.DESeqDataSet` from a
count matrix, we will read in count data from the `pasilla
<http://bioconductor.org/packages/pasilla>`_ package, available in `inmoose`. We
read in a count matrix, which we will name :code:`cts`, and the sample
information table, which we will name :code:`sample_data`.  Further below we
describe how to extract these objects from, *e.g.* :code:`featureCounts`
output:

.. repl::

   import importlib.resources
   from inmoose.utils import Factor

   data_dir = importlib.resources.files("inmoose.data.pasilla")
   pasCts = data_dir.joinpath("pasilla_gene_counts.tsv")
   pasAnno = data_dir.joinpath("pasilla_sample_annotation.csv")
   cts = pd.read_csv(pasCts, sep='\t', index_col=0)
   sample_data = pd.read_csv(pasAnno, index_col=0)
   sample_data = sample_data[["condition", "type"]]
   sample_data["condition"] = Factor(sample_data["condition"])
   sample_data["type"] = Factor(sample_data["type"])

We examine the count matrix and sample data to see if they are consistent in
terms of sample order:

.. repl::
   cts.head(2)
   sample_data

Note that these are not in the same order with respect to samples!

It is absolutely critical that the rows of the count matrix and the rows of
the sample data (information about samples) are in the same order.  DESeq2 will
not make guesses as to which row of the count matrix belongs to which row of
the sample data, these must be provided to DESeq2 already in consistent order.

As they are not in the correct order as given, we need to re-arrange one or the
other so that they are consistent in terms of sample order (if we do not, later
functions would produce an error). We additionally need to chop off the `"fb"`
of the row names of :code:`sample_data`, so the naming is consistent:

.. repl::

   sample_data.index = [i[:-2] for i in sample_data.index]
   all(sample_data.index.isin(cts.columns))
   all(sample_data.index == cts.columns)
   sample_data = sample_data.reindex(cts.columns)
   all(sample_data.index == cts.columns)

If you have used the :code:`featureCounts` function [Liao2013]_ in the `Rsubread
<http://bioconductor.org/packages/Rsubread>`_ package, the matrix of read counts
can be directly provided from the `"counts"` element in the list output.  The
count matrix and sample data can typically be read into Python from flat files
using import functions from :code:`pandas` or :code:`numpy`.

With the count matrix, :code:`cts`, and the sample information,
:code:`sample_data`, we can construct a :class:`~DESeqDataSet.DESeqDataSet`:

.. repl::

   dds = DESeqDataSet(countData = cts.T,
                      clinicalData = sample_data,
                      design = "~ condition")
   dds

If you have additional feature data, it can be added to the
:class:`~DESeqDataSet.DESeqDataSet` by adding to the metadata columns of a newly
constructed object. (Here we add redundant data just for demonstration, as the
gene names are already the rownames of the :code:`dds`.):

.. repl::

   featureData = dds.var.index
   dds.var["featureData"] = featureData
   dds.var.head()


.. _ad:

:class:`AnnData` input
----------------------

If one has already created or obtained an :class:`AnnData`, it can be easily
input into DESeq2 as follows. First we load the module containing the `airway`
dataset:

.. repl::

   from inmoose.data.airway import airway
   ad = airway()

The constructor function below shows the generation of a
:class:`~DESeqDataSet.DESeqDataSet` from an :class:`AnnData` :code:`ad`:

.. repl::

   ddsAD = DESeqDataSet(ad, design = "~ cell + dex")
   ddsAD


Pre-filtering
-------------

While it is not necessary to pre-filter low count genes before running the
DESeq2 functions, there are two reasons which make pre-filtering useful: by
removing rows in which there are very few reads, we reduce the memory size of
the :code:`dds` data object, and we increase the speed of the transformation and
testing functions within DESeq2. It can also improve visualizations, as features
with no information for differential expression are not plotted.

Here we perform a minimal pre-filtering to keep only rows that have at least 10
reads total. Note that more strict filtering to increase power is
*automatically* applied via :ref:`independent filtering<indfilt>` on the mean of
normalized counts within the :meth:`.DESeqDataSet.results` function:

.. repl::

   keep = dds.counts().sum(axis=1) >= 10
   dds = dds[keep,:]

Alternatively, a popular filter is to ensure at least :code:`X` samples with a
count of 10 or more, where :code:`X` can be chosen as the sample size of the
smallest group of samples:

>>> keep = (dds.counts() >= 10).sum(axis=1) >= X
>>> dds = dds[keep,:]

.. _factorLevels:

Note on factor levels
---------------------

By default, Python will choose a *reference level* for factors based on
alphabetical order, it chooses the first value as the reference. Then, if you 
never tell the DESeq2 functions which level you want to compare against 
(*e.g.* which level represents the control group), the comparisons will 
be based on the alphabetical order of the levels. There are two
solutions: you can either explicitly tell :meth:`.DESeqDataSet.results` which
comparison to make using the :code:`contrast` argument (this will be shown
later), or you can explicitly set the factors levels by specifying the desired 
reference value first. In order to see the change of reference levels reflected 
in the results names, you need to either run :func:`DESeq` or 
:func:`nbinomWaldTest` / :func:`nbinomLRT` after the re-leveling operation.  
Setting the factor levels can be done with the:code:`reorder_categories` function:

.. repl::

   dds.obs["condition"] = dds.obs["condition"].cat.reorder_categories(["untreated", "treated"])

If you need to subset the columns of a :class:`~DESeqDataSet.DESeqDataSet`,
*i.e.* when removing certain samples from the analysis, it is possible that all
the samples for one or more levels of a variable in the design formula would be
removed. In this case, the :code:`remove_unused_categories` function can be used
to remove those levels which do not have samples in the current
:class:`~DESeqDataSet.DESeqDataSet`:

.. repl::

   dds.obs["condition"] = dds.obs["condition"].cat.remove_unused_categories()

Collapsing technical replicates
-------------------------------

DESeq2 provides a function :func:`collapseReplicates` which can assist in
combining the counts from technical replicates into single columns of the count
matrix. The term *technical replicate* implies multiple sequencing runs of the
same library.  You should not collapse biological replicates using this
function.  See the manual page for an example of the use of
:func:`collapseReplicates`.

About the pasilla dataset
-------------------------

We continue with the :doc:`/pasilla` data constructed from the count matrix
method above. This data set is from an experiment on *Drosophila melanogaster*
cell cultures and investigated the effect of RNAi knock-down of the splicing
factor *pasilla* [Brooks2011]_.  The detailed transcript of the production of
the `pasilla <http://bioconductor.org/packages/pasilla>`_ data is provided in
the vignette of the data package `pasilla
<http://bioconductor.org/packages/pasilla>`_.

.. _de:

Differential expression analysis
================================

The standard differential expression analysis steps are wrapped into a single
function :func:`DESeq`. The estimation steps performed by this function are
described :ref:`below<theory>`, in the manual page for :func:`DESeq` and in the
Methods section of the DESeq2 publication [Love2014]_.

Results tables are generated using the function :meth:`.DESeqDataSet.results`,
which extracts a results table with log2 fold changes, *p*-values and adjusted
*p*-values. With no additional arguments to :meth:`.DESeqDataSet.results`, the
log2 fold change and Wald test *p*-value will be for the **last variable** in
the design formula, and if this is a factor, the comparison will be the **last
level** of this variable over the **reference level** (see previous :ref:`note
on factor levels<factorlevels>`).  However, the order of the variables of the
design does not matter so long as the user specifies the comparison to build a
results table for, using the :code:`name` or :code:`contrast` arguments of
:meth:`.DESeqDataSet.results`.

Details about the comparison are printed to the console, directly above the
results table. The text, `condition treated vs untreated`, tells you that the
estimates are of the logarithmic fold change log2(treated/untreated):

.. repl::

   dds.design = "~ condition"
   dds = DESeq(dds)
   res = dds.results()
   res.head()

Note that we could have specified the coefficient or contrast we want to build a
results table for, using either of the following equivalent commands:

>>> res = dds.results(name="condition_treated_vs_untreated")
>>> res = dds.results(contrast=["condition","treated","untreated"])

One exception to the equivalence of these two commands, is that, using
:code:`contrast` will additionally set to 0 the estimated LFC in a comparison of
two groups, where all of the counts in the two groups are equal to 0 (while
other groups have positive counts). As this may be a desired feature to have the
LFC in these cases set to 0, one can use :code:`contrast` to build these results
tables.  More information about extracting specific coefficients from a fitted
:class:`~DESeqDataSet.DESeqDataSet` object can be found in the help page
:meth:`.DESeqDataSet.results`.  The use of the :code:`contrast` argument is also
further discussed :ref:`below<contrasts>`.

.. _lfcShrink:

Log fold change shrinkage for visualization and ranking
-------------------------------------------------------

Shrinkage of effect size (LFC estimates) is useful for visualization and ranking
of genes. To shrink the LFC, we pass the :code:`dds` object to the function
:func:`lfcShrink`. Below we specify to use the :code:`apeglm` method for effect
size shrinkage [Zhu2018]_, which improves on the previous estimator.

We provide the :code:`dds` object and the name or number of the coefficient we
want to shrink, where the number refers to the order of the coefficient as it
appears in :code:`dds.resultsNames()`:

>>> dds.resultsNames()
>>> resLFC = dds.lfcShrink(coef="condition_treated_vs_untreated", type="apeglm")
>>> resLFC

Shrinkage estimation is discussed more in a :ref:`later section<shrink>`.

*p*-values and adjusted *p*-values
----------------------------------

We can order our results table by the smallest *p*-value:

.. repl::
   resOrdered = res.sort_values(by="pvalue")

We can summarize some basic tallies using the :meth:`.DESeqResults.summary`
function:

.. repl::
   print(res.summary())

How many adjusted *p*-values were less than 0.1:

.. repl::
   (res.padj < 0.1).sum()

The :meth:`.DESeqDataSet.results` function contains a number of arguments to
customize the results table which is generated. You can read about these
arguments by looking up the documentation of :meth:`.DESeqDataSet.results`.
Note that the :meth:`.DESeqDataSet.results` function automatically performs
independent filtering based on the mean of normalized counts for each gene,
optimizing the number of genes which will have an adjusted *p*-value below a
given FDR cutoff, :code:`alpha`.  Independent filtering is further discussed
:ref:`below<indfilt>`.  By default the argument :code:`alpha` is set to 0.1.  If
the adjusted *p*-value cutoff will be a value other than 0.1, :code:`alpha`
should be set to that value:

.. repl::
   res05 = dds.results(alpha=0.05)
   print(res05.summary())
   (res05.padj < 0.05).sum()

.. _IHW:

..
  Independent hypothesis weighting
  --------------------------------

  A generalization of the idea of *p*-value filtering is to *weight* hypotheses to
  optimize power. A Bioconductor package, `IHW
  <http://bioconductor.org/packages/IHW>`_, is available that implements the
  method of *Independent Hypothesis Weighting* [Ignatiadis2016]_.  Here we show
  the use of *IHW* for *p*-value adjustment of DESeq2 results.  For more details,
  please see the vignette of the `IHW <http://bioconductor.org/packages/IHW>`_
  package. The *IHW* result object is stored in the metadata::

    # (unevaluated code chunk)
    library("IHW")
    resIHW <- results(dds, filterFun=ihw)
    summary(resIHW)
    sum(resIHW$padj < 0.1, na.rm=TRUE)
    metadata(resIHW)$ihwResult

  .. note::
     If the results of independent hypothesis weighting are used in published
     research, please cite:

      Ignatiadis, N., Klaus, B., Zaugg, J.B., Huber, W. (2016)
      Data-driven hypothesis weighting increases detection power in genome-scale multiple testing.
      *Nature Methods*, **13**:7.
      :doi:`http://dx.doi.org/10.1038/nmeth.3885`

  For advanced users, note that all the values calculated by the DESeq2 package
  are stored in the :class:`~DESeqDataSet.DESeqDataSet` object or the
  :class:`DESeqResults` object, and access to these values is discussed
  :ref:`below<access>`.

Exploring and exporting results
===============================

MA-plot
-------

In DESeq2, the function :meth:`.DESeqResults.plotMA` shows the log2 fold changes
attributable to a given variable over the mean of normalized counts for all the
samples in the :class:`~DESeqDataSet.DESeqDataSet`.  Points will be colored red
if the adjusted *p*-value is less than 0.1.  Points which fall out of the window
are plotted as open triangles pointing either up or down:

.. repl::
   res.plotMA(ylim=[-2,2])
   plt.show()

It is more useful visualize the MA-plot for the shrunken log2 fold changes,
which remove the noise associated with log2 fold changes from low count genes
without requiring arbitrary filtering thresholds:

.. repl::
   resLFC.plotMA(ylim=[-2,2])
   plt.show()

..
  After calling :meth:`.DESeqResults.plotMA`, one can use the function
  :func:`identify` to interactively detect the row number of individual genes by
  clicking on the plot.  One can then recover the gene identifiers by saving the
  resulting indices::

    idx = identify(res.baseMean, res.log2FoldChange)
    res.index[idx]

.. _shrink:

Alternative shrinkage estimators
--------------------------------

The moderated log fold changes proposed by [Love2014]_ use a normal prior
distribution, centered on zero and with a scale that is fit to the data. The
shrunken log fold changes are useful for ranking and visualization, without the
need for arbitrary filters on low count genes. The normal prior can sometimes
produce too strong of shrinkage for certain datasets. In DESeq2 version 1.18, we
include two additional adaptive shrinkage estimators, available via the
:code:`type` argument of :func:`lfcShrink`.

The options for :code:`type` are:

* :code:`apeglm` is the adaptive t prior shrinkage estimator from the `apeglm
  <http://bioconductor.org/packages/apeglm>`_ package [Zhu2018]_. As of version
  1.28.0, it is the default estimator.
* :code:`ashr` is the adaptive shrinkage estimator from the `ashr
  <https://github.com/stephens999/ashr>`_ package [Stephens2016]_.  Here DESeq2
  uses the ashr option to fit a mixture of Normal distributions to form the
  prior, with :code:`method="shrinkage"`.
* :code:`normal`: is the the original DESeq2 shrinkage estimator, an adaptive
  Normal distribution as prior.

If the shrinkage estimator :code:`apeglm` is used in published research, please
cite [Zhu2018]_.

If the shrinkage estimator :code:`ashr` is used in published research, please
cite [Stephens2016]_.

In the LFC shrinkage code above, we specified
:code:`coef="condition_treated_vs_untreated"`. We can also just specify the
coefficient by the order that it appears in :code:`dds.resultsNames()`, in this
case :code:`coef=2`. For more details explaining how the shrinkage estimators
differ, and what kinds of designs, contrasts and output is provided by each, see
the :ref:`extended section on shrinkage estimators<moreshrink>`:

>>> dds.resultsNames()
>>> # because we are interested in treated vs untreated, we set 'coef=2'
>>> resNorm = dds.lfcShrink(coef=2, type="normal")
>>> resAsh = dds.lfcShrink(coef=2, type="ashr")

..
  ```{r fig.width=8, fig.height=3}
  par(mfrow=c(1,3), mar=c(4,4,2,1))
  xlim <- c(1,1e5); ylim <- c(-3,3)
  plotMA(resLFC, xlim=xlim, ylim=ylim, main="apeglm")
  plotMA(resNorm, xlim=xlim, ylim=ylim, main="normal")
  plotMA(resAsh, xlim=xlim, ylim=ylim, main="ashr")
  ```

..
  .. note::
     We have sped up the :code:`apeglm` method so it takes roughly about the same
     amount of time as :code:`normal`, e.g. ~5 seconds for the :code:`pasilla`
     dataset of ~10,000 genes and 7 samples.  If fast shrinkage estimation of LFC
     is needed, **but the posterior standard deviation is not needed**, setting
     :code:`apeMethod="nbinomC"` will produce a ~10x speedup, but the
     :code:`lfcSE` column will be returned with :code:`NA`.  A variant of this
     fast method, :code:`apeMethod="nbinomC*"` includes random starts.


.. note::
   If there is unwanted variation present in the data (*e.g.* batch effects) it
   is always recommended to correct for this, which can be accommodated in
   DESeq2 by including in the design any known batch variables or by using
   functions/packages such as :func:`.pycombat_seq`, :code:`svaseq` in `sva
   <http://bioconductor.org/packages/sva>`_ [Leek2014]_ or the :code:`RUV`
   functions in `RUVSeq <http://bioconductor.org/packages/RUVSeq>`_ [Risso2014]_
   to estimate variables that capture the unwanted variation.  In addition, the
   ashr developers have a `specific method <https://github.com/dcgerard/vicar>`_
   for accounting for unwanted variation in combination with ashr [Gerard2020]_.

..
  Plot counts
  -----------

  It can also be useful to examine the counts of reads for a single gene across
  the groups. A simple function for making this plot is :func:`plotCounts`, which
  normalizes counts by the estimated size factors (or normalization factors if
  these were used) and adds a pseudocount of 1/2 to allow for log scale plotting.
  The counts are grouped by the variables in :code:`intgroup`, where more than one
  variable can be specified. Here we specify the gene which had the smallest
  *p*-value from the results table created above. You can select the gene to plot
  by rowname or by numeric index::

    plotCounts(dds, gene=which.min(res$padj), intgroup="condition")

  For customized plotting, an argument :code:`returnData` specifies that the
  function should only return a data frame for plotting with :code:`ggplot`::

    d <- plotCounts(dds, gene=which.min(res$padj), intgroup="condition",
                    returnData=TRUE)
    library("ggplot2")
    ggplot(d, aes(x=condition, y=count)) +
      geom_point(position=position_jitter(w=0.1,h=0)) +
      scale_y_log10(breaks=c(25,100,400))


More information on results columns
-----------------------------------

Information about which variables and tests were used can be found by inspecting
the attribute :code:`description` on the :class:`~results.DESeqResults` object:

.. repl::
   res.description

For a particular gene, a log2 fold change of -1 for :code:`condition treated vs
untreated` means that the treatment induces a multiplicative change in observed
gene expression level of :math:`2^{-1} = 0.5` compared to the untreated
condition. If the variable of interest is continuous-valued, then the reported
log2 fold change is per unit of change of that variable.

.. _pvaluesNA:

.. admonition:: Note on *p*-values set to :code:`NA`

   Some values in the results table can be set to :code:`NA` for one of the
   following reasons:

   * If within a row, all samples have zero counts, the :code:`baseMean` column
     will be zero, and the log2 fold change estimates, *p*-value and adjusted
     *p*-value will all be set to :code:`NA`.
   * If a row contains a sample with an extreme count outlier then the *p*-value
     and adjusted *p*-value will be set to :code:`NA`.  These outlier counts are
     detected by Cook's distance.  Customization of this outlier filtering and
     description of functionality for replacement of outlier counts and
     refitting is described :ref:`below<outlier>`.
   * If a row is filtered by automatic independent filtering, for having a low
     mean normalized count, then only the adjusted *p*-value will be set to
     :code:`NA`.  Description and customization of independent filtering is
     described :ref:`below<indfilt>`.

..
  Rich visualization and reporting of results
  -------------------------------------------

  **ReportingTools** An HTML report of the results with plots and
  sortable/filterable columns can be generated using the `ReportingTools
  <http://bioconductor.org/packages/ReportingTools>`_ package on a
  :class:`~DESeqDataSet.DESeqDataSet` that has been processed by the :func:`DESeq`
  function.  For a code example, see the *RNA-seq differential expression*
  vignette at the `ReportingTools
  <http://bioconductor.org/packages/ReportingTools>`_ page, or the manual page for
  the :meth:`publish` method for the :class:`~DESeqDataSet.DESeqDataSet` class.

  **regionReport** An HTML and PDF summary of the results with plots can also be
  generated using the `regionReport
  <http://bioconductor.org/packages/regionReport>`_ package. The
  :func:`DESeq2Report` function should be run on a
  :class:`~DESeqDataSet.DESeqDataSet` that has been processed by the :func:`DESeq`
  function.  For more details see the manual page for :func:`DESeq2Report` and an
  example vignette in the `regionReport
  <http://bioconductor.org/packages/regionReport>`_ package.

  **Glimma** Interactive visualization of DESeq2 output, including MA-plots (also
  called MD-plot) can be generated using the `Glimma
  <http://bioconductor.org/packages/Glimma>`_ package. See the manual page for
  *glMDPlot.DESeqResults*.

  **pcaExplorer** Interactive visualization of DESeq2 output, including PCA plots,
  boxplots of counts and other useful summaries can be generated using the
  `pcaExplorer <http://bioconductor.org/packages/pcaExplorer>`_ package. See the
  *Launching the application* section of the package vignette.

  **iSEE** Provides functions for creating an interactive Shiny-based graphical
  user interface for exploring data stored in SummarizedExperiment objects,
  including row- and column-level metadata. Particular attention is given to
  single-cell data in a SingleCellExperiment object with visualization of
  dimensionality reduction results.  `iSEE
  <https://bioconductor.org/packages/iSEE>`_ is on Bioconductor.  An example
  wrapper function for converting a :class:`~DESeqDataSet.DESeqDataSet` to a
  SingleCellExperiment object for use with *iSEE* can be found at the following
  gist, written by Federico Marini:

  * <https://gist.github.com/federicomarini/4a543eebc7e7091d9169111f76d59de1>

  **DEvis** DEvis is a powerful, integrated solution for the analysis of
  differential expression data. This package includes an array of tools for
  manipulating and aggregating data, as well as a wide range of customizable
  visualizations, and project management functionality that simplify RNA-Seq
  analysis and provide a variety of ways of exploring and analyzing data.  *DEvis*
  can be found on `CRAN <https://cran.r-project.org/package=DEVis>` and `GitHub
  <https://github.com/price0416/DEvis>`.


Exporting results to CSV files
------------------------------

A plain-text file of the results can be exported using the method
:meth:`.DESeqResults.to_csv`. We suggest using a descriptive file name
indicating the variable and level which were tested:

.. repl::
   resOrdered.to_csv("condition_treated_results.csv")

Exporting only the results which pass an adjusted *p*-value threshold can be
accomplished by subsetting the results:

.. repl::
   resSig = resOrdered[resOrdered.padj < 0.1]
   resSig.head()


Multi-factor designs
====================

Experiments with more than one factor influencing the counts can be analyzed
using design formula that include the additional variables.  In fact, DESeq2 can
analyze any possible experimental design that can be expressed with fixed
effects terms (multiple factors, designs with interactions, designs with
continuous variables, splines, and so on are all possible).

By adding variables to the design, one can control for additional variation in
the counts. For example, if the condition samples are balanced across
experimental batches, by including the :code:`batch` factor to the design, one
can increase the sensitivity for finding differences due to :code:`condition`.
There are multiple ways to analyze experiments when the additional variables are
of interest and not just controlling factors (see :ref:`section on
interactions<interactions>`).

**Experiments with many samples**: in experiments with many samples (e.g. 50,
100, etc.) it is highly likely that there will be technical variation affecting
the observed counts. Failing to model this additional technical variation will
lead to spurious results. Many methods exist that can be used to model technical
variation, which can be easily included in the DESeq2 design to control for
technical variation which estimating effects of interest. See the `RNA-seq
workflow <http://www.bioconductor.org/help/workflows/rnaseqGene>`_ for examples
of using RUV or SVA in combination with DESeq2.  For more details on why it is
important to control for technical variation in large sample experiments, see
the following `thread
<https://twitter.com/mikelove/status/1513468597288452097>`_, also archived `here
<https://htmlpreview.github.io/?https://github.com/frederikziebell/science_tweetorials/blob/master/DESeq2_many_samples.html>`_
by Frederik Ziebell.

The data in the :doc:`/pasilla` module have a condition of interest (the column
:code:`condition`), as well as information on the type of sequencing which was
performed (the column :code:`type`), as we can see below:

.. repl::
   dds.obs

We create a copy of the :class:`~DESeqDataSet.DESeqDataSet`, so that we can
rerun the analysis using a multi-factor design:

.. repl::
   ddsMF = dds.copy()

We change the categories of :code:`type` so it only contains letters.  Be
careful when changing level names to use the same order as the current
categories:

.. repl::
   import re
   dds.obs["type"].cat.categories
   dds.obs["type"].cat.rename_categories([re.sub("-.*", "", c) for c in dds.obs["type"].cat.categories])
   dds.obs["type"].dtype.categories

We can account for the different types of sequencing, and get a clearer picture
of the differences attributable to the treatment.  As :code:`condition` is the
variable of interest, we put it at the end of the formula. Thus the
:meth:`.DESeqDataSet.results` function will by default pull the
:code:`condition` results unless :code:`contrast` or :code:`name` arguments are
specified.

Then we can re-run :func:`DESeq`:

.. repl::
   ddsMF.design = "~ type + condition"
   ddsMF = DESeq(ddsMF)

Again, we access the results using the :meth:`.DESeqDataSet.results` function:

.. repl::
   resMF = ddsMF.results()
   resMF.head()

It is also possible to retrieve the log2 fold changes, *p*-values and adjusted
*p*-values of variables other than the last one in the design.  While in this
case, :code:`type` is not biologically interesting as it indicates differences
across sequencing protocols, for other hypothetical designs, such as
:code:`~genotype + condition + genotype:condition`, we may actually be
interested in the difference in baseline expression across genotype, which is
not the last variable in the design.

In any case, the :code:`contrast` argument of the function
:meth:`.DESeqDataSet.results` takes a string list of length three: the name of
the variable, the name of the factor level for the numerator of the log2 ratio,
and the name of the factor level for the denominator.  The :code:`contrast`
argument can also take other forms, as described in the help page for
:meth:`.DESeqDataSet.results` and :ref:`below<contrasts>`:

.. repl::
   resMFType = ddsMF.results(contrast=("type", "single", "paired"))
   resMFType.head()

If the variable is continuous or an interaction term (see :ref:`section on
interactions<interactions>`) then the results can be extracted using the
:code:`name` argument to :meth:`.DESeqDataSet.results`, where the name is one of
elements returned by :code:`dds.resultsNames()`.

.. _transform:

..
  Data transformations and visualization
  ======================================

  Count data transformations
  --------------------------

  In order to test for differential expression, we operate on raw counts and use
  discrete distributions as described in the previous section on differential
  expression.  However for other downstream analyses -- *e.g.* for visualization
  or clustering -- it might be useful to work with transformed versions of the
  count data.

  Maybe the most obvious choice of transformation is the logarithm.  Since count
  values for a gene can be zero in some conditions (and non-zero in others), some
  advocate the use of *pseudocounts*, i.e. transformations of the form :math:`y =
  \log(n + n_0)`, where :math:`n` represents the count values and :math:`n_0` is a
  positive constant.

  In this section, we discuss two alternative approaches that offer more
  theoretical justification and a rational way of choosing parameters equivalent
  to :math:`n_0` above.  One makes use of the concept of variance stabilizing
  transformations (VST) [Tibshirani1988]_ [sagmb2003]_ [Anders2010]_, and the
  other is the *regularized logarithm* or *rlog*, which incorporates a prior on
  the sample differences [Love2014]_.  Both transformations produce transformed
  data on the log2 scale which has been normalized with respect to library size or
  other normalization factors.

  The point of these two transformations, the VST and the rlog, is to remove the
  dependence of the variance on the mean, particularly the high variance of the
  logarithm of count data when the mean is low. Both VST and rlog use the
  experiment-wide trend of variance over mean, in order to transform the data to
  remove the experiment-wide trend. Note that we do not require or desire that all
  the genes have *exactly* the same variance after transformation. Indeed, in a
  figure below, you will see that after the transformations the genes with the
  same mean do not have exactly the same standard deviations, but that the
  experiment-wide trend has flattened. It is those genes with row variance above
  the trend which will allow us to cluster samples into interesting groups.

  .. admonition:: Note on running time:

     If you have many samples (*e.g.* 100s), the :func:`rlog` function might take
     too long, and so the :func:`vst` function will be a faster choice.  The rlog
     and VST have similar properties, but the rlog requires fitting a shrinkage
     term for each sample and each gene which takes time. See the DESeq2 paper for
     more discussion on the differences [Love2014]_.

  Blind dispersion estimation
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^

  The two functions, :func:`vst` and :func:`rlog`: have an argument :code:`blind`,
  for whether the transformation should be blind to the sample information
  specified by the design formula. When :code:`blind` equals :code:`True` (the
  default), the functions will re-estimate the dispersions using only an
  intercept.  This setting should be used in order to compare samples in a manner
  wholly unbiased by the information about experimental groups, for example to
  perform sample QA (quality assurance) as demonstrated below.

  However, blind dispersion estimation is not the appropriate choice if one
  expects that many or the majority of genes will have large differences in counts
  which are explainable by the experimental design, and one wishes to transform
  the data for downstream analysis. In this case, using blind dispersion
  estimation will lead to large estimates of dispersion, as it attributes
  differences due to experimental design as unwanted *noise*, and will result in
  overly shrinking the transformed values towards each other.  By setting
  :code:`blind` to :code:`False`, the dispersions already estimated will be used
  to perform transformations, or if not present, they will be estimated using the
  current design formula. Note that only the fitted dispersion estimates from
  mean-dispersion trend line are used in the transformation (the global dependence
  of dispersion on mean for the entire experiment).  So setting :code:`blind` to
  :code:`False` is still for the most part not using the information about which
  samples were in which experimental group in applying the transformation.

  Extracting transformed values
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  These transformation functions return an object of class :class:`DESeqTransform`
  which is a subclass of :class:`AnnData`.  For ~20 samples, running on a newly
  created :class:`~DESeqDataSet.DESeqDataSet`, :func:`rlog` may take 30 seconds,
  while :func:`vst` takes less than 1 second.  The running times are shorter when
  using `blind=False` and if the function :func:`DESeq` has already been run,
  because then it is not necessary to re-estimate the dispersion values.  The
  matrix of normalized values is stored in the :attr:`X` attribute:

  >>> vsd = vst(dds, blind=FALSE)
  >>> rld = rlog(dds, blind=FALSE)
  >>> vsd.X.head(3)

  Variance stabilizing transformation
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  Above, we used a parametric fit for the dispersion. In this case, the
  closed-form expression for the variance stabilizing transformation is used by
  the :func:`vst` function. If a local fit is used (option
  :code:`fitType="locfit"` to :meth:`.DESeqDataSet.estimateDispersions`) a
  numerical integration is used instead. The transformed data should be
  approximated variance stabilized and also includes correction for size factors
  or normalization factors. The transformed data is on the log2 scale for large
  counts.

  Regularized log transformation
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  The function :func:`rlog`, stands for *regularized log*, transforming the
  original count data to the log2 scale by fitting a model with a term for each
  sample and a prior distribution on the coefficients which is estimated from the
  data. This is the same kind of shrinkage (sometimes referred to as
  regularization, or moderation) of log fold changes used by :func:`DESeq` and
  :func:`nbinomWaldTest`. The resulting data contains elements defined as:

  .. math::
     \log_2(q_{ij}) = \beta_{i0} + \beta_{ij}

  where :math:`q_{ij}` is a parameter proportional to the expected true
  concentration of fragments for gene :math:`i` and sample :math:`j` (see formula
  :ref:`below<theory>`), :math:`\beta_{i0}` is an intercept which does not undergo
  shrinkage, and :math:`\beta_{ij}` is the sample-specific effect which is shrunk
  toward zero based on the dispersion-mean trend over the entire dataset. The
  trend typically captures high dispersions for low counts, and therefore these
  genes exhibit higher shrinkage from the :func:`rlog`.

  Note that, as :math:`q_{ij}` represents the part of the mean value
  :math:`\mu_{ij}` after the size factor :math:`s_j` has been divided out, it is
  clear that the rlog transformation inherently accounts for differences in
  sequencing depth. Without priors, this design matrix would lead to a non-unique
  solution, however the addition of a prior on non-intercept betas allows for a
  unique solution to be found.

  Effects of transformations on the variance
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  The figure below plots the standard deviation of the transformed data, across
  samples, against the mean, using the shifted logarithm transformation, the
  regularized log transformation and the variance stabilizing transformation.  The
  shifted logarithm has elevated standard deviation in the lower count range, and
  the regularized log to a lesser extent, while for the variance stabilized data
  the standard deviation is roughly constant along the whole dynamic range.

  Note that the vertical axis in such plots is the square root of the variance
  over all samples, so including the variance due to the experimental conditions.
  While a flat curve of the square root of variance over the mean may seem like
  the goal of such transformations, this may be unreasonable in the case of
  datasets with many true differences due to the experimental conditions.

  ..
    ```{r meansd}
    # this gives log2(n + 1)
    ntd <- normTransform(dds)
    library("vsn")
    meanSdPlot(assay(ntd))
    meanSdPlot(assay(vsd))
    meanSdPlot(assay(rld))
    ```

  Data quality assessment by sample clustering and visualization
  --------------------------------------------------------------

  Data quality assessment and quality control (*i.e.* the removal of
  insufficiently good data) are essential steps of any data analysis. These steps
  should typically be performed very early in the analysis of a new data set,
  preceding or in parallel to the differential expression testing.

  We define the term *quality* as *fitness for purpose*.  Our purpose is the
  detection of differentially expressed genes, and we are looking in particular
  for samples whose experimental treatment suffered from an anormality that
  renders the data points obtained from these particular samples detrimental to
  our purpose.

  Heatmap of the count matrix
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^

  To explore a count matrix, it is often instructive to look at it as a heatmap.
  Below we show how to produce such a heatmap for various transformations of the
  data.

  TODO use clustermap instead of pheatmap

  .. repl::

     from seaborn import clustermap
     select = np.sort(dds.counts(normalized=True).mean(axis=0))[::-1][:20]
     df = dds.obs[["condition", "type"]]

  ..
    ```{r heatmap}
    library("pheatmap")
    select <- order(rowMeans(counts(dds,normalized=TRUE)),
                    decreasing=TRUE)[1:20]
    df <- as.data.frame(colData(dds)[,c("condition","type")])
    pheatmap(assay(ntd)[select,], cluster_rows=FALSE, show_rownames=FALSE,
             cluster_cols=FALSE, annotation_col=df)
    pheatmap(assay(vsd)[select,], cluster_rows=FALSE, show_rownames=FALSE,
             cluster_cols=FALSE, annotation_col=df)
    pheatmap(assay(rld)[select,], cluster_rows=FALSE, show_rownames=FALSE,
             cluster_cols=FALSE, annotation_col=df)
    ```

  Heatmap of the sample-to-sample distances
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  Another use of the transformed data is sample clustering. Here, we apply the
  :func:`dist` function to the transpose of the transformed count matrix to get
  sample-to-sample distances::

    sampleDists = dist(vsd.layers.T)

  A heatmap of this distance matrix gives us an overview over similarities and
  dissimilarities between samples.  We have to provide a hierarchical clustering
  :code:`hc` to the heatmap function based on the sample distances, or else the
  heatmap function would calculate a clustering based on the distances between the
  rows/columns of the distance matrix.

  TODO

  ..
    ```{r figHeatmapSamples, fig.height=4, fig.width=6}
    library("RColorBrewer")
    sampleDistMatrix <- as.matrix(sampleDists)
    rownames(sampleDistMatrix) <- paste(vsd$condition, vsd$type, sep="-")
    colnames(sampleDistMatrix) <- NULL
    colors <- colorRampPalette( rev(brewer.pal(9, "Blues")) )(255)
    pheatmap(sampleDistMatrix,
             clustering_distance_rows=sampleDists,
             clustering_distance_cols=sampleDists,
             col=colors)
    ```

  Principal component plot of the samples
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  Related to the distance matrix is the PCA plot, which shows the samples in the
  2D plane spanned by their first two principal components. This type of plot is
  useful for visualizing the overall effect of experimental covariates and batch
  effects.

  TODO

  ..
    ```{r figPCA}
    plotPCA(vsd, intgroup=c("condition", "type"))
    ```

  It is also possible to customize the PCA plot using the
  *ggplot* function.

  TODO

  ..
    ```{r figPCA2}
    pcaData <- plotPCA(vsd, intgroup=c("condition", "type"), returnData=TRUE)
    percentVar <- round(100 * attr(pcaData, "percentVar"))
    ggplot(pcaData, aes(PC1, PC2, color=condition, shape=type)) +
      geom_point(size=3) +
      xlab(paste0("PC1: ",percentVar[1],"% variance")) +
      ylab(paste0("PC2: ",percentVar[2],"% variance")) +
      coord_fixed()
    ```

Variations to the standard workflow
===================================

Wald test individual steps
--------------------------

The function :func:`DESeq` runs the following functions in order:

>>> dds = dds.estimateSizeFactors()
>>> dds = dds.estimateDispersions(dds)
>>> dds = nbinomWaldTest(dds)

Control features for estimating size factors
--------------------------------------------

In some experiments, it may not be appropriate to assume that a minority of
features (genes) are affected greatly by the condition, such that the standard
median-ratio method for estimating the size factors will not provide correct
inference (the log fold changes for features that were truly un-changing will
not centered on zero). This is a difficult inference problem for any method, but
there is an important feature that can be used: the :code:`controlGenes`
argument of :meth:`.DESeqDataSet.estimateSizeFactors`. If there is any prior
information about features (genes) that should not be changing with respect to
the condition, providing this set of features to :code:`controlGenes` will
ensure that the log fold changes for these features will be centered around 0.
The paradigm then becomes:

>>> dds = dds.estimateSizeFactors(controlGenes=ctrlGenes)
>>> dds = DESeq(dds)

.. _contrasts:

Contrasts
---------

A contrast is a linear combination of estimated log2 fold changes, which can be
used to test if differences between groups are equal to zero.  The simplest use
case for contrasts is an experimental design containing a factor with three
levels, say A, B and C.  Contrasts enable the user to generate results for all 3
possible differences: log2 fold change of B vs A, of C vs A, and of C vs B.  The
:code:`contrast` argument of :meth:`.DESeqDataSet.results` function is used to
extract test results of log2 fold changes of interest, for example:

>>> dds.results(contrast=["condition","C","B"])

Log2 fold changes can also be added and subtracted by providing a list to the
:code:`contrast` argument which has two elements: the names of the log2 fold
changes to add, and the names of the log2 fold changes to subtract. The names
used in the list should come from :code:`dds.resultsNames()`.  Alternatively, a
numeric vector of the length of :code:`dds.resultsNames()` can be provided, for
manually specifying the linear combination of terms. A `tutorial
<https://github.com/tavareshugo/tutorial_DESeq2_contrasts>`_ describing the use
of numeric contrasts for DESeq2 explains a general approach to comparing across
groups of samples.  Demonstrations of the use of contrasts for various designs
can be found in the examples section of the help page
:meth:`.DESeqDataSet.results`.  The mathematical formula that is used to
generate the contrasts can be found :ref:`below<theory>`.

.. _interactions:

Interactions
------------

Interaction terms can be added to the design formula, in order to test, for
example, if the log2 fold change attributable to a given condition is
*different* based on another factor, for example if the condition effect differs
across genotype.

Preliminary remarks
^^^^^^^^^^^^^^^^^^^

Many users begin to add interaction terms to the design formula, when in fact a
much simpler approach would give all the results tables that are desired.  We
will explain this approach first, because it is much simpler to perform.  If the
comparisons of interest are, for example, the effect of a condition for
different sets of samples, a simpler approach than adding interaction terms
explicitly to the design formula is to perform the following steps:

  * combine the factors of interest into a single factor with all
    combinations of the original factors
  * change the design to include just this factor, *e.g.* :code:`"~ group"`

Using this design is similar to adding an interaction term, in that it models
multiple condition effects which can be easily extracted with
:meth:`.DESeqDataSet.results`.  Suppose we have two factors :code:`genotype`
(with values I, II, and III) and :code:`condition` (with values A and B), and we
want to extract the condition effect specifically for each genotype. We could
use the following approach to obtain, *e.g.* the condition effect for genotype
I:

>>> dds.obs["group"] = Factor([f"{g}{c}" for (g,c) in zip(dds.obs["genotype"], dds.obs["condition"])])
>>> dds.design = "~ group"
>>> dds = DESeq(dds)
>>> dds.resultsNames()
>>> dds.results(contrast=["group", "IB", "IA"])


Adding interactions to the design
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following two plots diagram genotype-specific condition effects, which could
be modeled with interaction terms by using a design of :code:`~genotype +
condition + genotype:condition`.

In the first plot (Gene 1), note that the condition effect is consistent across
genotypes. Although condition A has a different baseline for I, II, and III, the
condition effect is a log2 fold change of about 2 for each genotype.  Using a
model with an interaction term :code:`genotype:condition`, the interaction terms
for genotype II and genotype III will be nearly 0.

Here, the y-axis represents :math:`\log(n+1)`, and each group has 20 samples
(black dots). A red line connects the mean of the groups within each genotype.

.. repl-quiet::
   from inmoose.utils import rnbinom
   import seaborn as sns

   npg = 20
   mu = 2**np.array([8,10,9,11,10,12])
   cond = np.repeat([np.repeat(["A","B"], npg)], 3, axis=0).flatten()
   geno = np.repeat(["I","II","III"], 2*npg)
   #table(cond, geno)
   counts = rnbinom(6*npg, mu = np.repeat(mu, npg), size=1/.01)
   d = pd.DataFrame({"log2c": np.log2(counts+1), "cond": cond, "geno": geno})

   def plotit(d, title):
     g = sns.FacetGrid(d, col="geno")
     g.map(sns.stripplot, "cond", "log2c", color="black")
     g.map(sns.pointplot, "cond", "log2c", color="red")
     g.fig.subplots_adjust(top=0.9)
     g.fig.suptitle(title)
     plt.show()

   plotit(d, "Gene 1")

In the second plot (Gene 2), we can see that the condition effect is not
consistent across genotype. Here the main condition effect (the effect for the
reference genotype I) is again 2. However, this time the interaction terms will
be around 1 for genotype II and -4 for genotype III. This is because the
condition effect is higher by 1 for genotype II compared to genotype I, and
lower by 4 for genotype III compared to genotype I.  The condition effect for
genotype II (or III) is obtained by adding the main condition effect and the
interaction term for that genotype.  Such a plot can be made using the
:func:`plotCounts` function as shown above.

.. repl-quiet::
   mu[3] = 2**12
   mu[5] = 2**8
   counts = rnbinom(6*npg, mu=np.repeat(mu, npg), size=1/.01)
   d2 = pd.DataFrame({"log2c": np.log2(counts+1), "cond": cond, "geno": geno})
   plotit(d2, "Gene 2")

Now we will continue to explain the use of interactions in order to test for
*differences* in condition effects. We continue with the example of condition
effects across three genotypes (I, II, and III).

The key point to remember about designs with interaction terms is that, unlike
for a design :code:`~genotype + condition`, where the condition effect
represents the *overall* effect controlling for differences due to genotype, by
adding :code:`genotype:condition`, the main condition effect only represents the
effect of condition for the *reference level* of genotype (I, or whichever level
was defined by the user as the reference level). The interaction terms
:code:`genotypeII.conditionB` and :code:`genotypeIII.conditionB` give the
*difference* between the condition effect for a given genotype and the condition
effect for the reference genotype.

This genotype-condition interaction example is examined in further detail in
Example 3 in the help page for :meth:`.DESeqDataSet.results`. In particular, we
show how to test for differences in the condition effect across genotype, and we
show how to obtain the condition effect for non-reference genotypes.

Time-series experiments
-----------------------

There are a number of ways to analyze time-series experiments, depending on the
biological question of interest. In order to test for any differences over
multiple time points, once can use a design including the time factor, and then
test using the likelihood ratio test as described in the following section,
where the time factor is removed in the reduced formula. For a control and
treatment time series, one can use a design formula containing the condition
factor, the time factor, and the interaction of the two. In this case, using the
likelihood ratio test with a reduced model which does not contain the
interaction terms will test whether the condition induces a change in gene
expression at any time point after the reference level time point (time 0). An
example of the later analysis is provided in the `RNA-seq workflow
<http://www.bioconductor.org/help/workflows/rnaseqGene>`_.

Likelihood ratio test
---------------------

DESeq2 offers two kinds of hypothesis tests: the Wald test, where we use the
estimated standard error of a log2 fold change to test if it is equal to zero,
and the likelihood ratio test (LRT). The LRT examines two models for the counts,
a *full* model with a certain number of terms and a *reduced* model, in which
some of the terms of the *full* model are removed. The test determines if the
increased likelihood of the data using the extra terms in the *full* model is
more than expected if those extra terms are truly zero.

The LRT is therefore useful for testing multiple terms at once, for example
testing 3 or more levels of a factor at once, or all interactions between two
variables.  The LRT for count data is conceptually similar to an analysis of
variance (ANOVA) calculation in linear regression, except that in the case of
the Negative Binomial GLM, we use an analysis of deviance (ANODEV), where the
*deviance* captures the difference in likelihood between a full and a reduced
model.

The likelihood ratio test can be performed by specifying :code:`test="LRT"` when
using the :func:`DESeq` function, and providing a reduced design formula, *e.g.*
one in which a number of terms from :code:`dds.design` are removed.  The degrees
of freedom for the test is obtained from the difference between the number of
parameters in the two models.  A simple likelihood ratio test, if the full
design was :code:`~condition` would look like:

>>> dds = DESeq(dds, test="LRT", reduced="~1")
>>> res = dds.results()

If the full design contained other variables, such as a batch variable, *e.g.*
:code:`~batch + condition` then the likelihood ratio test would look like:

>>> dds = DESeq(dds, test="LRT", reduced="~batch")
>>> res = dds.results()

.. _moreshrink:

Extended section on shrinkage estimators
----------------------------------------

Here we extend the :ref:`discussion of shrinkage estimators<shrink>`.  Below is
a summary table of differences between methods available in :func:`lfcShrink`
via the :code:`type` argument (and for further technical reference on use of
arguments please the documentation of :func:`lfcShrink`):

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - method
     - :code:`apeglm` [Zhu2018]_
     - :code:`ashr` [Stephens2016]_
     - :code:`normal` [Love2014]_
   * - Good for ranking by LFC
     - ✓
     - ✓
     - ✓
   * - Preserves size of large LFC
     - ✓
     - ✓
     -
   * - Can compute *s*-values [Stephens2016]_
     - ✓
     - ✓
     -
   * - Allows use of :code:`coef`
     - ✓
     - ✓
     - ✓
   * - Allows use of :code:`lfcThreshold`
     - ✓
     - ✓
     - ✓
   * - Allows use of :code:`contrast`
     - ✓
     - ✓
     -
   * - Can shrink interaction terms
     - ✓
     - ✓
     -

Beginning with the first row, all shrinkage methods provided by DESeq2 are good
for ranking genes by "effect size", that is the log2 fold change (LFC) across
groups, or associated with an interaction term. It is useful to contrast ranking
by effect size with ranking by a *p*-value or adjusted *p*-value associated with
a null hypothesis: while increasing the number of samples will tend to decrease
the associated *p*-value for a gene that is differentially expressed, the
estimated effect size or LFC becomes more precise. Also, a gene can have a small
*p*-value although the change in expression is not great, as long as the
standard error associated with the estimated LFC is small.

The next two rows point out that :code:`apeglm` and :code:`ashr` shrinkage
methods help to preserve the size of large LFC, and can be used to compute
*s-values*. These properties are related. As noted in the :ref:`previous
section<shrink>`, the original DESeq2 shrinkage estimator used a Normal
distribution, with a scale that adapts to the spread of the observed LFCs.
Because the tails of the Normal distribution become thin relatively quickly, it
was important when we designed the method that the prior scaling is sensitive to
the very largest observed LFCs. As you can read in the DESeq2 paper, under the
section, "*Empirical prior estimate*", we used the top 5% of the LFCs by
absolute value to set the scale of the Normal prior (we later added weighting
the quantile by precision). :code:`ashr`, published in 2016, and :code:`apeglm`
use wide-tailed priors to avoid shrinking large LFCs. While a typical RNA-seq
experiment may have many LFCs between -1 and 1, we might consider a LFC of >4 to
be very large, as they represent 16-fold increases or decreases in expression.
:code:`ashr` and :code:`apeglm` can adapt to the scale of the entirety of LFCs,
while not over-shrinking the few largest LFCs. The potential for over-shrinking
LFC is also why DESeq2's shrinkage estimator is not recommended for designs with
interaction terms.

What are *s-values*? This quantity proposed by [Stephens2016]_ gives the
estimated rate of *false sign* among genes with equal or smaller *s*-value.
[Stephens2016]_ points out they are analogous to the *q*-value of [Storey2003]_.
The *s*-value has a desirable property relative to the adjusted *p*-value or
*q*-value, in that it does not require supposing there be a set of null genes
with LFC = 0 (the most commonly used null hypothesis). Therefore, it can be
benchmarked by comparing estimated LFC and *s*-value to the "true LFC" in a
setting where this can be reasonably defined. For these estimated probabilities
to be accurate, the scale of the prior needs to match the scale of the
distribution of effect sizes, and so the original DESeq2 shrinkage method is not
really compatible with computing *s*-values.

The last four rows explain differences in whether coefficients or contrasts can
have shrinkage applied by the various methods. All three methods can use
:code:`coef` with either the name or numeric index from
:code:`dds.resultsNames()` to specify which coefficient to shrink.  All three
methods allow for a positive :code:`lfcThreshold` to be specified, in which
case, they will return *p*-values and adjusted *p*-values or *s*-values for the
LFC being greater in absolute value than the threshold (see :ref:`this
section<thresh>` for :code:`normal`).  For :code:`apeglm` and :code:`ashr`,
setting a threshold means that the *s*-values will give the "false sign or
small" rate (FSOS) among genes with equal or small *s*-value.  We found FSOS to
be a useful description for when the LFC is either the wrong sign or less than
the threshold distance from 0.

TODO

..
  ```{r apeThresh}
  resApeT <- lfcShrink(dds, coef=2, type="apeglm", lfcThreshold=1)
  plotMA(resApeT, ylim=c(-3,3), cex=.8)
  abline(h=c(-1,1), col="dodgerblue", lwd=2)
  ```

  ```{r ashThresh}
  resAshT <- lfcShrink(dds, coef=2, type="ashr", lfcThreshold=1)
  plotMA(resAshT, ylim=c(-3,3), cex=.8)
  abline(h=c(-1,1), col="dodgerblue", lwd=2)
  ```

Finally, :code:`normal` and :code:`ashr` can be used with arbitrary specified
:code:`contrast` because :code:`normal` shrinks multiple coefficients
simultaneously (:code:`apeglm` does not), and because :code:`ashr` does not
estimate a vector of coefficients but models estimated coefficients and their
standard errors from upstream methods (here, DESeq2's MLE).  Although
:code:`apeglm` cannot be used with :code:`contrast`, we note that many designs
can be easily rearranged such that what was a contrast becomes its own
coefficient. In this case, the dispersion does not have to be estimated again,
as the designs are equivalent, up to the meaning of the coefficients. Instead,
one need only run :code:`nbinomWaldTest` to re-estimate MLE coefficients --
these are necessary for :code:`apeglm` -- and then run :code:`lfcShrink`
specifying the coefficient of interest in `dds.resultsNames()`.

We give some examples below of producing equivalent designs for use with
:code:`coef`. We show how the coefficients change with :code:`patsy.dmatrix`,
but the user would, for example, either change the levels of `dds.obs.condition`
or replace the :attr:`.DESeqDataSet.design`, then run :func:`nbinomWaldTest`
followed by :func:`lfcShrink`.

Three groups:

.. repl::
   condition = Factor(["A", "A", "B", "B", "C", "C"])
   patsy.dmatrix("~condition")
   # to compare C vs B, make B the reference level,
   # and select the last coefficient
   condition = condition.reorder_categories(["B", "A", "C"])
   patsy.dmatrix("~condition")

Three groups, compare condition effects:

.. repl::
   grp = Factor([1,1,1,1,2,2,2,2,3,3,3,3])
   cnd = Factor(["A","A","B","B","A","A","B","B","A","A","B","B"])
   patsy.dmatrix("~ grp + cnd + grp:cnd")
   # to compare condition effect in group 3 vs 2,
   # make group 2 the reference level,
   # and select the last coefficient
   grp = grp.reorder_categories([2,1,3])
   patsy.dmatrix("~ grp + cnd + grp:cnd")

Two groups, two individuals per group, compare within-individual condition
effects:

.. repl::
   grp = Factor([1,1,1,1,2,2,2,2])
   ind = Factor([1,1,2,2,1,1,2,2])
   cnd = Factor(["A","B","A","B","A","B","A","B"])
   patsy.dmatrix("~ grp + grp:ind + grp:cnd")
   # to compare condition effect across group,
   # add a main effect for 'cnd',
   # and select the last coefficient
   patsy.dmatrix("~ grp + cnd + grp:ind + grp:cnd")

.. _singlecell:

Recommendations for single-cell analysis
----------------------------------------

The DESeq2 developers and collaborating groups have published recommendations
for the best use of DESeq2 for single-cell datasets, which have been described
first in [Berge2018]_. Default values for DESeq2 were designed for bulk data and
will not be appropriate for single-cell datasets. These settings and additional
improvements have also been tested subsequently and published in [Zhu2018]_ and
[AhlmannEltze2020]_.

* Use :code:`test="LRT"` for significance testing when working with single-cell
  data, over the Wald test. This has been observed across multiple single-cell
  benchmarks.
* Set the following :func:`DESeq` arguments to these values: :code:`useT=TRUE`,
  :code:`minmu=1e-6`, and :code:`minReplicatesForReplace=Inf`.  The default
  setting of :code:`minmu` was benchmarked on bulk RNA-seq and is not
  appropriate for single cell data when the expected count is often much less
  than 1.
* The default size factors are not optimal for single cell count matrices,
  instead consider setting :attr:`.DESeqDataSet.sizeFactors` from
  :code:`scran::computeSumFactors`.
* One important concern for single-cell data analysis is the size of the
  datasets and associated processing time. To address the speed concerns, DESeq2
  provides an interface to `glmGamPoi
  <https://bioconductor.org/packages/glmGamPoi/>`_, which implements faster
  dispersion and parameter estimation routines for single-cell data
  [AhlmannEltze2020]_. To use this feature, set :code:`fitType = "glmGamPoi"`.
  Alternatively, one can use *glmGamPoi* as a standalone package.  This
  provides the additional option to process data on-disk if the full dataset
  does not fit in memory, a quasi-likelihood framework for differential testing,
  and the ability to form pseudobulk samples (more details how to use
  *glmGamPoi* are in its `README <https://github.com/const-ae/glmGamPoi>`_).

Optionally, one can consider using the `zinbwave
<https://bioconductor.org/packages/zinbwave>`_ package to directly model the
zero inflation of the counts, and take account of these in the DESeq2 model.
This allows for the DESeq2 inference to apply to the part of the data which is
not due to zero inflation. Not all single cell datasets exhibit zero inflation,
and instead may just reflect low conditional estimated counts (conditional on
cell type or cell state). There is example code for combining *zinbwave* and
*DESeq2* package functions in the *zinbwave* vignette. We also have an example
of ZINB-WaVE + DESeq2 integration using the `splatter
<https://bioconductor.org/packages/splatter>`_ package for simulation at the
`zinbwave-deseq2 <https://github.com/mikelove/zinbwave-deseq2>`_ GitHub
repository.

.. _outlier:

Approach to count outliers
--------------------------

RNA-seq data sometimes contain isolated instances of very large counts that are
apparently unrelated to the experimental or study design, and which may be
considered outliers. There are many reasons why outliers can arise, including
rare technical or experimental artifacts, read mapping problems in the case of
genetically differing samples, and genuine, but rare biological events. In many
cases, users appear primarily interested in genes that show a consistent
behavior, and this is the reason why by default, genes that are affected by such
outliers are set aside by DESeq2, or if there are sufficient samples, outlier
counts are replaced for model fitting.  These two behaviors are described below.

The :func:`DESeq` function calculates, for every gene and for every sample, a
diagnostic test for outliers called *Cook's distance*. Cook's distance is a
measure of how much a single sample is influencing the fitted coefficients for a
gene, and a large value of Cook's distance is intended to indicate an outlier
count.  The Cook's distances are stored as a matrix available in
:code:`dds.layers["cooks"]`.

The :meth:`.DESeqDataSet.results` function automatically flags genes which
contain a Cook's distance above a cutoff for samples which have 3 or more
replicates.  The *p*-values and adjusted *p*-values for these genes are set to
:code:`NA`.  At least 3 replicates are required for flagging, as it is difficult
to judge which sample might be an outlier with only 2 replicates.  This
filtering can be turned off with :code:`dds.results(cooksCutoff=FALSE)`.

With many degrees of freedom -- *i.e.* many more samples than number of
parameters to be estimated -- it is undesirable to remove entire genes from the
analysis just because their data include a single count outlier. When there are
7 or more replicates for a given sample, the :func:`DESeq` function will
automatically replace counts with large Cook's distance with the trimmed mean
over all samples, scaled up by the size factor or normalization factor for that
sample. This approach is conservative, it will not lead to false positives, as
it replaces the outlier value with the value predicted by the null hypothesis.
This outlier replacement only occurs when there are 7 or more replicates, and
can be turned off with :code:`DESeq(dds, minReplicatesForReplace=Inf)`.

The default Cook's distance cutoff for the two behaviors described above depends
on the sample size and number of parameters to be estimated. The default is to
use the 99% quantile of the :math:`F(p,m-p)` distribution (with :math:`p` the
number of parameters including the intercept and :math:`m` the number of
samples).  The default for gene flagging can be modified using the
:code:`cooksCutoff` argument to the :meth:`.DESeqDataSet.results` function.  For
outlier replacement, :func:`DESeq` preserves the original counts in
:code:`dds.counts()` saving the replacement counts as a matrix named
:code:`"replaceCounts"` in :code:`dds.layers`.  Note that with continuous
variables in the design, outlier detection and replacement is not automatically
performed, as our current methods involve a robust estimation of within-group
variance which does not extend easily to continuous covariates. However, users
can examine the Cook's distances in :code:`dds.layers["cooks"]`, in order to
perform manual visualization and filtering if necessary.

.. note::
   If there are very many outliers (*e.g.* many hundreds or thousands) reported
   by :code:`res.summary()`, one might consider further exploration to see if a
   single sample or a few samples should be removed due to low quality.  The
   automatic outlier filtering/replacement is most useful in situations which
   the number of outliers is limited. When there are thousands of reported
   outliers, it might make more sense to turn off the outlier
   filtering/replacement (:func:`DESeq` with
   :code:`minReplicatesForReplace=np.inf` and :meth:`.DESeqDataSet.results` with
   :code:`cooksCutoff=FALSE`) and perform manual inspection:

   - first it would be advantageous to make a PCA plot as described above to
     spot individual sample outliers;
   - second, one can make a boxplot of the Cook's distances to see if one sample
     is consistently higher than others (here this is not the case):

   .. repl::
      sns.boxplot(pd.DataFrame(np.log10(dds.layers["cooks"].T),
                               columns=dds.obs_names))
      plt.xticks(rotation=25)
      plt.show()

Dispersion plot and fitting alternatives
----------------------------------------

Plotting the dispersion estimates is a useful diagnostic. The dispersion plot
below is typical, with the final estimates shrunk from the gene-wise estimates
towards the fitted estimates. Some gene-wise estimates are flagged as outliers
and not shrunk towards the fitted value, (this outlier detection is described in
the manual page for :func:`estimateDispersionsMAP`).  The amount
of shrinkage can be more or less than seen here, depending on the sample size,
the number of coefficients, the row mean and the variability of the gene-wise
estimates.

.. repl::
   dds.plotDispEsts()

Local or mean dispersion fit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A local smoothed dispersion fit is automatically substituted in the case that
the parametric curve does not fit the observed dispersion mean relationship.
This can be prespecified by providing the argument :code:`fitType="local"` to
either :func:`DESeq` or :meth:`.DESeqDataSet.estimateDispersions`.
Additionally, using the mean of gene-wise disperion estimates as the fitted
value can be specified by providing the argument :code:`fitType="mean"`.

Supply a custom dispersion fit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Any fitted values can be provided during dispersion estimation, using the
lower-level functions described in the manual page for
:func:`estimateDispersionsGeneEst`. In the code chunk below, we store the
gene-wise estimates which were already calculated and saved in the metadata
column :code:`dispGeneEst`. Then we calculate the median value of the dispersion
estimates above a threshold, and save these values as the fitted dispersions,
using the replacement function for :attr:`.DESeqDataSet.dispersionFunction`. In
the last line, the function :func:`estimateDispersionsMAP`, uses the fitted
dispersions to generate maximum *a posteriori* (MAP) estimates of dispersion:

.. repl::
   ddsCustom = dds.copy()
   useForMedian = ddsCustom.var["dispGeneEst"] > 1e-7
   medianDisp = np.nanmedian(ddsCustom.var["dispGeneEst"][useForMedian])
   ddsCustom.setDispFunction(lambda mu: medianDisp)
   ddsCustom = estimateDispersionsMAP(ddsCustom)

.. _indfilt:

Independent filtering of results
--------------------------------

The :meth:`.DESeqDataSet.results` function of the DESeq2 package performs
independent filtering by default using the mean of normalized counts as a filter
statistic.  A threshold on the filter statistic is found which optimizes the
number of adjusted *p*-values lower than a significance level :code:`alpha` (we
use the standard variable name for significance level, though it is unrelated to
the dispersion parameter :math:`\alpha`).  The theory behind independent
filtering is discussed in greater detail :ref:`below<indfilttheory>`. The
adjusted *p*-values for the genes which do not pass the filter threshold are set
to :code:`NA`.

The default independent filtering is performed using the
:func:`~.results.filtered_p` function of the `genefilter
<http://bioconductor.org/packages/genefilter>`_ package, and all of the
arguments of :func:`~.results.filtered_p` can be passed to the
:meth:`.DESeqDataSet.results` function.  The filter threshold value and the
number of rejections at each quantile of the filter statistic are available as
metadata of the object returned by :meth:`.DESeqDataSet.results`.

For example, we can visualize the optimization by plotting the
:code:`filterNumRej` attribute of the results object. The
:meth:`.DESeqDataSet.results` function maximizes the number of rejections
(adjusted *p*-value less than a significance level), over the quantiles of a
filter statistic (the mean of normalized counts). The threshold chosen (vertical
line) is the lowest quantile of the filter for which the number of rejections is
within 1 residual standard deviation to the peak of a curve fit to the number of
rejections over the filter quantiles:

.. repl::
   res.alpha
   res.filterThreshold
   plt.plot(res.filterNumRej["theta"],
            res.filterNumRej["numRej"],
            'o', color="black")
   plt.xlabel("quantiles of filter")
   plt.ylabel("number of rejections")
   plt.plot(res.lo_fit[:,0], res.lo_fit[:,1], color="red")
   plt.axvline(x=res.filterTheta, color="black")
   plt.show()

Independent filtering can be turned off by setting :code:`independentFiltering`
to :code:`FALSE`.

.. repl::
   resNoFilt = dds.results(independentFiltering=False)
   df = pd.DataFrame({"filtering": (res.padj < .1),
                      "noFiltering": (resNoFilt.padj < .1)})
   df.groupby(["filtering", "noFiltering"]).size()


.. _thresh:

Tests of log2 fold change above or below a threshold
----------------------------------------------------

It is also possible to provide thresholds for constructing Wald tests of
significance. Two arguments to the :meth:`.DESeqDataSet.results` function allow
for threshold-based Wald tests: :code:`lfcThreshold`, which takes a numeric of a
non-negative threshold value, and :code:`altHypothesis`, which specifies the
kind of test.  Note that the *alternative hypothesis* is specified by the user,
*i.e.* those genes which the user is interested in finding, and the test
provides *p*-values for the null hypothesis, the complement of the set defined
by the alternative. The :code:`altHypothesis` argument can take one of the
following four values, where :math:`\beta` is the log2 fold change specified by
the :code:`name` argument, and :math:`x` is the :code:`lfcThreshold`.

* `greaterAbs` - :math:`|\beta| > x` - tests are two-tailed
* `lessAbs` - :math:`|\beta| < x` - *p*-values are the maximum of the upper and lower tests
* `greater` - :math:`\beta > x`
* `less` - :math:`\beta < -x`

The four possible values of :code:`altHypothesis` are demonstrated in the
following code and visually by MA-plots in the following figures.

.. repl::
   ylim = [-2.5, 2.5]
   resGA = dds.results(lfcThreshold=.5, altHypothesis="greaterAbs")
   resLA = dds.results(lfcThreshold=.5, altHypothesis="lessAbs")
   resG = dds.results(lfcThreshold=.5, altHypothesis="greater")
   resL = dds.results(lfcThreshold=.5, altHypothesis="less")

   def drawlines(ax):
     ax.axhline(y=-.5, c="dodgerblue")
     ax.axhline(y=.5, c="dodgerblue")
     plt.show()

   drawlines(resGA.plotMA(ylim=ylim))
   drawlines(resLA.plotMA(ylim=ylim))
   drawlines(resG.plotMA(ylim=ylim))
   drawlines(resL.plotMA(ylim=ylim))

.. _access:

Access to all calculated values
-------------------------------

All row-wise calculated values (intermediate dispersion calculations,
coefficients, standard errors, etc.) are stored in the
:class:`~DESeqDataSet.DESeqDataSet` object, *e.g.* :code:`dds` in this vignette.
These values are accessible by inspecting the :code:`var` attribute of
:code:`dds`.  Descriptions of the columns are accessible through the
:code:`description` attribute:

.. repl::
   dds.var.iloc[:4,:4]
   dds.var.columns
   dds.var.description

The mean values :math:`\mu_{ij} = s_j q_{ij}` and the Cook's distances for each
gene and sample are stored as matrices in the :attr:`layers` attribute:

.. repl::
   dds.layers["mu"]
   dds.layers["cooks"]

The dispersions :math:`\alpha_i` can be accessed with the
:attr:`.DESeqDataSet.dispersions` attribute:

.. repl::
   dds.dispersions.head()
   dds.var["dispersion"].head()

The size factors :math:`s_j` are accessible via the
:attr:`.DESeqDataSet.sizeFactors` attribute:

.. repl::
   dds.sizeFactors

For advanced users, we also include a convenience function :func:`coef` for
extracting the matrix :math:`[\beta_{ir}]` for all genes :math:`i` and model
coefficients :math:`r`.  This function can also return a matrix of standard
errors, see the documentation of :func:`coef`.  The columns of this matrix
correspond to the effects returned by :meth:`.DESeqDataSet.resultsNames`.  Note
that the :meth:`.DESeqDataSet.results` function is best for building results
tables with *p*-values and adjusted *p*-values:

.. repl::
   coef(dds).head()

The beta prior variance :math:`\sigma_r^2` is stored as an attribute of the
:class:`~DESeqDataSet.DESeqDataSet`:

.. repl::
   dds.betaPriorVar

General information about the prior used for log fold change shrinkage is also
stored in a slot of the :class:`.DESeqResults` object. This would also contain
information about what other packages were used for log2 fold change shrinkage:

.. repl::
   resLFC.priorInfo
   resNorm.priorInfo
   resAsh.priorInfo

The dispersion prior variance :math:`\sigma_d^2` is stored as an attribute of
the dispersion function:

.. repl::
   dds.dispersionFunction
   dds.dispersionFunction.dispPriorVar

..
  The version of DESeq2 which was used to construct the
  :class:`~DESeqDataSet.DESeqDataSet` object, or the version used when
  :func:`DESeq` was run, is stored here:

  .. repl::
     dds.version


Sample-/gene-dependent normalization factors
--------------------------------------------

In some experiments, there might be gene-dependent dependencies which vary
across samples. For instance, GC-content bias or length bias might vary across
samples coming from different labs or processed at different times. We use the
terms *normalization factors* for a gene x sample matrix, and *size factors* for
a single number per sample.  Incorporating normalization factors, the mean
parameter :math:`\mu_{ij}` becomes:

.. math::
   \mu_{ij} = NF_{ij} q_{ij}

with normalization factor matrix :math:`NF` having the same dimensions as the
counts matrix :math:`K`. This matrix can be incorporated as shown below. We
recommend providing a matrix with gene-wise geometric means of 1, so that the
mean of normalized counts for a gene is close to the mean of the unnormalized
counts.  This can be accomplished by dividing out the current gene geometric
means:

>>> normFactors = normFactors / np.exp(np.mean(np.log(normFactors), axis=0))
>>> dds.normalizationFactors = normFactors

These steps then replace :meth:`.DESeqDataSet.estimateSizeFactors` which occurs
within the :func:`DESeq` function. The :func:`DESeq` function will look for
pre-existing normalization factors and use these in the place of size factors
(and a message will be printed confirming this).

..
  The methods provided by the `cqn <http://bioconductor.org/packages/cqn>`_ or
  `EDASeq <http://bioconductor.org/packages/EDASeq>`_ packages can help correct
  for GC or length biases. They both describe in their vignettes how to create
  matrices which can be used by DESeq2.  From the formula above, we see that
  normalization factors should be on the scale of the counts, like size factors,
  and unlike offsets which are typically on the scale of the predictors (*i.e.*
  the logarithmic scale for the negative binomial GLM). At the time of writing,
  the transformation from the matrices provided by these packages should be:

  ```{r offsetTransform, eval=FALSE}
  cqnOffset <- cqnObject$glm.offset
  cqnNormFactors <- exp(cqnOffset)
  EDASeqNormFactors <- exp(-1 * EDASeqOffset)
  ```

"Model matrix not full rank"
----------------------------

While most experimental designs run easily using design formula, some design
formulas can cause problems and result in the :func:`DESeq` function returning
an error with the text: "the model matrix is not full rank, so the model cannot
be fit as specified."  There are two main reasons for this problem: either one
or more columns in the model matrix are linear combinations of other columns, or
there are levels of factors or combinations of levels of multiple factors which
are missing samples. We address these two problems below and discuss possible
solutions.

Linear combinations
^^^^^^^^^^^^^^^^^^^

The simplest case is the linear combination, or linear dependency problem, when
two variables contain exactly the same information, such as in the following
sample table. The software cannot fit an effect for :code:`batch` and
:code:`condition`, because they produce identical columns in the model matrix.
This is also referred to as *perfect confounding*. A unique solution of
coefficients (the :math:`\beta_i` in the formula :ref:`below<theory>`) is not
possible:

.. repl::
   pd.DataFrame({"batch": Factor([1,1,2,2]),
                 "condition": Factor(["A", "A", "B", "B"])})

Another situation which will cause problems is when the variables are not
identical, but one variable can be formed by the combination of other factor
levels. In the following example, the effect of batch 2 vs 1 cannot be fit
because it is identical to a column in the model matrix which represents the
condition C vs A effect:

.. repl::
   pd.DataFrame({"batch": Factor([1,1,1,1,2,2]),
                 "condition": Factor(["A", "A", "B", "B", "C", "C"])})

In both of these cases above, the batch effect cannot be fit and must be removed
from the model formula. There is just no way to tell apart the condition effects
and the batch effects. The options are either to assume there is no batch effect
(which we know is highly unlikely given the literature on batch effects in
sequencing datasets) or to repeat the experiment and properly balance the
conditions across batches.  A balanced design would look like:

.. repl::
   pd.DataFrame({"batch": Factor([1,1,1,2,2,2]),
                 "condition": Factor(["A", "B", "C", "A", "B", "C"])})

.. _nested-div:

Group-specific condition effects, individuals nested within groups
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, there is a case where we *can* in fact perform inference, but we may
need to re-arrange terms to do so. Consider an experiment with grouped
individuals, where we seek to test the group-specific effect of a condition or
treatment, while controlling for individual effects. The individuals are nested
within the groups: an individual can only be in one of the groups, although each
individual has one or more observations across condition.

An example of such an experiment is below:

.. repl::
   colData = pd.DataFrame({"grp": Factor(["X","X","X","X","X","X","Y","Y","Y","Y","Y","Y"]),
                           "ind": Factor([1,1,2,2,3,3,4,4,5,5,6,6]),
                           "cnd": Factor(["A","B","A","B","A","B","A","B","A","B","A","B"])})
   colData

Note that individual (:code:`ind`) is a *factor* not a numeric. This is very
important.

We have two groups of samples X and Y, each with three distinct individuals
(labeled here 1-6). For each individual, we have conditions A and B (for
example, this could be control and treated).

This design can be analyzed by DESeq2 but requires a bit of refactoring in order
to fit the model terms. Here we will use a trick described in the `edgeR
<http://bioconductor.org/packages/edgeR>`_ user guide, from the section
*Comparisons Both Between and Within Subjects*.  If we try to analyze with a
formula such as, :code:`"~ ind + grp*cnd"`, we will obtain an error, because the
effect for group is a linear combination of the individuals.

However, the following steps allow for an analysis of group-specific condition
effects, while controlling for differences in individual.  For object
construction, you can use a simple design, such as :code:`"~ ind + cnd"`, as
long as you remember to replace it before running :func:`DESeq`.  Then add a
column :code:`ind_n` which distinguishes the individuals nested within a group.
Here, we add this column to :code:`colData`, but in practice you would add this
column to :code:`dds`:

.. repl::
   colData["ind_n"] = Factor([1,1,2,2,3,3,1,1,2,2,3,3])
   colData

Now we can reassign our :class:`~DESeqDataSet.DESeqDataSet` a design of
:code:`"~ grp + grp:ind.n + grp:cnd"`, before we call :func:`DESeq`. This new
design will result in the following model matrix:

.. repl::
   patsy.dmatrix("~ grp + grp:ind_n + grp:cnd", colData)


Note that, if you have unbalanced numbers of individuals in the two groups, you
will have zeros for some of the interactions between :code:`grp` and
:code:`ind.n`. You can remove these columns manually from the model matrix and
pass the corrected model matrix to the :code:`full` argument of the
:func:`DESeq` function. See example code in the next section. Note that, in this
case, you will not be able to create the :class:`~DESeqDataSet.DESeqDataSet`
with the design that leads to less than full rank model matrix. You can either
use :code:`design="~1"` when creating the dataset object, or you can provide the
corrected model matrix to the :attr:`.DESeqDataSet.design` attribute of the
dataset from the start.

Above, the terms :code:`grpX.cndB` and :code:`grpY.cndB` give the group-specific
condition effects, in other words, the condition B vs A effect for group X
samples, and likewise for group Y samples. These terms control for all of the
six individual effects.  These group-specific condition effects can be extracted
using :meth:`.DESeqDataSet.results` with the :code:`name` argument.

Furthermore, :code:`grpX.cndB` and :code:`grpY.cndB` can be contrasted using the
:code:`contrast` argument, in order to test if the condition effect is different
across group:

>>> dds.results(contrast=("grpY.cndB", "grpX.cndB"))


Levels without samples
^^^^^^^^^^^^^^^^^^^^^^

The function :func:`patsy.dmatrix` will produce a column of zeros if a level is
missing from a factor or a combination of levels is missing from an interaction
of factors. The solution to the first case is to call
:code:`remove_unused_categories` on the column, which will remove levels without
samples. This was shown in the beginning of this vignette.

The second case is also solvable, by manually editing the model matrix, and then
providing this to :func:`DESeq`. Here we construct an example dataset to
illustrate it:

.. repl::
   group = Factor([1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3])
   condition = Factor(["A","A","B","B","C","C","A","A","B","B","C","C","A","A","B","B","C","C"])
   d = pd.DataFrame({"group": group, "condition": condition})[:16]
   d

Note that if we try to estimate all interaction terms, we introduce a column
with all zeros, as there are no condition C samples for group 3
(:code:`np.asarray` is used to display the matrix):

.. repl::
   m1 = patsy.dmatrix("~condition*group", d)
   m1.design_info.column_names
   np.asarray(m1)
   all_zero = (m1 == 0).all(axis=0)
   all_zero

We can remove this column like so:

.. repl::
   m1 = m1[:,~all_zero]
   m1


Now this matrix :code:`m1` can be provided to the :code:`full` argument of
:func:`DESeq`.  For a likelihood ratio test of interactions, a model matrix
using a reduced design such as :code:`"~ condition + group"` can be given to the
:code:`reduced` argument. Wald tests can also be generated instead of the
likelihood ratio test, but for user-supplied model matrices, the argument
:code:`betaPrior` must be set to :code:`False`.

.. _theory:

Theory behind DESeq2
====================

The DESeq2 model
----------------

The DESeq2 model and all the steps taken in the software are described in detail
in our publication [Love2014]_, and we include the formula and descriptions in
this section as well.  The differential expression analysis in DESeq2 uses a
generalized linear model of the form:

.. math::
   K_{ij} \sim \textrm{NB}(\mu_{ij}, \alpha_i)

   \mu_{ij} = s_j q_{ij}

   \log_2(q_{ij}) = x_{j.} \beta_i

where counts :math:`K_{ij}` for gene :math:`i`, sample :math:`j` are modeled
using a negative binomial distribution with fitted mean :math:`\mu_{ij}` and a
gene-specific dispersion parameter :math:`\alpha_i`.  The fitted mean is
composed of a sample-specific size factor :math:`s_j` and a parameter
:math:`q_{ij}` proportional to the expected true concentration of fragments for
sample :math:`j`.  The coefficients :math:`\beta_i` give the log2 fold changes
for gene :math:`i` for each column of the model matrix :math:`X`.  Note that the
model can be generalized to use sample- and gene-dependent normalization factors
:math:`s_{ij}`.

The dispersion parameter :math:`\alpha_i` defines the relationship between the
variance of the observed count and its mean value. In other words, how far do we
expected the observed count will be from the mean value, which depends both on
the size factor :math:`s_j` and the covariate-dependent part :math:`q_{ij}` as
defined above.

.. math::
   \textrm{Var}(K_{ij}) = E[ (K_{ij} - \mu_{ij})^2 ] = \mu_{ij} + \alpha_i \mu_{ij}^2

An option in DESeq2 is to provide maximum *a posteriori* estimates of the log2
fold changes in :math:`\beta_i` after incorporating a zero-centered Normal prior
(:code:`betaPrior`). While previously, these moderated, or shrunken, estimates
were generated by :func:`DESeq` or :func:`nbinomWaldTest` functions, they are
now produced by the :func:`lfcShrink` function.  Dispersions are estimated using
expected mean values from the maximum likelihood estimate of log2 fold changes,
and optimizing the Cox-Reid adjusted profile likelihood, as first implemented
for RNA-seq data in `edgeR <http://bioconductor.org/packages/edgeR>`_
[@CR,edgeR_GLM]. The steps performed by the :func:`DESeq` function are
documented in its manual page; briefly, they are:

  1. estimation of size factors :math:`s_j` by
     :meth:`.DESeqDataSet.estimateSizeFactors`
  2. estimation of dispersion :math:`\alpha_i` by
     :meth:`.DESeqDataSet.estimateDispersions`
  3. negative binomial GLM fitting for :math:`\beta_i` and Wald statistics by
     :func:`nbinomWaldTest`

For access to all the values calculated during these steps, see the section
:ref:`above<access>`.

Changes compared to R DESeq2
----------------------------

This module is a Python port of the R package `DESeq2
<https://bioconductor.org/packages/release/bioc/html/DESeq2.html>`_. The port is
based on version 1.39.3, and changes to the R package since this version have
not been reflected in the present module.  Also note that the port is partial:
not all features may have been ported yet.  To help us prioritize our future
focus, please open a "feature request" issue on `inmoose GitHub
repository <https://github.com/epigenelabs/inmoose>`_.

The present page mirrors DESeq2 vignette, and despite our efforts, some code
examples may not accurately reflect the state of the Python module.

The main changes in this module compared to the original DESeq2 package are as
follows:

  - :class:`anndata.AnnData` is used as the superclass for storage of input
    data, intermediate calculations and results.


.. _changes:

Methods changes since the 2014 DESeq2 paper
-------------------------------------------

.. note::

   The changes below are present in the R DESeq2 package, and "we" stands for
   the authors of the R package, not the authors of InMoose.

* In version 1.18 (November 2017), we add two :ref:`alternative shrinkage
  estimators<shrink>`, which can be used via :func:`lfcShrink`: an estimator
  using a t prior from the apeglm packages, and an estimator with a fitted
  mixture of normals prior from the ashr package.
* In version 1.16 (November 2016), the log2 fold change shrinkage is no longer
  default for the :func:`DESeq` and :func:`nbinomWaldTest` functions, by setting
  the defaults of these to :code:`betaPrior=FALSE`, and by introducing a
  separate function :func:`lfcShrink`, which performs log2 fold change shrinkage
  for visualization and ranking of genes.  While for the majority of bulk
  RNA-seq experiments, the LFC shrinkage did not affect statistical testing,
  DESeq2 has become used as an inference engine by a wider community, and
  certain sequencing datasets show better performance with the testing separated
  from the use of the LFC prior. Also, the separation of LFC shrinkage to a
  separate function :func:`lfcShrink` allows for easier methods development of
  alternative effect size estimators.
* A small change to the independent filtering routine: instead of taking the
  quantile of the filter (the mean of normalized counts) which directly
  *maximizes* the number of rejections, the threshold chosen is the lowest
  quantile of the filter for which the number of rejections is close to the peak
  of a curve fit to the number of rejections over the filter quantiles.  "Close
  to" is defined as within 1 residual standard deviation.  This change was
  introduced in version 1.10 (October 2015).
* For the calculation of the beta prior variance, instead of matching the
  empirical quantile to the quantile of a Normal distribution, DESeq2 now uses
  the weighted quantile function of the Hmisc package. The weighting is
  described in the manual page for :func:`nbinomWaldTest`.  The weights are the
  inverse of the expected variance of log counts (as used in the diagonals of
  the matrix :math:`W` in the GLM). The effect of the change is that the
  estimated prior variance is robust against noisy estimates of log fold change
  from genes with very small counts. This change was introduced in version 1.6
  (October 2014).

Count outlier detection
-----------------------

DESeq2 relies on the negative binomial distribution to make estimates and
perform statistical inference on differences.  While the negative binomial is
versatile in having a mean and dispersion parameter, extreme counts in
individual samples might not fit well to the negative binomial. For this reason,
we perform automatic detection of count outliers. We use Cook's distance, which
is a measure of how much the fitted coefficients would change if an individual
sample were removed [Cook1977]_. For more on the implementation of Cook's
distance see the manual page for the :meth:`.DESeqDataSet.results` function.
Below we plot the maximum value of Cook's distance for each row over the rank of
the test statistic to justify its use as a filtering criterion.

.. repl::
   W = res["stat"]
   maxCooks = dds.layers["cooks"].max(axis=0)
   idx = ~np.isnan(W)
   plt.scatter(np.argsort(W[idx]), maxCooks[idx], c="grey")
   plt.xlabel("rank of Wald statistic")
   plt.ylabel("maximum Cook's distance per gene")
   m = dds.n_obs
   p = 3
   plt.axhline(y = scipy.stats.f.ppf(.99, p, m-p))
   plt.show()


Contrasts
---------

Contrasts can be calculated for a :class:`~DESeqDataSet.DESeqDataSet` object for
which the GLM coefficients have already been fit using the Wald test steps
(:func:`DESeq` with :code:`test="Wald"` or using :func:`nbinomWaldTest`).  The
vector of coefficients :math:`\beta` is left multiplied by the contrast vector
:math:`c` to form the numerator of the test statistic. The denominator is formed
by multiplying the covariance matrix :math:`\Sigma` for the coefficients on
either side by the contrast vector :math:`c`. The square root of this product is
an estimate of the standard error for the contrast. The contrast statistic is
then compared to a Normal distribution as are the Wald statistics for the DESeq2
package.

.. math::
   W = \frac{c^t \beta}{\sqrt{c^t \Sigma c}}


Expanded model matrices
-----------------------

For the specific combination of :func:`lfcShrink` with the type :code:`normal`
and using :code:`contrast`, DESeq2 uses *expanded model matrices* to produce
shrunken log2 fold change estimates where the shrinkage is independent of the
choice of reference level. In all other cases, DESeq2 uses standard model
matrices, as produced by :func:`patsy.dmatrix`.  The expanded model matrices
differ from the standard model matrices, in that they have an indicator column
(and therefore a coefficient) for each level of factors in the design formula in
addition to an intercept. This is described in the DESeq2 paper. Using type
:code:`normal` with :func:`coef` uses standard model matrices, as does the
:code:`apeglm` shrinkage estimator.

.. _indfilttheory:

Independent filtering and multiple testing
------------------------------------------

Filtering criteria
^^^^^^^^^^^^^^^^^^

The goal of independent filtering is to filter out those tests from the
procedure that have no, or little chance of showing significant evidence,
without even looking at their test statistic. Typically, this results in
increased detection power at the same experiment-wide type I error. Here, we
measure experiment-wide type I error in terms of the false discovery rate.

A good choice for a filtering criterion is one that:

  1. is statistically independent from the test statistic under the null
     hypothesis,
  2. is correlated with the test statistic under the alternative, and
  3. does not notably change the dependence structure -- if there is any --
     between the tests that pass the filter, compared to the dependence
     structure between the tests before filtering.

The benefit from filtering relies on property (2), and we will explore it
further below. Its statistical validity relies on property (1) -- which is
simple to formally prove for many combinations of filter criteria with test
statistics -- and (3), which is less easy to theoretically imply from first
principles, but rarely a problem in practice.  We refer to [Bourgon2010]_ for
further discussion of this topic.

A simple filtering criterion readily available in the results object is the mean
of normalized counts irrespective of biological condition, and so this is the
criterion which is used automatically by the :meth:`.DESeqDataSet.results`
function to perform independent filtering.  Genes with very low counts are not
likely to see significant differences typically due to high dispersion. For
example, we can plot the :math:`-\log_{10}` *p*-values from all genes over the
normalized mean counts:

.. repl::
   plt.scatter(res["baseMean"]+1, -np.log10(res["pvalue"]), c="black")
   plt.xscale("log")
   plt.xlabel("mean of normalized counts")
   plt.ylabel("-log[10](pvalue)")
   plt.ylim(0,30)
   plt.show()


Why does it work?
^^^^^^^^^^^^^^^^^

Consider the *p*-value histogram below: it shows how the filtering ameliorates
the multiple testing problem -- and thus the severity of a multiple testing
adjustment -- by removing a background set of hypotheses whose *p*-values are
distributed more or less uniformly in [0,1].

.. repl::
   use = res["baseMean"] > res.filterThreshold
   plt.hist(res["pvalue"][~use], bins=50, color="khaki", label="do not pass")
   plt.hist(res["pvalue"][use], bins=50, color="powderblue", label="pass")
   plt.legend(loc="upper right")
   plt.show()

Histogram of *p*-values for all tests.  The area shaded in blue indicates the
subset of those that pass the filtering, the area in khaki those that do not
pass.


References
==========

.. [AhlmannEltze2020] C. Ahlmann-Eltze, W. Huber. 2020. glmGamPoi: fitting
   Gamma-Poisson generalized linear models on single-cell count data.
   *Bioinformatics*, 36(24). :doi:`10.1093/bioinformatics/btaa1009`

.. [Anders2010] S. Anders and W. Huber. 2010. Differential expression for
   sequence count data. *Genome Biology*, 11:106.
   :doi:`10.1186/gb-2010-11-10-r106`

.. [Berge2018] K. van den Berge, F. Perraudeau, C. Soneson, M.I. Loce, D. Risso,
   J.P. Vert, M.D. Robinson, S. Dudoit, L. Clement. 2018. Observation weights
   unlock bulk RNA-seq tools for zero inflation and single-cell applications.
   *Genome Biology*, 19(24). :doi:`10.1186/s13059-018-1406-4`

.. [Bourgon2010] R. Bourgon, R. Gentleman, W. Huber. 2010. Independent filtering
   increases detection power for high-throughput experiments. *PNAS*,
   107(21):9546-51.  :doi:`10.1073/pnas.0914005107`

.. [Cook1977] R.D. Cook. 1977. Detection of inferential observation in linear
   regression. *Technometrics*, 19(1):15-18. :doi:`10.2307/1268249`

.. [Gerard2020] D. Gerard, M. Stephens. 2020. Empirical Bayes shrinkage and
   false discovery rate estimation, allowing for unwanted variation.
   *Biostatistics*, 21(1):15-32. :doi:`10.1093/biostatistics/kxy029`

.. [Leek2014] J.T. Leek. 2014. svaseq: removing batch effects and other unwanted
   noise from sequencing data. *Nucleic Acids Research*, 42(21).
   :doi:`10.1093/nar/gku864`

.. [Liao2013] Y. Liao, G.K. Smyth, W. Shi. 2013. featureCounts: an efficient
   general purpose program for assigning sequence reads to genomic features.
   *Bioinformatics*, 30(7):923-30. :doi:`10.1093/bioinformatics/btt656`

.. [Love2014] M.I. Love, W. Huber, S. Anders. 2014. Moderated estimation of fold
   change and dispersion for RNA-seq data with DESeq2. *Genome Biology*, 15(12),
   550.  :doi:`10.1186/s13059-014-0550-8`

.. [McCarthy2012] D.J. McCarthy, Y. Chen, G.K. Smyth. 2012. Differential
   expression analysis of multifactor RNA-Seq experiments with respect to
   biological variation. *Nucleic Acids Research* 40, 4288-4297.
   :doi:`10.1093/nar/gks042`

.. [Risso2014] D. Risso, J. Ngai. T.P. Speed, S. Dudoit. 2014. Normalization of
   RNA-seq data using factor analysis of control genes or samples. *Nature
   Biotechnology*, 32(9). :doi:`10.1038/nbt.2931`

.. [Stephens2016] M. Stephens. 2016. False discovery rates: a new deal.
   *Biostatistics*, 18(2). :doi:`10.1093/biostatistics/kxw041`

.. [Storey2003] J. Storey. 2003. The positive false discovery rate: a Bayesian
   interpretation and the *q*-value. *The Annals of Statistics*,
   31(6):2013-2035.

.. [Wu2012] H. Wu, C. Wang, Z. Wu. 2012. A new shrinkage estimator for
   dispersion improves differential detection in RNA-Seq data. *Biostatistics*.
   :doi:`10.1093/biostatistics/kxs033`

.. [Zhu2018] A. Zhu, J.G. Ibrahim, M.I. Love. 2018. Heavy-tailed prior
   distributions for sequence count data: removing the noise and preserving
   large differences. *Bioinformatics*, 35(12), 2084-2092.
   :doi:`10.1093/bioinformatics/bty895`

Code documentation
==================

.. autosummary::
   :toctree: generated/

   ~DESeqDataSet.DESeqDataSet
   ~results.DESeqResults

   DESeq
   collapseReplicates
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

