=====
limma
=====

.. currentmodule:: inmoose.limma

This module is a partial port in Python of the R Bioconductor `limma package
<https://bioconductor.org/packages/release/bioc/html/limma.html>`_.

Introduction
============

:mod:`limma` is a package for the analysis of gene expression microarray data,
especially the use of linear models for analysing designed experiments and the
assessment of differential expression. :mod:`limma` provides the ability to
analyse comparisons between many RNA targets simultaneously in arbitrarily
complicated designed experiments. Empirical Bayesian methods are used to provide
stable results even when the number of arrays is small. The linear model and
differential expression functions apply to all gene expression technologies,
including microarrays, RNA-Seq and quantitative PCR.

There are three types of documentation available:

  1. the *LIMMA User's Guide* can be reached through the "User Guides and
     Package Vignettes" links at the top of the LIMMA contents page.
  2. an overview of limma functions grouped by purpose is contained in the
     chapters of the present page
  3. the Code documentation section gives an alphabetical index of detailed help
     topics

Classes Defined in this Module
==============================

This module defines the following data classes:

- :class:`MArrayLM`: store the result of fitting gene-wise linear models to the
  normalized intensities or log-ratios. Usually created by :func:`lmFit`.
  Objects of this class normally contain only one row for each unique probe.
- :class:`TestResults`: store the results of testing a set of contrasts equal to
  zero for each probe. Usually created by :func:`decideTests`. Objects of this
  class normally contain one row for each unique probe.

.. _linearmodels:

Linear Models for Microarrays
=============================

This section gives an overview of the LIMMA functions available to fit linear
models and to interpret the results. This section covers models for two color
arrays in terms of log-ratios or for single-channel arrays in terms of
log-intensities. If you wish to fit models to the individual channel
log-intensities from two color arrays, see :ref:`singlechannel`.

The core of this module is the fitting of gene-wise linear models to microarray
data. The basic idea is to estimate log-ratios between two or more target RNA
samples simultaneously. See the LIMMA User's Guide for several case studies.

Fitting Models
--------------

The main function for model fitting is :func:`lmFit`. This is the recommended
interface for most users. :func:`lmFit` produces a fitted model object of class
:class:`MArrayLM` containing coefficients, standard errors and residual standard
errors for each gene. :func:`lmFit` calls one of the following three functions
to do the actual computations:

- :func:`lm_series` Straightforward least squares fitting of a linear model for
  each gene.
- :func:`mrlm` An alternative to :func:`lm_series` using robust regression as
  implemented by the :code:`rlm` function in the MASS package.
- :func:`gls_series` Generalized least squares taking into account correlations
  between duplicate spots (*i.e.* replicate spots on the same array) or related
  arrays. The function :func:`duplicateCorrelation` is used to estimate the
  inter-duplicate or inter-block correlation before using :func:`gls_series`.

All the function which fit linear models use :func:`getEAWP` to extract data
from microarray data objects, and :func:`unwrapdups` which provides a unified
method for handling duplicate spots.

Forming the Design Matrix
-------------------------

:func:`lmFit` has two main arguments: the expression data and the design matrix.
The design matrix is essentially an indicator matrix which specifies which
target RNA samples were applied to each channel on each array. There is
considerable freedom in choosing the design matrix -- there is always more than
one choice which is correct provided it is interpreted correctly.

Design matrices for Affymetrix or single-color arrays can be created using the
function :func:`patsy.dmatrix`. The function :func:`modelMatrix` is provided to
assist with the creation of an appropriate design matrix for two-color
microarray experiments. For direct two-color designs, without a common
reference, the design matrix often needs to be created by hand.

Making Comparisons of Interest
------------------------------

Once a linear model has been fit using an appropriate design matrix, the
function :func:`makeContrasts` may be used to form a contrast matrix to make
comparisons of interest. The fit and the contrast matrix are used by
:func:`contrasts_fit` to compute fold changes and *t*-statistics for the
contrasts of interest. This is a way to compute all possible pairwise
comparisons between treatments, for example in an experiment which compares many
treatments to a common reference.

Assessing Differential Expression
---------------------------------

After fitting a linear model, the standard errors are moderated using a simple
empirical Bayes model using :func:`eBayes` or :func:`treat`. A moderated
*t*-statistic and a log-odds of differential expression is computed for each
contrast for each gene. :func:`treat` tests whether log-fold-changes are greater
than a threshold rather than merely different to zero.

:func:`eBayes` and :func:`treat` use internal functions :func:`squeezeVar`,
:func:`fitFDist`, :func:`tmixture_matrix` and :func:`tmixture_vector`.

Summarizing Model Fits
----------------------

After the above steps, the results may be displayed or further processed using:

- :func:`topTable` Presents a list of the genes most likely to be differentially
  expressed for a given contrast or set of contrasts.
- :func:`topTableF` Presents a list of the genes most likely to be
  differentially expressed for a given set of contrasts. Equivalent to
  :func:`topTable` with :code:`coef` set to all the coefficients,
  :code:`coef=range(fit.shape[1])`.
- :func:`volcanoplot` Volcano plot of fold change versus the *B*-statistic for
  any fitted coefficient.
- :func:`plotlines` Plots fitted coefficients or log-intensity values for
  time-course data.
- :func:`genas` Estimates biological correlation between two coefficients.
- :func:`write_fit` Writes a :class:`MArrayLM` object to a file. Note that if
  :code:`fit` is a :class:`MArrayLM` object, either :func:`write_fit` or
  :func:`write_table` can be used to write the results to a delimited text file.

For multiple testing functions which operate on linear model fits, see
:ref:`tests`.

Model Selection
---------------

:func:`selectModel` provides a means to choose between alternative linear models
using AIC or BIC information criteria.


.. _singlechannel:

Individual Channel Analysis of Two-Color Microarrays
====================================================

This section gives an overview of the LIMMA functions fit linear models to
two-color microarray data in terms of the log-intensities rather than
log-ratios.

The function :func:`intrapotCorrelation` estimates the intra-spot correlation
between the two channels. The regression function :func:`lmscFit` takes the
correlation as an argument and fits linear models to the two-color data in terms
of the individual log-intensities. The output of :func:`lmscFit` is a
:class:`MArrayLM` object just the same as from :func:`lmFit`, so inference
proceeds the same way as for log-ratios once the linear model is fitted. See
:ref:`linearmodels`.

The function :func:`targetsA2C` converts two-color format target data frames to
single channel format, *i.e.* converts from array-per-line to channel-per-line,
to facilitate the formulation of the design matrix.

.. _tests:

Hypothesis Testing for Linear Models
====================================

LIMMA provides a number of functions for multiple testing across both contrasts
and genes. The starting point is a :class:`MArrayLM` object, called :code:`fit`
say, resulting from fitting a linear model and running :func:`eBayes` and,
optionally, :func:`contrasts_fit`. See :ref:`linearmodels` or
:ref:`singlechannel` for details.

Multiple Testing across Genes and Contrasts
-------------------------------------------

The key function is :func:`decideTests`. This function writes an object of class
:class:`TestResults`, which is basically a matrix of :math:`-1`, :math:`0` or
:math:`1` elements, of the same dimension as :code:`fit_coefficients`,
indicating whether each coefficient is significantly different from zero. A
number of multiple testing strategies are provided. :func:`decideTests` calls
:func:`classifyTestsF` to implement the nested *F*-test strategy.

:func:`selectModel` chooses between linear models for each probe using AIC or
BIC criteria. This is an alternative to hypothesis testing and can choose
between non-nested models.

A number of other functions are provided to display the results of
:func:`decideTests`. The function :func:`heatDiagram` displays the results in a
heat-map style display. This allows visual comparison of the results across many
different conditions in the linear model.

The functions :func:`vennCounts` and :func:`vennDiagram` provide Venn diagrams
style summaries of the results.

Summary and :func:`show` method exists for objects of class
:class:`TestResults`.

The results from :func:`decideTests` can also be included when the results of a
linear model fit are written to a file using :func:`write_fit`.

#Gene Set Tests
#--------------
#
#Competitive gene set testing for an individual gene set is provided by
#:func:`wilcoxGST` or :func:`geneSetTest`, which permute genes. The gene set can
#be displayed using :func:`barcodeplot`.
#
#Self-contained gene set testing for an individual set is provided by
#:func:`roast`, which uses rotation technology, analogous to permuting arrays.
#
#Gene set enrichment analysis for a large database of gene sets is provided by
#:func:`romer`. :func:`topRomer` is used to rank results from :func:`romer`.
#
#The functions :func:`alias2Symbol`, :func:`alias2SymbolTable` and
#:func:`alias2SymbolUsingNCBI` are provided to help match gene sets with
#microarray probes by way of official gene symbols.
#
#Global Tests
#------------
#
#The function :func:`genas` can test for associations between two contrasts in a
#linear model.
#
#Given a set of *p*-values, the function :func:`propTrueNull` can be used to
#estimate the proportion of true null hypothesis.
#
#When evaluating test procedures with simulated or known results, the utility
#function :func:`auROC` can be used to compute the area under the Receiver
#Operating Curve for the test results for a given probe.


References
==========

.. [Law2014] C.W. Law, Y. Chen, W. Shi, G.K. Smyth. 2014. Voom: precision
   weights unlock linear model analysis tools for RNA-seq read counts. *Genome
   Biology* 15(R29). :doi:`10.1186/gb-2014-15-2-r29`

.. [Loennstedt2002] I. Loennstedt, T.P. Speed. 2002. Replicated microarray data.
   *Statistica Sinica* 12(31-46).

.. [Michaud2008] J. Michaud, K.M. Simpson, R. Escher, K. Buchet-Poyau, R.
   Beissbarth, C. Carmichael, M.E. Ritchie, F. Schutz, P. Cannon, M. Liu, X.
   Shen, Y. Ito, W.H. Raskind, M.S. Horwitz, M. Osato, D.R. Turner, T.P. Speed,
   M. Kavallaris, G.K. Smyth, H.S. Scott. 2008. Integrative analysis of RUNX1
   downstream pathways and target genes. *BMC Genomics* 9(363).
   :doi:`10.1186/1471-2164-9-363`

.. [Ritchie2015] M.E. Ritchie, B. Phipson, D. Wu, Y. Hu, C.W. Law, W. Shi, G.K.
   Smyth. 2015. Limma powers differential expression analyses for RNA-sequencing
   and microarray studies. *Nucleic Acids Research* 43(7).
   :doi:`10.1093/nar/gkv007`

.. [Sartor2006] M.A. Sartor, C.R. Tomlinson, S.C. Wesselkamper, S. Sivaganesan,
   G.D. Leikauf, M. Medvedovic. 2006. Intensity-based hierarchical Bayes method
   improves testing for differentially expressed genes in microarray
   experiments. *BMC Bioinformatics* 7(538). :doi:`10.1186/1471-2105-7-538`

.. [Smyth2004] G.K. Smyth. 2004. Linear models and empirical Bayes methods for
   assessing differential expression in microarray experiments. *Statistical
   Applications in Genetics and Molecular Biology* 3(1).
   :doi:`10.2202/1544-6115.1027`


Code documentation
==================

.. autosummary::
   :toctree: generated/

   MArrayLM
   TestResults

   classifyTestsF
   contrasts_fit
   eBayes
   fitFDist
   lmFit
   lm_series
   makeContrasts
   squeezeVar
   topTable
   tmixture_matrix
   tmixture_vector
