======
edgepy
======

.. currentmodule:: inmoose.edgepy

This module is a partial port in Python of the R Bioconductor `edgeR package
<https://bioconductor.org/packages/release/bioc/html/edgeR.html>`_.
Only the functionalities necessary to :func:`inmoose.pycombat.pycombat_seq` have
been ported so far.

References
==========

.. [Chen2016] Y. Chen, A.T.L Lun, G.K. Smyth. 2016. From reads to genes to
   pathways: differential expression analysis of RNA-Seq experiments using
   Rsubread and the edgeR quasi-likelihood pipeline. *F1000Research* 5, 1438.
   :doi:`10.12688/f1000research.8987.2`

.. [Gibbons1975] J.D. Gibbons, J.W. Pratt. 1975. P-values: interpretation and
   methodology. *The American Statistician* 29, 20-25.
   :doi:`10.1080/00031305.1975.10479106`

.. [Lun2016] A.T.L. Lun, Y. Chen, G.K. Smyth. 2016. It's DE-licious: a recipe
   for differential expression analyses of RNA-seq experiments using
   quasi-likelihood methods in edgeR.  *Methods in Molecular Biology* 1418,
   391-416. :doi:`10.1007/978-1-4939-3578-9_19`

.. [Lund2012] S.P. Lund, D. Nettleton, D.J. McCarthy, G.K. Smyth. 2012.
   Detecting differential expression in RNA-sequence data using quasi-likelihood
   with shrunken dispersion estimates. *Statistical Applications in Genetics and
   Molecular Biology* Volume 11, Issue 5, Article 8.
   :doi:`10.1515/1544-6115.1826`

.. [Lun2017] A.T.L. Lun, G.K. Smyth. 2017. No counts, no variance: allowing for
   loss of degrees of freedom when assessing biological variability from RNA-seq
   data. *Statistical Applications in Genetics and Molecular Biology* 16(2),
   83-93. :doi:`10.1515/sagmb-2017-0010`

.. [McCarthy2012] D. J. McCarthy, Y. Chen, G. K. Smyth. 2012. Differential
   expression analysis of multifactor RNA-Seq experiments with respect to
   biological variation. Nucleic Acids Research 40, 4288-4297.
   :doi:`10.1093/nar/gks042`

.. [Phipson2016] B. Phipson, S. Lee, I.J. Majewski, W. S. Alexander, G.K. Smyth.
   2016. Robust hyperparameter estimation protects against hypervariable genes
   and improves power to detect differential expression. *Annals of Applied
   Statistics* 10, 946-963. :doi:`10.1214/16-AOAS920`

.. [Robinson2008] M.D. Robinson, g.K. Smyth. 2008. Small-sample estimation of
   negative binomial dispersion, with applications to SAGE data.
   *Biostatistics* 9, 321-332.  :doi:`10.1093/biostatistics/kxm030`


Code documentation
==================

.. autosummary::
   :toctree: generated/

   DGEList

   addPriorCount
   adjustedProfileLik
   aveLogCPM
   binomTest
   designAsFactor
   dispCoxReid
   dispCoxReidInterpolateTagwise
   estimateGLMCommonDisp
   estimateGLMTagwiseDisp
   exactTest
   exactTestBetaApprox
   exactTestByDeviance
   exactTestBySmallP
   exactTestDoubleTail
   glmFit
   glmLRT
   glmQLFit
   glmQLFTest
   mglmLevenberg
   mglmOneGroup
   mglmOneWay
   movingAverageByCol
   nbinomDeviance
   plotQLDisp
   predFC
   splitIntoGroups
   systematicSubset
   topTags
   validDGEList
