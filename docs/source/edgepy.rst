edgepy
======

.. currentmodule:: inmoose.edgepy

This module is a partial port in Python of the R Bioconductor `edgeR package
<https://bioconductor.org/packages/release/bioc/html/edgeR.html>`_.
Only the functionalities necessary to :func:`inmoose.pycombat.pycombat_seq` have
been ported so far.

.. autosummary::
   :toctree: generated/

   DGEList

   addPriorCount
   adjustedProfileLik
   aveLogCPM
   designAsFactor
   dispCoxReid
   dispCoxReidInterpolateTagwise
   estimateGLMCommonDisp
   estimateGLMTagwiseDisp
   glmFit
   mglmLevenberg
   mglmOneGroup
   mglmOneWay
   movingAverageByCol
   nbinomDeviance
   predFC
   splitIntoGroups
   systematicSubset
   validDGEList
