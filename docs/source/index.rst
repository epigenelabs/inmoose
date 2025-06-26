=================================
Welcome to InMoose documentation!
=================================

InMoose is the INtegrated Multi Omic Open Source Environment.

InMoose is intended as a comprehensive state-of-the-art Python package for -omic
data analysis. Its current focus is on analysis of bulk transcriptomic data
(microarray and RNA-Seq).
It comprises Python ports of popular and recognized R tools, name ComBat
[Johnson2007]_, ComBat-Seq [Zhang2020]_, DESeq2 [Love2014]_, edgeR [Chen2016]_,
limma [Ritchie2015]_ and splatter [Zappia2017]_.

.. toctree::
   :maxdepth: 1
   :caption: Features

   pycombat
   data
   diffexp

.. toctree::
   :maxdepth: 1
   :caption: API

   deseq
   clustering
   cohort_qc
   edgepy
   limma
   sim
   utils

Contributing to InMoose
=======================

Contribution guidelines are described in `CONTRIBUTING.md <https://github.com/epigenelabs/inmoose/blob/master/CONTRIBUTING.md>`_.

Authors
=======

Contact
-------
To report bugs (if any?), ask for support or request improvements and new
features, please open a ticket on our Github repository.
You may also directly contact:

Maximilien Colange at maximilien@epigenelabs.com

Logo
----

The InMoose logo was designed by Léa Meunier.

Citing
======

The :doc:`pycombat <pycombat>` module was previously `distributed independently <https://github.com/epigenelabs/pycombat>`_.

To cite InMoose, please use one of the following references:

M. Colange, G. Appé, L. Meunier, S. Weill, W.E. Johnson, A. Nordor, A. Behdenna.
2025.  Bridging the gap between R and Python in bulk transcriptomic data
analysis with InMoose. Nature Scientific Reports 15:18104.
:doi:`10.1038/s41598-025-03376-y`.

A. Behdenna, M. Colange, J. Haziza, A. Gema, G. Appé, C.-A. Azencott and A.
Nordor. 2023. pyComBat, a Python tool for batch effects correction in
high-throughput molecular data using empirical Bayes methods.  *BMC
Bioinformatics* 24:459. :doi:`10.1186/s12859-023-05578-5`

M. Colange, G. Appé, L. Meunier, S. Weill, A. Nordor, A.  Behdenna. 2025.
Differential Expression Analysis with InMoose, the Integrated Multi-Omic
Open-Source Environment in Python. *BMC Bioinformatics* 26:160.
:doi:`10.1186/s12859-025-06180-7`


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
