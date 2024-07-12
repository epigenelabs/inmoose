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

To cite InMoose, please use the following reference:

A. Behdenna, M. Colange, J. Haziza, A. Gema, G. Appé, C.-A. Azencott and A.
Nordor. 2023. pyComBat, a Python tool for batch effects correction in
high-throughput molecular data using empirical Bayes methods.  *BMC
Bioinformatics* 7;24(1):459. :doi:`10.1186/s12859-023-05578-5`


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
