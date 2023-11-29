Welcome to InMoose documentation!
=================================

InMoose is the INtegrated Multi Omic Open Source Environment.

InMoose is intended as a comprehensive state-of-the-art Python package for -omic
data analysis. It comprises Python ports of popular and recognized R tools,
namely ComBat [1]_ and ComBat-Seq [2]_.

.. toctree::
   :maxdepth: 1
   :caption: Features

   pycombat
   data

.. toctree::
   :maxdepth: 1
   :caption: API

   edgepy
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
Nordor. 2020. pyComBat, a Python tool for batch effects correction in
high-throughput molecular data using empirical Bayes methods. *bioRxiv*.
:doi:`10.1101/2020.03.17.995431`

References
==========

.. [1] W. E. Johnson, C. Li, A. Rabinovic. 2007. Adjusting batch effects in
   microarray expression data using empirical Bayes methods. *Biostatistics*, 8,
   118–12.  :doi:`10.1093/biostatistics/kxj037`

.. [2] Y. Zhang, G. Parmigiani, W. E. Johnson. 2020. ComBat-Seq: batch effect
   adjustment for RNASeq count data. *NAR Genomics and Bioinformatics*, 2(3).
   :doi:`10.1093/nargab/lqaa078`

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
