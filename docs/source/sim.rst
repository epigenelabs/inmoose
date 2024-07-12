===
sim
===

.. currentmodule:: inmoose.sim

This module is a partial reimplementation of the R Bioconductor `splatter
package <https://bioconductor.org/packages/release/bioc/html/splatter.html>`_.
Its main feature is :func:`sim_rnaseq`, a function to generate simulated RNA-Seq
or single-cell RNA-Seq data based on the Splat model described in [Zappia2017]_.

.. autosummary::
   :toctree: generated/

   sim_rnaseq

References
==========

.. [Zappia2017] L. Zappia, B. Phipson, A. Oshlack. 2017. Splatter: simulation of
   single-cell RNA sequencing data. *Genome Biology* 18(174).
   :doi:`10.1186/s13059-017-1305-0`


