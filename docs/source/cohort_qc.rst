======================
Cohort Quality Control
======================

.. currentmodule:: inmoose.cohort_qc

In addition to its batch effect correction features, InMoose can generate a
HTML report to assess how well batch effects are corrected, and how they
correlate with co-variates.

We illustrate its usage with data freely available on NCBI `Gene Expression Omnibus <https://www.ncbi.nlm.nih.gov/geo/>`_, namely:

  * GSE18520
  * GSE66957
  * GSE69428

The corresponding expression files are stored on InMoose repository in the
`data` subfolder.

.. code-block:: Python

   import pandas as pd

   from inmoose.cohort_qc import CohortMetric, QCReport
   from inmoose.pycombat import pycombat_norm

   dataset_1 = pd.read_pickle("data/GSE18520.pickle")
   dataset_2 = pd.read_pickle("data/GSE66957.pickle")
   dataset_3 = pd.read_pickle("data/GSE69428.pickle")
   datasets = [dataset_1, dataset_2, dataset_3]

   # merge all three datasets into a single one, keeping only common genes
   df_expression = pd.concat(datasets, join="inner", axis=1)

   batch = [j for j, ds in enumerate(datasets) for _ in range(len(ds.columns))]

   # run pycombat_norm
   df_corrected = pycombat_norm(df_expression, batch)

   # compute cohort metrics
   cohort_metric = CohortMetric(
       clinical_df=pd.DataFrame({"batch": batch}, index=df_expression.columns),
       batch_column="batch",
       data_expression_df=df_corrected,
       data_expression_df_before=df_expression,
   )
   cohort_metric.process()

   # build QC report
   report = QCReport(cohort_metric)
   report.save_report("report.html")

The snippet above generates the following report:

.. raw:: html
   :file: qc_report.html
