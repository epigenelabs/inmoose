# InMoose release changelog

## [0.7.7-dev]

- improve CC plots layout
- fix `glmLRT` when given a contrast in argument
- fix argument type in `fitDispGridWrapper`

## [0.7.6]

- add doc for cohort QC report
- better handling of PCAs in QC report

## [0.7.5]

- harmonize the adjusted p-values in diff exp modules
- fix build issues with Cython 3.1.0

## [0.7.4]

- fix single sample clustering error, changing cluster consensus to nan
- improve pyDESeq2 documentation
- various doc improvements
- support for Python 3.13
- beautify the cohort QC report
- raise error on cohort metrics when there is only one batch
- fix patsy formula parsing of categorical variables with spaces in their names

## [0.7.3]

- fix dispersion estimation and MAD computation in `deseq2` module
- improve GLM fit in `deseq2` module
- create `cohort_qc` module with `CohortMetric` and `QCReport` classes to
  evaluate dataset quality after batch effect correction
- improve `cohort_qc` module when the number of covariates is 0 or 1
- improve documentation for `diffexp` module
- improve documentation for `consensus_clustering` module
- batch effect correction functions now accept `AnnData` as input
- fix p-value adjustment in `deseq2` module
- add logos and badges in the README

## [0.7.2]

- opt out of `numpy` 2.1.0 until `statsmodels` supports it
- fix an array indexing bug in function `meta_de`
- function `meta_de` is now robust to NaN *p*-values
- allow arbitrary types for `batch` and `group` arguments in function
  `sim_rnaseq`

## [0.7.1]

- fix a bug when splicing a `DEResults`
- fix coverage report upload in CI/CD pipeline
- improve documentation landing page and references
- use a package-specific logger

## [0.7.0]

- harmonize differential expression analysis results across `deseq2`, `edgepy`
  and `limma` modules
- add a meta-analysis feature for differential expression

## [0.6.0]

- move most of C++ code to Python and Cython
- compatibility with `numpy` 2.0.0

## [0.5.1]

- fix pycombat covariates action remove with count dataframe output

## [0.5.0]

- add experimental support for MSVC compiler
- add R-like functions for fitting linear models in module `utils`
- add a function to compute natural cubic spines basis in module `utils`
- add tmixture functions from R package limma
- add linear model fit functions from R package limma
- add functions handling duplicated probes from R package limma
- lint code with `ruff`
- add a "differential expression analysis" section to the doc

## [0.4.1]

- fix an incorrect github action version number

## [0.4.0]

- support for Python 3.12
- bump github actions versions
- publish coverage report to coveralls.io
- port `edgeR` functions for differential expression to `inmoose`
- Modifiy clustering stability plot
- remove many (not all) warnings

## [0.3.1]

- Fix issue #25 and #10 related to counts parameter data format check and gestion (when it's a pandas dataframe)
- add DESeq2 variance stabilizing transformation to the `deseq2` package
- improve the error message for confounding variable in `pycombat` package
- fix build on readthedocs

## [0.3.0]

- add and optimize a few functions in the `utils.stats` module
- add a module featuring a port of the DESeq2 R package

## [0.2.4]

- add a `data` module, with two RNA-Seq datasets: `pasilla` and `airway`
- fix build on readthedocs
- updated bibliography with our BMC Bioinformatics paper
- add verbose to consensus clustering computation
- add a utility class `VirtualCohortInput` to better check the well definition
  of inputs to the batch effect correction routines

## [0.2.3]

- fix syntax-impairing typos in `pyproject.toml`

## [0.2.2]

- proper documentation for the `sim` module
- code is now formatted with `black`
- add a link to the documentation in the README file
- better handling of covariates in `pycombat` module

## [0.2.1]

- add a function to generate simulated RNA-Seq and scRNA-Seq data
- drop support of Python 3.8 due to memory management issues in extensions
- add a module to perform consensus clustering and select the optimal number of
  clusters

## [0.2.0]

- `inmoose` now requires Python >= 3.8
- reorganize tests directory
- refactor module for batch effect correction. It is now named `pycombat`
  (instead of `batch`), and the function to correct batch effect on microarray
  data is now `pycombat_norm` (instead of `pycombat`).
- refactor design matrix computation, to share the code between `pycombat_norm`
  and `pycombat_seq`
- batch effect correction functions now accept both `numpy` arrays and `pandas`
  dataframes as input for the counts matrix
- improved logging: no more `print`s, better log formatting
- `inmoose` doc is now on `readthedocs.org`

## [0.1.1]

- upgrade to Cython 3.0.0b2
- fix C++ extension loading on Linux
- add CI/CD to build, test and publish distributions

## [0.1.0]

Initial release

