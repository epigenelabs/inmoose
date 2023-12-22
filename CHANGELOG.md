# InMoose release changelog

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

