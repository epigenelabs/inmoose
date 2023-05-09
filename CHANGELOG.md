# InMoose release changelog

## [0.1.2-dev]

- reorganize tests directory
- refactor module for batch effect correction. It is now named `pycombat`
  (instead of `batch`), and the function to correct batch effect on microarray
  data is now `pycombat_norm` (instead of `pycombat`).

## [0.1.1]

- upgrade to Cython 3.0.0b2
- fix C++ extension loading on Linux
- add CI/CD to build, test and publish distributions

## [0.1.0]

Initial release

