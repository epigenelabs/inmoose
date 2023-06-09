# Contributing to InMoose

InMoose aims at brining state-of-the-art bioinformatics tools to the Python
ecosystem. The project is still in development, and any contribution is
important!

## How to contribute

You have **many** ways to contribute to the InMoose project, and make it successful by:

* adding features

* reporting and fixing issues (from plain bugs to ergonomic concerns or
  documentation)

* providing interesting examples and use cases (and make the data available)

## Contributing in practice

Any contribution should first be traced in a github issue. Before opening an
issue, please make sure the topic you wish to address is not already covered by
an existing issue.

An issue is the place where a topic is discussed by the community: scope and
implementation ideas for enhancements, reproduction steps and possible
workarounds for bugs, ...

Once an issue is mature enough for implementation, fork the github repo, and
create a dedicated branch for the issue. Once your work is ready for review,
please open a pull request (PR) to merge your branch into the master branch.
Note that your code will be reviewed before merging, which may trigger further
discussions or change requests on your code. Once the review process has
converged towards approval, your PR (*i.e.* your contribution) will be merged in
InMoose. Voilà!

To be accepted, your contribution should be complete:
- the targeted issue should be fully addressed, or limitations clearly
  documented. If needed, an issue may be split into several issues.
- the new behavior should be properly tested. It means adding tests for new
  features, and updating existing tests when modifying an existing features.
  Bugfixes should typically add test cases to make sure that the bug is properly
  fixed.
- the new behavior should be properly documented. This means:
  + updating the documentation upon adding a feature, or when modifying an
    existing feature
  + properly comment the code, especially parts that are not self-explanatory
  + write sensible commit messages
  + update the issue to describe the solution implemented

## Testing InMoose on new data

We believe that the best way of increasing InMoose robustness is to use it on as
many various datasets as possible.

## Writing documentation

InMoose uses
[Sphinx](https://www.sphinx-doc.org/en/master/usage/quickstart.html) to
automatically document the main scripts. You can use the following command to
build the documentation (from the docs folder):

```bash
sphinx-build -b html source/ .
```
