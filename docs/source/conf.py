# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import inmoose

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "InMoose"
copyright = "2022-2025, Maximilien Colange"
author = "Maximilien Colange"

version = inmoose.__version__
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# logo
html_logo = "inmoose.png"
html_favicon = "epigenelogo_favicon.png"

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx_rtd_theme",
    "sphinxcontrib.repl",
]

nitpick_ignore = [("py:class", "optional"), ("py:class", "array-like")]

# Napoleon settings
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Turn on autosummary
autosummary_generate = True
autosummary_generate_overwrite = False

# Add a doi role
extlinks = {
    "doi": ("https://dx.doi.org/%s", "doi:%s"),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# TODO temporary work-around
autodoc_mock_imports = ["edgepy_cpp"]
