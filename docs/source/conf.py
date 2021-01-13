# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from sphinx.builders.latex import LaTeXBuilder

sys.path.insert(0, os.path.abspath('../../../'))

master_doc = 'index'

# -- Project information -----------------------------------------------------

project = 'Time Series Generator'
copyright = '2020, Daniel Kaminski de Souza'
author = 'Daniel Kaminski de Souza'

# The full version, including alpha/beta/rc tags
version = '0.1.5'
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    # 'sphinx.ext.imgconverter',
    'sphinxcontrib.inkscapeconverter',
    # 'sphinxcontrib.rsvgconverter',
    # 'sphinxcontrib.cairosvgconverter',
    'sphinx_autodoc_typehints',
    'sphinx.ext.intersphinx',
    'nbsphinx',
    'recommonmark',
    'IPython.sphinxext.ipython_console_highlighting',
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_use_ivar = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
    'torch': ('http://pytorch.org/docs/master/', None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

html_css_files = [
    'css/theme_modifs.css',
]

# html_context = {
#     "display_github": True, # Integrate GitHub
#     "github_user": "krypton-unite", # Username
#     "github_repo": "time_series_generator", # Repo name
#     "github_version": "master", # Version
#     "conf_py_path": "/docs/source/", # Path in the checkout to the docs root
# }

# LaTeXBuilder.supported_image_types = ['image/png', 'image/pdf','image/svg+xml' ]