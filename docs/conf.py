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
from datetime import datetime

from packaging.version import parse

from cherab.phix import __version__

sys.path.insert(0, os.path.abspath("."))


# -- Project information -----------------------------------------------------

project = "cherab-phix"
author = "Koyo Munechika"
copyright = f"2019-{datetime.now().year}, {author}"
version_obj = parse(__version__)
release = str(version_obj)

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx-prompt",
    "sphinx_copybutton",
    "sphinx_design",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_codeautolink",
    "sphinx_github_style",
]

default_role = "obj"

# autodoc config
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_member_order = "bysource"

# autosummary config
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = True
autosummary_ignore_module_all = False

# napoleon config
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_ivar = False

# todo config
todo_include_todos = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# This is added to the end of RST files — a good place to put substitutions to
# be used globally.
rst_epilog = ""
with open("common_links.rst") as cl:
    rst_epilog += cl.read()

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "common_links.rst",
]

# The suffix of source filenames.
source_suffix = ".rst"

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"

# Define the json_url for our version switcher.
json_url = "https://cherab-phix.readthedocs.io/en/latest/_static/switcher.json"
version_match = os.environ.get("READTHEDOCS_VERSION")
# If READTHEDOCS_VERSION doesn't exist, we're not on RTD
# If it is an integer, we're in a PR build and the version isn't correct.
# If it's "latest" → change to "dev" (that's what we want the switcher to call it)
if not version_match or version_match.isdigit() or version_match == "latest":
    # For local development, infer the version to match from the package.
    if version_obj.is_prerelease:
        version_match = "dev"
    else:
        version_match = f"v{release}"
elif version_match == "stable":
    version_match = f"v{release}"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/munechika-koyo/cherab_phix",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/cherab-phix",
            "icon": "fa-solid fa-box",
        },
    ],
    "pygment_light_style": "default",
    "pygment_dark_style": "native",
    "switcher": {
        "json_url": json_url,
        "version_match": version_match,
    },
    "show_version_warning_banner": True,
    "navbar_start": ["navbar-logo", "version-switcher"],
}
# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "Cherab-PHiX"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "raysect": ("http://www.raysect.org", None),
    "cherab": ("https://www.cherab.info", None),
    "plotly": ("https://plotly.com/python-api-reference/", None),
    "calcam": ("https://euratom-software.github.io/calcam/html/", None),
    "contourpy": ("https://contourpy.readthedocs.io/en/latest/", None),
}

intersphinx_timeout = 10

# === NB Sphinx configuration ============================================
nbsphinx_allow_errors = True
# nbsphinx_execute = "never"
nbsphinx_prolog = """
{% set docname = env.doc2path(env.docname, base=None) %}
.. only:: html

    .. role:: raw-html(raw)
        :format: html

    .. note::
        This page was generated from `docs/{{ docname }}`__.
    __ https://github.com/munechika-koyo/cherab_phix/blob/master/docs/{{ docname }}
"""
nbsphinx_thumbnails = {
    "notebooks/observer/fast_camera_LoS": "_static/images/nbsphinx_thumbnails/PHiX_CAD_LoS.png",
    "notebooks/observer/fast_camera_raytracing": "_static/images/plots/rgb.png",
    "notebooks/others/Integral_approx": "_static/images/index-images/example_gallery.png",
    "notebooks/inversion/rtm2svd": "_static/images/index-images/example_gallery.png",
    "notebooks/inversion/tomography_experiment": "_static/images/frames.gif",
}

# === sphinx_github_style configuration ============================================
# get tag name which exists in GitHub
tag = "master" if version_obj.is_devrelease else f"v{version_obj.public}"

# set sphinx_github_style options
top_level = "cherab"
linkcode_blob = tag
linkcode_url = "https://github.com/munechika-koyo/cherab_phix"
linkcode_link_text = "Source"
