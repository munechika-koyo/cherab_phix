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
from pathlib import Path

from cherab.phix import __file__ as phix_file
from cherab.phix import __version__

sys.path.insert(0, os.path.abspath("."))


# -- Project information -----------------------------------------------------

project = "cherab-phix"
author = "Koyo Munechika"
copyright = f"2019-{datetime.now().year}, {author}"
# The short X.Y version.
version = __version__
# The full version, including alpha/beta/rc tags.
release = __version__


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
]

default_role = "obj"

# autodoc config
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_member_order = "groupwise"

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
}
# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "Cherab-PHiX Documentation"

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

# -----------------------------------------------------------------------------
# Source code links
# -----------------------------------------------------------------------------
link_github = True
# You can add build old with link_github = False

if link_github:
    import inspect

    # from packaging.version import parse

    extensions.append("sphinx.ext.linkcode")

    def linkcode_resolve(domain, info):
        """Determine the URL corresponding to Python object."""
        if domain != "py":
            return None

        modname = info["module"]
        fullname = info["fullname"]

        submod = sys.modules.get(modname)
        if submod is None:
            return None

        obj = submod
        for part in fullname.split("."):
            try:
                obj = getattr(obj, part)
            except AttributeError:
                return None

        if inspect.isfunction(obj):
            obj = inspect.unwrap(obj)
        try:
            fn = inspect.getsourcefile(obj)
        except TypeError:
            fn = None
        if not fn or fn.endswith("__init__.py"):
            try:
                fn = inspect.getsourcefile(sys.modules[obj.__module__])
            except (TypeError, AttributeError, KeyError):
                fn = None
        if not fn:
            return None

        try:
            source, lineno = inspect.getsourcelines(obj)
        except (OSError, TypeError):
            lineno = None

        linespec = f"#L{lineno:d}-L{lineno + len(source) - 1:d}" if lineno else ""

        startdir = Path(phix_file).parent.parent.parent
        try:
            fn = os.path.relpath(fn, start=startdir).replace(os.path.sep, "/")
        except ValueError:
            return None

        if not fn.startswith(("cherab")):
            return None

        # version = parse(__version__)
        # tag is temporarily tied to master
        tag = "master"
        # tag = "master" if version.is_devrelease else f"v{version.public}"
        return f"https://github.com/munechika-koyo/cherab_phix/blob/{tag}/{fn}{linespec}"

else:
    extensions.append("sphinx.ext.viewcode")
