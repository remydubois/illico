import os
import sys

# Add project root to sys.path so autodoc can import the package
sys.path.insert(0, os.path.abspath(".."))

project = "illico"
author = "illico contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

# Allow parsing Markdown files with MyST
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# Prefer the Furo theme when available, fall back otherwise
html_theme = "furo"

html_static_path = ["_static"]

# Mock imports that may not be installed in the build environment so autodoc can
# import the package even when heavy scientific deps are missing.
autodoc_mock_imports = [
    "anndata",
    "joblib",
    "loguru",
    "numba",
    "h5py",
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    "statsmodels",
    "tqdm",
    "illico.rust_backend",
]

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
